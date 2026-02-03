/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroElimination.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace faiss {

IndexNeuroElimination::IndexNeuroElimination(
        Index* inner,
        NeuroEliminationStrategy strategy,
        bool own_inner)
        : IndexNeuro(inner, own_inner), strategy(strategy) {}

void IndexNeuroElimination::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");
    inner_index->train(n, x);
    is_trained = inner_index->is_trained;

    // ED-03: compute per-column variance to determine optimal ordering
    if (strategy == NEURO_VARIANCE_ORDER && n > 0 && x != nullptr) {
        idx_t n_sample = std::max(
                idx_t(1),
                static_cast<idx_t>(n * sample_fraction));
        if (n_sample > n)
            n_sample = n;

        // Compute per-column variance from sample
        std::vector<double> col_mean(d, 0.0);
        std::vector<double> col_var(d, 0.0);

        for (idx_t i = 0; i < n_sample; i++) {
            for (int j = 0; j < d; j++) {
                col_mean[j] += x[i * d + j];
            }
        }
        for (int j = 0; j < d; j++) {
            col_mean[j] /= n_sample;
        }
        for (idx_t i = 0; i < n_sample; i++) {
            for (int j = 0; j < d; j++) {
                double diff = x[i * d + j] - col_mean[j];
                col_var[j] += diff * diff;
            }
        }
        for (int j = 0; j < d; j++) {
            col_var[j] /= n_sample;
        }

        // Sort columns by variance descending (highest variance first)
        variance_column_order.resize(d);
        std::iota(variance_column_order.begin(), variance_column_order.end(), 0);
        std::sort(
                variance_column_order.begin(),
                variance_column_order.end(),
                [&col_var](int a, int b) {
                    return col_var[a] > col_var[b];
                });
    }
}

namespace {

/// Get the raw float data pointer from the inner index.
/// Requires inner to be an IndexFlat.
const float* get_flat_data(const Index* index) {
    auto flat = dynamic_cast<const IndexFlat*>(index);
    FAISS_THROW_IF_NOT_MSG(
            flat, "IndexNeuroElimination requires inner_index to be IndexFlat");
    return flat->get_xb();
}

/// Build default column order (reversed: d-1, d-2, ..., 0)
std::vector<int> default_column_order(int d) {
    std::vector<int> order(d);
    for (int i = 0; i < d; i++) {
        order[i] = d - 1 - i;
    }
    return order;
}

/// Compute adaptive cutoff based on dispersion of single-column distances
/// V2: Added minimum cutoff floor to prevent over-aggressive elimination
float adaptive_cutoff(
        const std::vector<float>& col_dists,
        size_t n_candidates,
        float dispersion_low,
        float dispersion_high,
        float cutoff_low,
        float cutoff_high) {
    if (n_candidates <= 1) {
        return 1.0f;
    }

    // Compute mean and std of single-column distances
    double sum = 0, sum_sq = 0;
    for (size_t i = 0; i < n_candidates; i++) {
        double v = col_dists[i];
        sum += v;
        sum_sq += v * v;
    }
    double mean = sum / n_candidates;
    double var = sum_sq / n_candidates - mean * mean;
    if (var < 0)
        var = 0;
    double std_dev = std::sqrt(var);

    if (mean < 1e-10) {
        // All distances nearly zero, no discriminative power
        return cutoff_low; // pass most candidates
    }

    float dispersion = static_cast<float>(std_dev / mean);

    // Map dispersion to cutoff via linear interpolation
    float cutoff;
    if (dispersion <= dispersion_low) {
        cutoff = cutoff_low;
    } else if (dispersion >= dispersion_high) {
        cutoff = cutoff_high;
    } else {
        float t = (dispersion - dispersion_low) /
                (dispersion_high - dispersion_low);
        cutoff = cutoff_low + t * (cutoff_high - cutoff_low);
    }

    // V2: Enforce minimum cutoff floor (never eliminate more than 40%)
    // cutoff_high serves as the floor (default 0.6)
    return std::max(cutoff, cutoff_high);
}

/// Search for a single query using progressive elimination.
///
/// The algorithm accumulates partial L2 distances across columns.
/// After processing each column, it adds the squared difference to
/// each candidate's running total, then eliminates candidates whose
/// accumulated partial distance exceeds the cutoff threshold.
void search_one_elimination(
        const float* query,
        const float* data,
        idx_t ntotal,
        int d,
        idx_t k,
        float* out_distances,
        idx_t* out_labels,
        const std::vector<int>& col_order,
        float cutoff_percentile,
        int min_cand,
        NeuroEliminationStrategy strategy,
        float dispersion_low,
        float dispersion_high,
        float cutoff_low_disp,
        float cutoff_high_disp,
        NeuroSearchStats* stats,
        float confidence_threshold = 0.4f,
        int max_accumulated_columns = 3) {
    int effective_min = min_cand > 0 ? min_cand : static_cast<int>(k * 2);
    if (effective_min < k) {
        effective_min = static_cast<int>(k);
    }

    // Initialize candidate set with accumulated partial distances
    size_t n_cand = static_cast<size_t>(ntotal);
    std::vector<idx_t> candidates(n_cand);
    std::iota(candidates.begin(), candidates.end(), 0);
    std::vector<float> partial_dists(n_cand, 0.0f);

    int64_t calcs = 0;
    int cols_used = 0;
    int n_cols = static_cast<int>(col_order.size());
    int deferred_count = 0; // ED-04: tracks accumulated columns without elimination

    // Track which columns have been processed (for remaining-col final pass)
    std::vector<bool> col_processed(d, false);

    // Progressive elimination: iterate columns, accumulate partial distances
    for (int ci = 0; ci < n_cols; ci++) {
        int col = col_order[ci];

        if (static_cast<int>(n_cand) <= effective_min) {
            break;
        }

        col_processed[col] = true;

        // Compute this column's squared differences
        float qval = query[col];
        std::vector<float> col_diffs(n_cand);
        for (size_t i = 0; i < n_cand; i++) {
            float diff = qval - data[candidates[i] * d + col];
            col_diffs[i] = diff * diff;
            partial_dists[i] += col_diffs[i];
        }
        calcs += n_cand;
        cols_used++;

        // Determine cutoff for this round
        float cutoff;
        if (strategy == NEURO_ADAPTIVE_DISPERSION) {
            // Use column-level dispersion to judge discriminative power
            cutoff = adaptive_cutoff(
                    col_diffs,
                    n_cand,
                    dispersion_low,
                    dispersion_high,
                    cutoff_low_disp,
                    cutoff_high_disp);
        } else if (strategy == NEURO_UNCERTAINTY_DEFERRED) {
            // Compute dispersion of this column's distances
            float col_disp = 0.0f;
            if (n_cand > 1) {
                double sum = 0, sum_sq = 0;
                for (size_t i = 0; i < n_cand; i++) {
                    double v = col_diffs[i];
                    sum += v;
                    sum_sq += v * v;
                }
                double mean = sum / n_cand;
                double var = sum_sq / n_cand - mean * mean;
                if (var < 0) var = 0;
                if (mean > 1e-10) {
                    col_disp = static_cast<float>(std::sqrt(var) / mean);
                }
            }

            // Low dispersion = low confidence = defer
            if (col_disp < confidence_threshold) {
                deferred_count++;
                if (deferred_count < max_accumulated_columns) {
                    continue; // Skip elimination, just accumulate
                }
                // Forced elimination after max accumulated
                deferred_count = 0;
            } else {
                deferred_count = 0;
            }
            cutoff = cutoff_percentile;
        } else {
            cutoff = cutoff_percentile;
        }

        // Determine how many to keep
        size_t n_keep =
                std::max(static_cast<size_t>(effective_min),
                         static_cast<size_t>(std::ceil(n_cand * cutoff)));
        if (n_keep >= n_cand) {
            continue; // No elimination this round
        }

        // Partial sort by accumulated distance: keep the n_keep smallest
        std::vector<size_t> order(n_cand);
        std::iota(order.begin(), order.end(), 0);
        std::nth_element(
                order.begin(),
                order.begin() + n_keep,
                order.end(),
                [&partial_dists](size_t a, size_t b) {
                    return partial_dists[a] < partial_dists[b];
                });

        // Compact survivors
        std::vector<idx_t> new_candidates(n_keep);
        std::vector<float> new_partial(n_keep);
        for (size_t i = 0; i < n_keep; i++) {
            new_candidates[i] = candidates[order[i]];
            new_partial[i] = partial_dists[order[i]];
        }
        candidates = std::move(new_candidates);
        partial_dists = std::move(new_partial);
        n_cand = candidates.size();
    }

    // Compute remaining columns for survivors (columns not yet processed)
    size_t n_survivors = candidates.size();
    std::vector<float> full_dists(n_survivors);
    for (size_t i = 0; i < n_survivors; i++) {
        float remaining = 0.0f;
        const float* vec = data + candidates[i] * d;
        for (int c = 0; c < d; c++) {
            if (!col_processed[c]) {
                float diff = query[c] - vec[c];
                remaining += diff * diff;
            }
        }
        full_dists[i] = partial_dists[i] + remaining;
    }
    // Count remaining column computations
    int remaining_cols = 0;
    for (int c = 0; c < d; c++) {
        if (!col_processed[c])
            remaining_cols++;
    }
    calcs += n_survivors * remaining_cols;

    // Find top-k
    std::vector<size_t> top_order(n_survivors);
    std::iota(top_order.begin(), top_order.end(), 0);

    size_t actual_k = std::min(static_cast<size_t>(k), n_survivors);
    std::partial_sort(
            top_order.begin(),
            top_order.begin() + actual_k,
            top_order.end(),
            [&full_dists](size_t a, size_t b) {
                return full_dists[a] < full_dists[b];
            });

    for (size_t i = 0; i < actual_k; i++) {
        out_distances[i] = full_dists[top_order[i]];
        out_labels[i] = candidates[top_order[i]];
    }
    // Fill remaining with -1 if k > n_survivors
    for (size_t i = actual_k; i < static_cast<size_t>(k); i++) {
        out_distances[i] = std::numeric_limits<float>::max();
        out_labels[i] = -1;
    }

    if (stats) {
        stats->calculations_performed = calcs;
        stats->columns_used = cols_used;
    }
}

} // anonymous namespace

void IndexNeuroElimination::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");

    const float* data = get_flat_data(inner_index);
    idx_t nb = inner_index->ntotal;

    // Resolve column order
    std::vector<int> col_order;
    float cutoff = cutoff_percentile;
    int min_cand = min_candidates;
    float disp_lo = dispersion_low;
    float disp_hi = dispersion_high;
    float cut_lo = cutoff_low_dispersion;
    float cut_hi = cutoff_high_dispersion;

    // ED-04 params
    float conf_thresh = confidence_threshold;
    int max_accum = max_accumulated_columns;

    // Allow overriding from params
    auto neuro_params = dynamic_cast<const NeuroEliminationParams*>(params);
    if (neuro_params) {
        if (!neuro_params->column_order.empty()) {
            col_order = neuro_params->column_order;
        }
        cutoff = neuro_params->cutoff_percentile;
        min_cand = neuro_params->min_candidates;
        disp_lo = neuro_params->dispersion_low;
        disp_hi = neuro_params->dispersion_high;
        cut_lo = neuro_params->cutoff_low_dispersion;
        cut_hi = neuro_params->cutoff_high_dispersion;
        conf_thresh = neuro_params->confidence_threshold;
        max_accum = neuro_params->max_accumulated_columns;
    }

    if (col_order.empty()) {
        if (strategy == NEURO_VARIANCE_ORDER &&
            !variance_column_order.empty()) {
            col_order = variance_column_order;
        } else if (!column_order.empty()) {
            col_order = column_order;
        } else {
            col_order = default_column_order(d);
        }
    }

    // Map strategy to the elimination behavior used in the inner loop
    NeuroEliminationStrategy effective_strategy = strategy;
    if (strategy == NEURO_VARIANCE_ORDER) {
        // ED-03 uses fixed cutoff but with variance-sorted columns
        effective_strategy = NEURO_FIXED;
    }
    // NEURO_UNCERTAINTY_DEFERRED stays as-is; handled in inner loop

    // Determine whether to collect stats
    bool collect = false;
    auto neuro_search = dynamic_cast<const NeuroSearchParameters*>(params);
    if (neuro_search) {
        collect = neuro_search->collect_stats;
    }

    // Reset aggregate stats
    last_stats = NeuroSearchStats{};

#pragma omp parallel for if (n > 1)
    for (idx_t i = 0; i < n; i++) {
        NeuroSearchStats query_stats;
        search_one_elimination(
                x + i * d,
                data,
                nb,
                d,
                k,
                distances + i * k,
                labels + i * k,
                col_order,
                cutoff,
                min_cand,
                effective_strategy,
                disp_lo,
                disp_hi,
                cut_lo,
                cut_hi,
                collect ? &query_stats : nullptr,
                conf_thresh,
                max_accum);

        if (collect) {
#pragma omp critical
            {
                last_stats.calculations_performed +=
                        query_stats.calculations_performed;
                last_stats.columns_used =
                        std::max(last_stats.columns_used,
                                 query_stats.columns_used);
            }
        }
    }
}

} // namespace faiss
