/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroDropoutEnsemble.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace faiss {

IndexNeuroDropoutEnsemble::IndexNeuroDropoutEnsemble(
        Index* inner,
        bool own_inner)
        : IndexNeuro(inner, own_inner) {}

void IndexNeuroDropoutEnsemble::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");
    inner_index->train(n, x);
    is_trained = inner_index->is_trained;
}

namespace {

/// Get the raw float data pointer from the inner index.
const float* get_flat_data_de(const Index* index) {
    auto flat = dynamic_cast<const IndexFlat*>(index);
    FAISS_THROW_IF_NOT_MSG(
            flat,
            "IndexNeuroDropoutEnsemble requires inner_index to be IndexFlat");
    return flat->get_xb();
}

/// Generate dropout masks based on the chosen mode.
/// Each mask is a vector of active column indices.
std::vector<std::vector<int>> generate_masks(
        int d,
        int num_views,
        float dropout_rate,
        NeuroDropoutMode mode,
        unsigned seed) {
    int active_dims = std::max(1, static_cast<int>(d * (1.0f - dropout_rate)));
    std::vector<std::vector<int>> masks(num_views);
    std::mt19937 rng(seed);

    switch (mode) {
        case NEURO_DROPOUT_RANDOM: {
            // Independent random masks per view
            std::vector<int> all_dims(d);
            std::iota(all_dims.begin(), all_dims.end(), 0);
            for (int v = 0; v < num_views; v++) {
                std::vector<int> shuffled = all_dims;
                std::shuffle(shuffled.begin(), shuffled.end(), rng);
                masks[v].assign(
                        shuffled.begin(), shuffled.begin() + active_dims);
                std::sort(masks[v].begin(), masks[v].end());
            }
            break;
        }

        case NEURO_DROPOUT_COMPLEMENTARY: {
            // Distribute dimensions across views to minimize overlap
            // and guarantee each dimension appears in at least one view.
            // Round-robin assignment, then fill remaining slots randomly.
            std::vector<std::vector<int>> view_dims(num_views);

            // Round-robin: assign each dimension to views
            for (int j = 0; j < d; j++) {
                view_dims[j % num_views].push_back(j);
            }

            // Each view needs active_dims columns; fill from other dims
            for (int v = 0; v < num_views; v++) {
                if (static_cast<int>(view_dims[v].size()) >= active_dims) {
                    // Trim to active_dims
                    view_dims[v].resize(active_dims);
                } else {
                    // Need more dims: add from pool not yet in this view
                    std::unordered_set<int> already(
                            view_dims[v].begin(), view_dims[v].end());
                    std::vector<int> candidates;
                    for (int j = 0; j < d; j++) {
                        if (already.find(j) == already.end()) {
                            candidates.push_back(j);
                        }
                    }
                    std::shuffle(
                            candidates.begin(), candidates.end(), rng);
                    int need = active_dims -
                            static_cast<int>(view_dims[v].size());
                    for (int i = 0; i < need && i < static_cast<int>(candidates.size()); i++) {
                        view_dims[v].push_back(candidates[i]);
                    }
                }
                std::sort(view_dims[v].begin(), view_dims[v].end());
                masks[v] = view_dims[v];
            }
            break;
        }

        case NEURO_DROPOUT_STRUCTURED: {
            // Semantic grouping: divide dimensions into blocks
            // View 0: first half, View 1: second half, etc.
            // For more views, use overlapping blocks.
            int block_size = std::max(1, d / num_views);
            for (int v = 0; v < num_views; v++) {
                int start = (v * d) / num_views;
                // Use active_dims centered around this block
                std::vector<int> dims;
                for (int j = start;
                     j < d && static_cast<int>(dims.size()) < active_dims;
                     j++) {
                    dims.push_back(j);
                }
                // Wrap around if needed
                for (int j = 0;
                     static_cast<int>(dims.size()) < active_dims && j < start;
                     j++) {
                    dims.push_back(j);
                }
                std::sort(dims.begin(), dims.end());
                masks[v] = dims;
            }
            break;
        }

        case NEURO_DROPOUT_ADVERSARIAL: {
            // Each view excludes the columns that were most important
            // in previous views. First view is random; subsequent views
            // exclude top-variance columns from previous masks.
            std::vector<int> all_dims(d);
            std::iota(all_dims.begin(), all_dims.end(), 0);

            // Track column usage frequency
            std::vector<int> usage_count(d, 0);

            for (int v = 0; v < num_views; v++) {
                if (v == 0) {
                    // First view: random
                    std::vector<int> shuffled = all_dims;
                    std::shuffle(shuffled.begin(), shuffled.end(), rng);
                    masks[v].assign(
                            shuffled.begin(), shuffled.begin() + active_dims);
                } else {
                    // Sort by usage count (ascending) - prefer least-used
                    std::vector<int> sorted_dims = all_dims;
                    std::sort(
                            sorted_dims.begin(),
                            sorted_dims.end(),
                            [&usage_count](int a, int b) {
                                return usage_count[a] < usage_count[b];
                            });
                    masks[v].assign(
                            sorted_dims.begin(),
                            sorted_dims.begin() + active_dims);
                }
                // Update usage counts
                for (int col : masks[v]) {
                    usage_count[col]++;
                }
                std::sort(masks[v].begin(), masks[v].end());
            }
            break;
        }
    }

    return masks;
}

/// Compute masked L2 distance (only on active dimensions),
/// renormalized to full dimension scale.
float masked_l2(
        const float* x,
        const float* y,
        int d,
        const std::vector<int>& active_dims) {
    float sum = 0.0f;
    for (int col : active_dims) {
        float diff = x[col] - y[col];
        sum += diff * diff;
    }
    // Renormalize: scale up as if all dimensions were present
    int n_active = static_cast<int>(active_dims.size());
    if (n_active > 0 && n_active < d) {
        sum *= static_cast<float>(d) / n_active;
    }
    return sum;
}

/// Integrate results from multiple views using VOTING.
/// Returns top-k candidates by appearance count.
void integrate_voting(
        const std::vector<std::vector<idx_t>>& view_labels,
        const std::vector<std::vector<float>>& view_dists,
        int top_k_per_view,
        idx_t k,
        float* out_distances,
        idx_t* out_labels,
        const float* query,
        const float* data,
        int d) {
    // Count appearances
    std::unordered_map<idx_t, int> vote_count;
    std::unordered_map<idx_t, float> best_dist;

    for (size_t v = 0; v < view_labels.size(); v++) {
        int n = std::min(top_k_per_view,
                         static_cast<int>(view_labels[v].size()));
        for (int i = 0; i < n; i++) {
            idx_t id = view_labels[v][i];
            if (id < 0) continue;
            vote_count[id]++;
            float d_val = view_dists[v][i];
            if (best_dist.find(id) == best_dist.end() ||
                d_val < best_dist[id]) {
                best_dist[id] = d_val;
            }
        }
    }

    // Sort by vote count desc, then by best distance asc
    std::vector<idx_t> candidates;
    candidates.reserve(vote_count.size());
    for (auto& p : vote_count) {
        candidates.push_back(p.first);
    }
    std::sort(candidates.begin(), candidates.end(),
              [&vote_count, &best_dist](idx_t a, idx_t b) {
                  if (vote_count[a] != vote_count[b])
                      return vote_count[a] > vote_count[b];
                  return best_dist[a] < best_dist[b];
              });

    // Compute full distances for top candidates and return top-k
    size_t n_cand = std::min(candidates.size(), static_cast<size_t>(k * 3));
    std::vector<std::pair<float, idx_t>> scored(n_cand);
    for (size_t i = 0; i < n_cand; i++) {
        scored[i] = {fvec_L2sqr(query, data + candidates[i] * d, d),
                     candidates[i]};
    }
    std::partial_sort(
            scored.begin(),
            scored.begin() + std::min(n_cand, static_cast<size_t>(k)),
            scored.end());

    size_t actual_k = std::min(static_cast<size_t>(k), n_cand);
    for (size_t i = 0; i < actual_k; i++) {
        out_distances[i] = scored[i].first;
        out_labels[i] = scored[i].second;
    }
    for (size_t i = actual_k; i < static_cast<size_t>(k); i++) {
        out_distances[i] = std::numeric_limits<float>::max();
        out_labels[i] = -1;
    }
}

/// Integrate results from multiple views using BORDA count.
void integrate_borda(
        const std::vector<std::vector<idx_t>>& view_labels,
        const std::vector<std::vector<float>>& view_dists,
        int top_k_per_view,
        idx_t k,
        float* out_distances,
        idx_t* out_labels,
        const float* query,
        const float* data,
        int d) {
    // Borda: sum of ranks (lower = better). Absent = max rank.
    std::unordered_map<idx_t, float> borda_scores;
    int num_views = static_cast<int>(view_labels.size());

    for (int v = 0; v < num_views; v++) {
        int n = std::min(top_k_per_view,
                         static_cast<int>(view_labels[v].size()));
        for (int i = 0; i < n; i++) {
            idx_t id = view_labels[v][i];
            if (id < 0) continue;
            borda_scores[id] += static_cast<float>(i); // rank = position
        }
    }

    // Sort by borda score ascending
    std::vector<idx_t> candidates;
    candidates.reserve(borda_scores.size());
    for (auto& p : borda_scores) {
        candidates.push_back(p.first);
    }
    std::sort(candidates.begin(), candidates.end(),
              [&borda_scores](idx_t a, idx_t b) {
                  return borda_scores[a] < borda_scores[b];
              });

    // Compute full distances for top candidates
    size_t n_cand = std::min(candidates.size(), static_cast<size_t>(k * 3));
    std::vector<std::pair<float, idx_t>> scored(n_cand);
    for (size_t i = 0; i < n_cand; i++) {
        scored[i] = {fvec_L2sqr(query, data + candidates[i] * d, d),
                     candidates[i]};
    }
    std::partial_sort(
            scored.begin(),
            scored.begin() + std::min(n_cand, static_cast<size_t>(k)),
            scored.end());

    size_t actual_k = std::min(static_cast<size_t>(k), n_cand);
    for (size_t i = 0; i < actual_k; i++) {
        out_distances[i] = scored[i].first;
        out_labels[i] = scored[i].second;
    }
    for (size_t i = actual_k; i < static_cast<size_t>(k); i++) {
        out_distances[i] = std::numeric_limits<float>::max();
        out_labels[i] = -1;
    }
}

/// Integrate using mean distance across views.
void integrate_mean_dist(
        const std::vector<std::vector<idx_t>>& view_labels,
        const std::vector<std::vector<float>>& view_dists,
        int top_k_per_view,
        idx_t k,
        float* out_distances,
        idx_t* out_labels,
        const float* query,
        const float* data,
        int d) {
    std::unordered_map<idx_t, float> sum_dist;
    std::unordered_map<idx_t, int> count;

    for (size_t v = 0; v < view_labels.size(); v++) {
        int n = std::min(top_k_per_view,
                         static_cast<int>(view_labels[v].size()));
        for (int i = 0; i < n; i++) {
            idx_t id = view_labels[v][i];
            if (id < 0) continue;
            sum_dist[id] += view_dists[v][i];
            count[id]++;
        }
    }

    // Sort by mean distance
    std::vector<idx_t> candidates;
    candidates.reserve(sum_dist.size());
    for (auto& p : sum_dist) {
        candidates.push_back(p.first);
    }
    std::sort(candidates.begin(), candidates.end(),
              [&sum_dist, &count](idx_t a, idx_t b) {
                  return (sum_dist[a] / count[a]) < (sum_dist[b] / count[b]);
              });

    // Compute full distances for top candidates
    size_t n_cand = std::min(candidates.size(), static_cast<size_t>(k * 3));
    std::vector<std::pair<float, idx_t>> scored(n_cand);
    for (size_t i = 0; i < n_cand; i++) {
        scored[i] = {fvec_L2sqr(query, data + candidates[i] * d, d),
                     candidates[i]};
    }
    std::partial_sort(
            scored.begin(),
            scored.begin() + std::min(n_cand, static_cast<size_t>(k)),
            scored.end());

    size_t actual_k = std::min(static_cast<size_t>(k), n_cand);
    for (size_t i = 0; i < actual_k; i++) {
        out_distances[i] = scored[i].first;
        out_labels[i] = scored[i].second;
    }
    for (size_t i = actual_k; i < static_cast<size_t>(k); i++) {
        out_distances[i] = std::numeric_limits<float>::max();
        out_labels[i] = -1;
    }
}

/// Integrate using full rerank: union of all candidates, full L2.
void integrate_full_rerank(
        const std::vector<std::vector<idx_t>>& view_labels,
        const std::vector<std::vector<float>>& /*view_dists*/,
        int top_k_per_view,
        idx_t k,
        float* out_distances,
        idx_t* out_labels,
        const float* query,
        const float* data,
        int d) {
    // Collect union of all candidates
    std::unordered_set<idx_t> union_set;
    for (size_t v = 0; v < view_labels.size(); v++) {
        int n = std::min(top_k_per_view,
                         static_cast<int>(view_labels[v].size()));
        for (int i = 0; i < n; i++) {
            if (view_labels[v][i] >= 0) {
                union_set.insert(view_labels[v][i]);
            }
        }
    }

    std::vector<idx_t> candidates(union_set.begin(), union_set.end());
    size_t n_cand = candidates.size();

    // Full L2 for all candidates
    std::vector<std::pair<float, idx_t>> scored(n_cand);
    for (size_t i = 0; i < n_cand; i++) {
        scored[i] = {fvec_L2sqr(query, data + candidates[i] * d, d),
                     candidates[i]};
    }

    size_t actual_k = std::min(static_cast<size_t>(k), n_cand);
    std::partial_sort(
            scored.begin(),
            scored.begin() + actual_k,
            scored.end());

    for (size_t i = 0; i < actual_k; i++) {
        out_distances[i] = scored[i].first;
        out_labels[i] = scored[i].second;
    }
    for (size_t i = actual_k; i < static_cast<size_t>(k); i++) {
        out_distances[i] = std::numeric_limits<float>::max();
        out_labels[i] = -1;
    }
}

/// Search a single query with dropout ensemble.
void search_one_dropout(
        const float* query,
        const float* data,
        idx_t ntotal,
        int d,
        idx_t k,
        float* out_distances,
        idx_t* out_labels,
        int num_views,
        float dropout_rate,
        NeuroDropoutMode dropout_mode,
        NeuroIntegrationMethod integration,
        int top_k_per_view,
        unsigned seed) {
    int effective_topk = top_k_per_view > 0 ? top_k_per_view
                                            : static_cast<int>(k * 2);

    // Generate masks
    auto masks = generate_masks(d, num_views, dropout_rate, dropout_mode, seed);

    // Per-view search
    std::vector<std::vector<idx_t>> view_labels(num_views);
    std::vector<std::vector<float>> view_dists(num_views);

    for (int v = 0; v < num_views; v++) {
        const auto& active = masks[v];
        int n_active = static_cast<int>(active.size());
        if (n_active == 0) continue;

        // Compute masked distances for all candidates
        std::vector<std::pair<float, idx_t>> scores(ntotal);
        for (idx_t i = 0; i < ntotal; i++) {
            scores[i] = {masked_l2(query, data + i * d, d, active), i};
        }

        // Get top-k per view
        int n_keep = std::min(effective_topk, static_cast<int>(ntotal));
        std::partial_sort(
                scores.begin(), scores.begin() + n_keep, scores.end());

        view_labels[v].resize(n_keep);
        view_dists[v].resize(n_keep);
        for (int i = 0; i < n_keep; i++) {
            view_labels[v][i] = scores[i].second;
            view_dists[v][i] = scores[i].first;
        }
    }

    // Integrate results
    switch (integration) {
        case NEURO_INTEGRATE_VOTING:
            integrate_voting(
                    view_labels, view_dists, effective_topk, k,
                    out_distances, out_labels, query, data, d);
            break;
        case NEURO_INTEGRATE_BORDA:
            integrate_borda(
                    view_labels, view_dists, effective_topk, k,
                    out_distances, out_labels, query, data, d);
            break;
        case NEURO_INTEGRATE_MEAN_DIST:
            integrate_mean_dist(
                    view_labels, view_dists, effective_topk, k,
                    out_distances, out_labels, query, data, d);
            break;
        case NEURO_INTEGRATE_FULL_RERANK:
            integrate_full_rerank(
                    view_labels, view_dists, effective_topk, k,
                    out_distances, out_labels, query, data, d);
            break;
    }
}

} // anonymous namespace

void IndexNeuroDropoutEnsemble::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");

    const float* data = get_flat_data_de(inner_index);
    idx_t nb = inner_index->ntotal;

    // Resolve parameters
    int nv = num_views;
    float dr = dropout_rate;
    NeuroDropoutMode dm = dropout_mode;
    NeuroIntegrationMethod im = integration;
    int tkpv = top_k_per_view;

    auto dp = dynamic_cast<const NeuroDropoutParams*>(params);
    if (dp) {
        nv = dp->num_views;
        dr = dp->dropout_rate;
        dm = dp->dropout_mode;
        im = dp->integration;
        tkpv = dp->top_k_per_view;
    }

#pragma omp parallel for if (n > 1)
    for (idx_t i = 0; i < n; i++) {
        // Use query index as part of seed for reproducibility
        unsigned seed = static_cast<unsigned>(i * 1000 + 12345);
        search_one_dropout(
                x + i * d,
                data,
                nb,
                d,
                k,
                distances + i * k,
                labels + i * k,
                nv,
                dr,
                dm,
                im,
                tkpv,
                seed);
    }
}

} // namespace faiss
