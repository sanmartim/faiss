/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroMushroomBody.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_set>

namespace faiss {

namespace {

/// Get raw data pointer from IndexFlat
const float* get_flat_data_mb(const Index* index) {
    auto flat = dynamic_cast<const IndexFlat*>(index);
    FAISS_THROW_IF_NOT_MSG(
            flat, "IndexNeuroMushroomBody requires inner_index to be IndexFlat");
    return flat->get_xb();
}

/// Compute intersection size of two sorted vectors
int intersection_size_mb(const std::vector<int>& a, const std::vector<int>& b) {
    int count = 0;
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] < b[j]) {
            i++;
        } else if (a[i] > b[j]) {
            j++;
        } else {
            count++;
            i++;
            j++;
        }
    }
    return count;
}

} // anonymous namespace

IndexNeuroMushroomBody::IndexNeuroMushroomBody(
        Index* inner,
        int n_kc,
        float kc_sparsity,
        bool own_inner)
        : IndexNeuro(inner, own_inner),
          n_kc(n_kc),
          kc_sparsity(kc_sparsity) {}

void IndexNeuroMushroomBody::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");
    FAISS_THROW_IF_NOT_MSG(d > 0, "dimension must be positive");

    // Compute sparsity level
    k_sparse = std::max(1, static_cast<int>(n_kc * kc_sparsity));

    // Generate sparse random projection PN → KC
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> idx_dist(0, d - 1);
    std::normal_distribution<float> weight_dist(0.0f, 1.0f / std::sqrt(connections_per_kc));

    pn_to_kc_indices.resize(n_kc * connections_per_kc);
    pn_to_kc_weights.resize(n_kc * connections_per_kc);

    for (int kc = 0; kc < n_kc; kc++) {
        std::unordered_set<int> selected;
        while (static_cast<int>(selected.size()) < connections_per_kc) {
            selected.insert(idx_dist(rng));
        }

        int j = 0;
        for (int idx : selected) {
            pn_to_kc_indices[kc * connections_per_kc + j] = idx;
            pn_to_kc_weights[kc * connections_per_kc + j] = weight_dist(rng);
            j++;
        }
    }

    // Initialize MBON weights uniformly
    mbon_weights.assign(n_compartments * n_kc, 1.0f / n_kc);

    // DR-02: If training data provided, adapt threshold
    if (pattern_separation_mode && n > 0 && x != nullptr) {
        adapt_threshold(std::min(n, static_cast<idx_t>(1000)), x);
    }

    kc_codes.clear();
    is_trained = true;
}

void IndexNeuroMushroomBody::compute_kc_code(
        const float* x,
        std::vector<int>& code) const {
    // Step 1: Compute KC activations via sparse projection
    std::vector<std::pair<float, int>> activations(n_kc);
    for (int kc = 0; kc < n_kc; kc++) {
        float act = 0.0f;
        for (int j = 0; j < connections_per_kc; j++) {
            int idx = pn_to_kc_indices[kc * connections_per_kc + j];
            float w = pn_to_kc_weights[kc * connections_per_kc + j];
            act += w * x[idx];
        }
        activations[kc] = {act, kc};
    }

    // Step 2: APL inhibition (winner-take-all)
    std::partial_sort(
            activations.begin(),
            activations.begin() + k_sparse,
            activations.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

    // DR-02: Apply adaptive threshold in pattern separation mode
    int n_active = k_sparse;
    if (pattern_separation_mode && separation_threshold > 0) {
        n_active = 0;
        for (int i = 0; i < k_sparse; i++) {
            if (activations[i].first >= separation_threshold) {
                n_active++;
            } else {
                break;
            }
        }
        n_active = std::max(1, n_active);
    }

    // Extract sorted indices of active KCs
    code.resize(n_active);
    for (int i = 0; i < n_active; i++) {
        code[i] = activations[i].second;
    }
    std::sort(code.begin(), code.end());
}

void IndexNeuroMushroomBody::compute_mbon_output(
        const std::vector<int>& kc_code,
        std::vector<float>& output) const {
    output.assign(n_compartments, 0.0f);

    for (int c = 0; c < n_compartments; c++) {
        const float* weights = mbon_weights.data() + c * n_kc;
        for (int kc : kc_code) {
            output[c] += weights[kc];
        }
    }
}

void IndexNeuroMushroomBody::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained before adding");

    IndexNeuro::add(n, x);

    // Compute KC codes for new vectors
    for (idx_t i = 0; i < n; i++) {
        std::vector<int> code;
        compute_kc_code(x + i * d, code);
        kc_codes.push_back(std::move(code));
    }
}

void IndexNeuroMushroomBody::reset() {
    IndexNeuro::reset();
    kc_codes.clear();
}

void IndexNeuroMushroomBody::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained");

    const float* data = get_flat_data_mb(inner_index);
    idx_t nb = inner_index->ntotal;

    // Resolve parameters
    bool do_rerank = rerank;
    int compartment = -1;

    auto mp = dynamic_cast<const NeuroMushroomBodyParams*>(params);
    if (mp) {
        do_rerank = mp->rerank;
        compartment = mp->compartment;
    }

    bool collect = false;
    if (params) {
        auto nsp = dynamic_cast<const NeuroSearchParameters*>(params);
        if (nsp) {
            collect = nsp->collect_stats;
        }
    }

    int64_t total_calcs = 0;

#pragma omp parallel for reduction(+ : total_calcs)
    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * d;

        // Compute query KC code
        std::vector<int> query_code;
        compute_kc_code(query, query_code);

        // Compute MBON output for query
        std::vector<float> query_mbon;
        compute_mbon_output(query_code, query_mbon);

        // Score all database vectors
        std::vector<std::pair<float, idx_t>> scored(nb);

        for (idx_t i = 0; i < nb; i++) {
            // Compute MBON similarity (dot product of MBON outputs)
            std::vector<float> data_mbon;
            compute_mbon_output(kc_codes[i], data_mbon);

            float score = 0.0f;
            if (compartment >= 0 && compartment < n_compartments) {
                score = query_mbon[compartment] * data_mbon[compartment];
            } else {
                for (int c = 0; c < n_compartments; c++) {
                    score += query_mbon[c] * data_mbon[c];
                }
            }

            // Also add KC overlap bonus (Jaccard-like)
            int overlap = intersection_size_mb(query_code, kc_codes[i]);
            float jaccard = 0.0f;
            int total = static_cast<int>(query_code.size() + kc_codes[i].size()) - overlap;
            if (total > 0) {
                jaccard = static_cast<float>(overlap) / total;
            }

            // Combined score (negative for min-heap behavior)
            scored[i] = {-(score + jaccard), i};
        }

        // Get top candidates
        size_t n_cand = std::min(static_cast<size_t>(k * 3), static_cast<size_t>(nb));
        std::partial_sort(
                scored.begin(),
                scored.begin() + n_cand,
                scored.end());

        // Rerank if requested
        if (do_rerank) {
            for (size_t i = 0; i < n_cand; i++) {
                idx_t idx = scored[i].second;
                float dist;
                if (metric) {
                    dist = metric->distance(query, data + idx * d, d);
                } else {
                    dist = fvec_L2sqr(query, data + idx * d, d);
                }
                scored[i].first = dist;
                total_calcs += d;
            }
            std::partial_sort(
                    scored.begin(),
                    scored.begin() + std::min(n_cand, static_cast<size_t>(k)),
                    scored.end());
        }

        // Output results
        size_t actual_k = std::min(static_cast<size_t>(k), n_cand);
        for (size_t i = 0; i < actual_k; i++) {
            distances[q * k + i] = scored[i].first;
            labels[q * k + i] = scored[i].second;
        }
        for (size_t i = actual_k; i < static_cast<size_t>(k); i++) {
            distances[q * k + i] = std::numeric_limits<float>::max();
            labels[q * k + i] = -1;
        }
    }

    if (collect) {
        last_stats.calculations_performed = total_calcs;
    }
}

void IndexNeuroMushroomBody::feedback(
        idx_t nq,
        const float* queries,
        const float* positives,
        const float* negatives,
        int compartment) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained first");

    int c_start = (compartment >= 0) ? compartment : 0;
    int c_end = (compartment >= 0) ? compartment + 1 : n_compartments;

    for (idx_t q = 0; q < nq; q++) {
        // Compute KC codes
        std::vector<int> query_code, pos_code, neg_code;
        compute_kc_code(queries + q * d, query_code);
        compute_kc_code(positives + q * d, pos_code);
        compute_kc_code(negatives + q * d, neg_code);

        // Find KCs that overlap with query
        std::vector<int> pos_overlap, neg_overlap;
        size_t i = 0, j = 0;

        // Query-positive overlap
        i = 0; j = 0;
        while (i < query_code.size() && j < pos_code.size()) {
            if (query_code[i] < pos_code[j]) {
                i++;
            } else if (query_code[i] > pos_code[j]) {
                j++;
            } else {
                pos_overlap.push_back(query_code[i]);
                i++;
                j++;
            }
        }

        // Query-negative overlap
        i = 0; j = 0;
        while (i < query_code.size() && j < neg_code.size()) {
            if (query_code[i] < neg_code[j]) {
                i++;
            } else if (query_code[i] > neg_code[j]) {
                j++;
            } else {
                neg_overlap.push_back(query_code[i]);
                i++;
                j++;
            }
        }

        // Dopaminergic update: strengthen positive, weaken negative
        for (int c = c_start; c < c_end; c++) {
            float* weights = mbon_weights.data() + c * n_kc;

            // Reward: increase weights for KCs in positive overlap
            for (int kc : pos_overlap) {
                weights[kc] += learning_rate;
            }

            // Punishment: decrease weights for KCs in negative overlap
            for (int kc : neg_overlap) {
                weights[kc] -= learning_rate;
                weights[kc] = std::max(0.0f, weights[kc]);
            }

            // Apply decay
            for (int kc = 0; kc < n_kc; kc++) {
                weights[kc] *= weight_decay;
            }
        }
    }

    feedback_count++;
}

// DR-02: Pattern Separation methods

float IndexNeuroMushroomBody::separation_metric(
        const float* x1,
        const float* x2) const {
    // Compute input correlation
    float input_dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
    for (int i = 0; i < d; i++) {
        input_dot += x1[i] * x2[i];
        norm1 += x1[i] * x1[i];
        norm2 += x2[i] * x2[i];
    }
    float input_corr = 0.0f;
    if (norm1 > 0 && norm2 > 0) {
        input_corr = input_dot / (std::sqrt(norm1) * std::sqrt(norm2));
    }

    // Compute KC code correlation (Jaccard similarity)
    std::vector<int> code1, code2;
    compute_kc_code(x1, code1);
    compute_kc_code(x2, code2);

    int overlap = intersection_size_mb(code1, code2);
    int total = static_cast<int>(code1.size() + code2.size()) - overlap;
    float kc_corr = (total > 0) ? static_cast<float>(overlap) / total : 0.0f;

    // Separation = how much correlation was reduced
    // Higher is better (more decorrelation)
    return std::max(0.0f, input_corr - kc_corr);
}

void IndexNeuroMushroomBody::adapt_threshold(
        idx_t n,
        const float* x,
        float target_separation) {
    if (n < 2) return;

    // Sample pairs and measure separation at different thresholds
    std::mt19937 rng(42);
    std::uniform_int_distribution<idx_t> idx_dist(0, n - 1);

    int n_samples = std::min(static_cast<int>(n * (n - 1) / 2), 100);

    // Try different threshold values
    std::vector<float> thresholds = {0.0f, 0.1f, 0.2f, 0.5f, 1.0f};
    float best_threshold = 0.0f;
    float best_diff = std::numeric_limits<float>::max();

    float original_threshold = separation_threshold;

    for (float thresh : thresholds) {
        separation_threshold = thresh;

        // Measure average separation
        double total_sep = 0.0;
        for (int s = 0; s < n_samples; s++) {
            idx_t i = idx_dist(rng);
            idx_t j = idx_dist(rng);
            while (j == i) j = idx_dist(rng);

            total_sep += separation_metric(x + i * d, x + j * d);
        }
        float avg_sep = static_cast<float>(total_sep / n_samples);

        float diff = std::abs(avg_sep - target_separation);
        if (diff < best_diff) {
            best_diff = diff;
            best_threshold = thresh;
        }
    }

    separation_threshold = best_threshold;
}

} // namespace faiss
