/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroFlyHash.h>
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
const float* get_flat_data_fly(const Index* index) {
    auto flat = dynamic_cast<const IndexFlat*>(index);
    FAISS_THROW_IF_NOT_MSG(
            flat, "IndexNeuroFlyHash requires inner_index to be IndexFlat");
    return flat->get_xb();
}

/// Compute intersection size of two sorted ranges
int intersection_size_range(
        const int* a_begin, const int* a_end,
        const int* b_begin, const int* b_end) {
    int count = 0;
    const int* ai = a_begin;
    const int* bi = b_begin;
    while (ai < a_end && bi < b_end) {
        if (*ai < *bi) {
            ai++;
        } else if (*ai > *bi) {
            bi++;
        } else {
            count++;
            ai++;
            bi++;
        }
    }
    return count;
}

/// Compute union size of two sorted ranges
int union_size_range(
        const int* a_begin, const int* a_end,
        const int* b_begin, const int* b_end) {
    int count = 0;
    const int* ai = a_begin;
    const int* bi = b_begin;
    while (ai < a_end && bi < b_end) {
        if (*ai < *bi) {
            count++;
            ai++;
        } else if (*ai > *bi) {
            count++;
            bi++;
        } else {
            count++;
            ai++;
            bi++;
        }
    }
    count += static_cast<int>(a_end - ai);
    count += static_cast<int>(b_end - bi);
    return count;
}

/// Compute intersection size of two sorted vectors
int intersection_size(const std::vector<int>& a, const std::vector<int>& b) {
    return intersection_size_range(
            a.data(), a.data() + a.size(),
            b.data(), b.data() + b.size());
}

/// Compute union size of two sorted vectors
int union_size(const std::vector<int>& a, const std::vector<int>& b) {
    return union_size_range(
            a.data(), a.data() + a.size(),
            b.data(), b.data() + b.size());
}

} // anonymous namespace

/*************************************************************
 * IndexNeuroFlyHash (HS-02)
 *************************************************************/

IndexNeuroFlyHash::IndexNeuroFlyHash(
        Index* inner,
        int expansion_factor,
        float sparsity,
        bool own_inner)
        : IndexNeuro(inner, own_inner),
          expansion_factor(expansion_factor),
          sparsity(sparsity) {}

void IndexNeuroFlyHash::train(idx_t /*n*/, const float* /*x*/) {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");
    FAISS_THROW_IF_NOT_MSG(d > 0, "dimension must be positive");

    // Compute expanded dimension and sparsity level
    m = expansion_factor * d;
    k_sparse = std::max(1, static_cast<int>(m * sparsity));

    // Generate sparse random projection
    // Each expanded neuron receives input from connections_per_neuron random inputs
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> idx_dist(0, d - 1);
    std::normal_distribution<float> weight_dist(0.0f, 1.0f / std::sqrt(connections_per_neuron));

    projection_indices.resize(m * connections_per_neuron);
    projection_weights.resize(m * connections_per_neuron);

    for (int i = 0; i < m; i++) {
        // Select random input indices for this expanded neuron
        std::unordered_set<int> selected;
        while (static_cast<int>(selected.size()) < connections_per_neuron) {
            selected.insert(idx_dist(rng));
        }

        int j = 0;
        for (int idx : selected) {
            projection_indices[i * connections_per_neuron + j] = idx;
            projection_weights[i * connections_per_neuron + j] = weight_dist(rng);
            j++;
        }
    }

    codes_data.clear();
    codes_offsets.clear();
    codes_offsets.push_back(0);
    is_trained = true;
}

void IndexNeuroFlyHash::compute_code(
        const float* x,
        std::vector<int>& code) const {
    // Step 1: Compute expanded activations
    std::vector<std::pair<float, int>> activations(m);
    for (int i = 0; i < m; i++) {
        float act = 0.0f;
        for (int j = 0; j < connections_per_neuron; j++) {
            int idx = projection_indices[i * connections_per_neuron + j];
            float w = projection_weights[i * connections_per_neuron + j];
            act += w * x[idx];
        }
        activations[i] = {act, i};
    }

    // Step 2: Winner-take-all - keep top k_sparse
    std::partial_sort(
            activations.begin(),
            activations.begin() + k_sparse,
            activations.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

    // Step 3: Extract sorted indices of active neurons
    code.resize(k_sparse);
    for (int i = 0; i < k_sparse; i++) {
        code[i] = activations[i].second;
    }
    std::sort(code.begin(), code.end());
}

float IndexNeuroFlyHash::jaccard_similarity(
        const std::vector<int>& a,
        const std::vector<int>& b) const {
    int inter = intersection_size(a, b);
    int uni = union_size(a, b);
    if (uni == 0) return 1.0f;
    return static_cast<float>(inter) / uni;
}

void IndexNeuroFlyHash::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained before adding");

    IndexNeuro::add(n, x);

    // Compute codes for new vectors
    for (idx_t i = 0; i < n; i++) {
        std::vector<int> code;
        compute_code(x + i * d, code);
        // Append to flattened storage
        codes_data.insert(codes_data.end(), code.begin(), code.end());
        codes_offsets.push_back(codes_data.size());
    }
}

void IndexNeuroFlyHash::reset() {
    IndexNeuro::reset();
    codes_data.clear();
    codes_offsets.clear();
    codes_offsets.push_back(0);
}

void IndexNeuroFlyHash::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained");

    const float* data = get_flat_data_fly(inner_index);
    idx_t nb = inner_index->ntotal;

    // Resolve parameters
    bool do_rerank = rerank;
    int ks = k_sparse;

    auto fp = dynamic_cast<const NeuroFlyHashParams*>(params);
    if (fp) {
        do_rerank = fp->rerank;
        if (fp->k_sparse > 0) {
            ks = fp->k_sparse;
        }
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

        // Compute query code
        std::vector<int> query_code;
        compute_code(query, query_code);

        // Compute Jaccard distances to all database codes
        std::vector<std::pair<float, idx_t>> scored(nb);
        for (idx_t i = 0; i < nb; i++) {
            // Get code for vector i from flattened storage
            const int* code_begin = codes_data.data() + codes_offsets[i];
            const int* code_end = codes_data.data() + codes_offsets[i + 1];

            int inter = intersection_size_range(
                    query_code.data(), query_code.data() + query_code.size(),
                    code_begin, code_end);
            int uni = union_size_range(
                    query_code.data(), query_code.data() + query_code.size(),
                    code_begin, code_end);

            float jaccard_dist = (uni == 0) ? 0.0f : 1.0f - static_cast<float>(inter) / uni;
            scored[i] = {jaccard_dist, i};
        }

        // Get top candidates by Jaccard
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

/*************************************************************
 * IndexNeuroBioHash (HS-03)
 *************************************************************/

IndexNeuroBioHash::IndexNeuroBioHash(
        Index* inner,
        int expansion_factor,
        float sparsity,
        bool own_inner)
        : IndexNeuroFlyHash(inner, expansion_factor, sparsity, own_inner) {}

void IndexNeuroBioHash::train_pairs(
        idx_t n_pairs,
        const float* vec_a,
        const float* vec_b,
        const int* is_similar) {
    for (idx_t i = 0; i < n_pairs; i++) {
        update_pair(
                vec_a + i * d,
                vec_b + i * d,
                is_similar[i] != 0);
    }
}

void IndexNeuroBioHash::update_pair(
        const float* vec_a,
        const float* vec_b,
        bool is_similar) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained first");

    // Compute codes for both vectors
    std::vector<int> code_a, code_b;
    compute_code(vec_a, code_a);
    compute_code(vec_b, code_b);

    // Find shared active neurons
    std::vector<int> shared;
    size_t i = 0, j = 0;
    while (i < code_a.size() && j < code_b.size()) {
        if (code_a[i] < code_b[j]) {
            i++;
        } else if (code_a[i] > code_b[j]) {
            j++;
        } else {
            shared.push_back(code_a[i]);
            i++;
            j++;
        }
    }

    // Update weights for shared neurons
    float delta = is_similar ? learning_rate : -learning_rate;

    for (int neuron : shared) {
        for (int c = 0; c < connections_per_neuron; c++) {
            projection_weights[neuron * connections_per_neuron + c] += delta;
        }
    }

    // Apply weight decay to all weights
    for (size_t w = 0; w < projection_weights.size(); w++) {
        projection_weights[w] *= weight_decay;
    }

    training_iterations++;
}

void IndexNeuroBioHash::recompute_codes() {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");

    const float* data = get_flat_data_fly(inner_index);
    idx_t nb = inner_index->ntotal;

    // Clear and recompute all codes
    codes_data.clear();
    codes_offsets.clear();
    codes_offsets.push_back(0);

    for (idx_t i = 0; i < nb; i++) {
        std::vector<int> code;
        compute_code(data + i * d, code);
        codes_data.insert(codes_data.end(), code.begin(), code.end());
        codes_offsets.push_back(codes_data.size());
    }
}

} // namespace faiss
