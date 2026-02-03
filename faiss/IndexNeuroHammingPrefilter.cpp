/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroHammingPrefilter.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace faiss {

IndexNeuroHammingPrefilter::IndexNeuroHammingPrefilter(int d, float keep_ratio)
        : IndexNeuro(nullptr, false), keep_ratio(keep_ratio) {
    this->d = d;
    words_per_vec = (d + 63) / 64;
    thresholds.resize(d, 0.0f);
    is_trained = false;
}

void IndexNeuroHammingPrefilter::encode(
        const float* vec,
        uint64_t* code) const {
    // Initialize code to zero
    for (int w = 0; w < words_per_vec; w++) {
        code[w] = 0;
    }

    // Set bits based on threshold comparison
    for (int j = 0; j < d; j++) {
        if (vec[j] > thresholds[j]) {
            int word_idx = j / 64;
            int bit_idx = j % 64;
            code[word_idx] |= (1ULL << bit_idx);
        }
    }
}

int IndexNeuroHammingPrefilter::hamming_distance(
        const uint64_t* code1,
        const uint64_t* code2) const {
    int dist = 0;
    for (int w = 0; w < words_per_vec; w++) {
        dist += __builtin_popcountll(code1[w] ^ code2[w]);
    }
    return dist;
}

void IndexNeuroHammingPrefilter::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(n > 0);

    // Create inner index if not provided
    if (!inner_index) {
        inner_index = new IndexFlatL2(d);
        own_inner = true;
    }

    // Compute per-dimension medians as thresholds
    std::vector<float> col_values(n);
    for (int j = 0; j < d; j++) {
        for (idx_t i = 0; i < n; i++) {
            col_values[i] = x[i * d + j];
        }
        std::nth_element(
                col_values.begin(),
                col_values.begin() + n / 2,
                col_values.end());
        thresholds[j] = col_values[n / 2];
    }

    // Clear existing codes
    binary_codes.clear();

    // Encode all training vectors
    std::vector<uint64_t> code(words_per_vec);
    for (idx_t i = 0; i < n; i++) {
        encode(x + i * d, code.data());
        binary_codes.insert(binary_codes.end(), code.begin(), code.end());
    }

    // Add to inner index
    inner_index->add(n, x);
    ntotal = n;
    is_trained = true;
}

void IndexNeuroHammingPrefilter::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "Index must be trained first");

    // Encode new vectors
    std::vector<uint64_t> code(words_per_vec);
    for (idx_t i = 0; i < n; i++) {
        encode(x + i * d, code.data());
        binary_codes.insert(binary_codes.end(), code.begin(), code.end());
    }

    // Add to inner index
    if (inner_index) {
        inner_index->add(n, x);
    }
    ntotal += n;
}

void IndexNeuroHammingPrefilter::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(ntotal > 0);
    FAISS_THROW_IF_NOT(inner_index);

    auto* flat = dynamic_cast<IndexFlat*>(inner_index);
    FAISS_THROW_IF_NOT_MSG(flat, "inner_index must be IndexFlat");

    const float* xb = flat->get_xb();

    // Parse parameters
    float kr = keep_ratio;

    auto* hfp = dynamic_cast<const NeuroHammingPrefilterParams*>(params);
    if (hfp) {
        kr = hfp->keep_ratio;
    }

    bool collect = false;
    if (params) {
        auto* nsp = dynamic_cast<const NeuroSearchParameters*>(params);
        if (nsp) {
            collect = nsp->collect_stats;
        }
    }

    // How many candidates to keep
    idx_t keep_n = std::max(
            static_cast<idx_t>(ntotal * kr),
            std::min(k * 2, ntotal));

    int64_t total_calcs = 0;

#pragma omp parallel for reduction(+ : total_calcs)
    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * d;

        // Encode query
        std::vector<uint64_t> query_code(words_per_vec);
        encode(query, query_code.data());

        // Compute Hamming distances to all vectors
        std::vector<std::pair<int, idx_t>> hamming_dists(ntotal);
        for (idx_t i = 0; i < ntotal; i++) {
            const uint64_t* vec_code =
                    binary_codes.data() + i * words_per_vec;
            int hdist = hamming_distance(query_code.data(), vec_code);
            hamming_dists[i] = {hdist, i};
        }

        // Partial sort to get top keep_n by Hamming (ascending)
        std::partial_sort(
                hamming_dists.begin(),
                hamming_dists.begin() + keep_n,
                hamming_dists.end());

        // Compute L2 distances for candidates
        std::vector<std::pair<float, idx_t>> l2_dists;
        l2_dists.reserve(keep_n);

        for (idx_t i = 0; i < keep_n; i++) {
            idx_t cand = hamming_dists[i].second;
            float dist = fvec_L2sqr(query, xb + cand * d, d);
            l2_dists.push_back({dist, cand});
        }
        total_calcs += keep_n * d;

        // Sort and output top-k
        size_t actual_k = std::min(static_cast<size_t>(k), l2_dists.size());
        std::partial_sort(
                l2_dists.begin(), l2_dists.begin() + actual_k, l2_dists.end());

        for (size_t i = 0; i < actual_k; i++) {
            distances[q * k + i] = l2_dists[i].first;
            labels[q * k + i] = l2_dists[i].second;
        }
        for (size_t i = actual_k; i < static_cast<size_t>(k); i++) {
            distances[q * k + i] = std::numeric_limits<float>::max();
            labels[q * k + i] = -1;
        }
    }

    if (collect) {
        last_stats.calculations_performed = total_calcs;
        last_stats.columns_used = d;
    }
}

void IndexNeuroHammingPrefilter::reset() {
    binary_codes.clear();
    if (inner_index) {
        inner_index->reset();
    }
    ntotal = 0;
}

} // namespace faiss
