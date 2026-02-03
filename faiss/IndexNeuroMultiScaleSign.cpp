/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroMultiScaleSign.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace faiss {

IndexNeuroMultiScaleSign::IndexNeuroMultiScaleSign(
        int d,
        const std::vector<float>& scales)
        : IndexNeuro(nullptr, false), scales(scales) {
    this->d = d;
    words_per_vec = (d + 63) / 64;
    signatures.resize(scales.size());
    is_trained = false;
}

void IndexNeuroMultiScaleSign::compute_signature(
        const float* vec,
        float scale,
        uint64_t* sig) const {
    // Initialize signature to zero
    for (int w = 0; w < words_per_vec; w++) {
        sig[w] = 0;
    }

    // Set bits based on sign(tanh(x * scale))
    for (int j = 0; j < d; j++) {
        float val = std::tanh(vec[j] * scale);
        if (val >= 0.0f) {
            int word_idx = j / 64;
            int bit_idx = j % 64;
            sig[word_idx] |= (1ULL << bit_idx);
        }
    }
}

int IndexNeuroMultiScaleSign::hamming_distance(
        const uint64_t* sig1,
        const uint64_t* sig2) const {
    int dist = 0;
    for (int w = 0; w < words_per_vec; w++) {
        dist += __builtin_popcountll(sig1[w] ^ sig2[w]);
    }
    return dist;
}

void IndexNeuroMultiScaleSign::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(n > 0);

    // Create inner index if not provided
    if (!inner_index) {
        inner_index = new IndexFlatL2(d);
        own_inner = true;
    }

    // Clear existing signatures
    for (auto& sigs : signatures) {
        sigs.clear();
    }

    // Compute signatures for all training vectors
    std::vector<uint64_t> sig(words_per_vec);
    for (idx_t i = 0; i < n; i++) {
        const float* vec = x + i * d;
        for (size_t s = 0; s < scales.size(); s++) {
            compute_signature(vec, scales[s], sig.data());
            signatures[s].insert(
                    signatures[s].end(), sig.begin(), sig.end());
        }
    }

    // Add to inner index
    inner_index->add(n, x);
    ntotal = n;
    is_trained = true;
}

void IndexNeuroMultiScaleSign::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "Index must be trained first");

    // Compute signatures for new vectors
    std::vector<uint64_t> sig(words_per_vec);
    for (idx_t i = 0; i < n; i++) {
        const float* vec = x + i * d;
        for (size_t s = 0; s < scales.size(); s++) {
            compute_signature(vec, scales[s], sig.data());
            signatures[s].insert(
                    signatures[s].end(), sig.begin(), sig.end());
        }
    }

    // Add to inner index
    if (inner_index) {
        inner_index->add(n, x);
    }
    ntotal += n;
}

void IndexNeuroMultiScaleSign::search(
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
    int max_hamming = max_hamming_distance;
    bool use_intersection = true;

    auto* msp = dynamic_cast<const NeuroMultiScaleSignParams*>(params);
    if (msp) {
        if (msp->max_hamming_distance > 0) {
            max_hamming = msp->max_hamming_distance;
        }
        use_intersection = msp->use_intersection;
    }

    // Auto-compute max Hamming distance if not set
    // For small datasets, allow more bits to differ
    if (max_hamming <= 0) {
        max_hamming = d / 4;  // ~25% of bits can differ for better recall
    }

    bool collect = false;
    if (params) {
        auto* nsp = dynamic_cast<const NeuroSearchParameters*>(params);
        if (nsp) {
            collect = nsp->collect_stats;
        }
    }

    int64_t total_calcs = 0;

#pragma omp parallel for reduction(+ : total_calcs)
    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * d;

        // Compute query signatures for all scales
        std::vector<std::vector<uint64_t>> query_sigs(scales.size());
        for (size_t s = 0; s < scales.size(); s++) {
            query_sigs[s].resize(words_per_vec);
            compute_signature(query, scales[s], query_sigs[s].data());
        }

        // Filter candidates by Hamming distance
        std::vector<idx_t> candidates;
        for (idx_t i = 0; i < ntotal; i++) {
            bool passes_all = true;
            bool passes_any = false;

            for (size_t s = 0; s < scales.size(); s++) {
                const uint64_t* vec_sig =
                        signatures[s].data() + i * words_per_vec;
                int hdist = hamming_distance(query_sigs[s].data(), vec_sig);

                if (hdist <= max_hamming) {
                    passes_any = true;
                } else {
                    passes_all = false;
                }
            }

            if (use_intersection ? passes_all : passes_any) {
                candidates.push_back(i);
            }
        }

        // If no candidates pass, fall back to all vectors
        if (candidates.empty()) {
            candidates.resize(ntotal);
            for (idx_t i = 0; i < ntotal; i++) {
                candidates[i] = i;
            }
        }

        // Compute L2 distances for candidates
        std::vector<std::pair<float, idx_t>> dists;
        dists.reserve(candidates.size());

        for (idx_t cand : candidates) {
            float dist = fvec_L2sqr(query, xb + cand * d, d);
            dists.push_back({dist, cand});
        }
        total_calcs += candidates.size() * d;

        // Sort and output top-k
        size_t actual_k = std::min(static_cast<size_t>(k), dists.size());
        std::partial_sort(
                dists.begin(), dists.begin() + actual_k, dists.end());

        for (size_t i = 0; i < actual_k; i++) {
            distances[q * k + i] = dists[i].first;
            labels[q * k + i] = dists[i].second;
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

void IndexNeuroMultiScaleSign::reset() {
    for (auto& sigs : signatures) {
        sigs.clear();
    }
    if (inner_index) {
        inner_index->reset();
    }
    ntotal = 0;
}

} // namespace faiss
