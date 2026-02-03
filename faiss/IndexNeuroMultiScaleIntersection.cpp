/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroMultiScaleIntersection.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace faiss {

IndexNeuroMultiScaleIntersection::IndexNeuroMultiScaleIntersection(
        int d,
        const std::vector<float>& scales)
        : IndexNeuro(nullptr, false), scales(scales) {
    this->d = d;
    words_per_vec = (d + 63) / 64;
    signatures.resize(scales.size());
    is_trained = false;
}

void IndexNeuroMultiScaleIntersection::compute_signature(
        const float* vec,
        float scale,
        uint64_t* sig) const {
    for (int w = 0; w < words_per_vec; w++) {
        sig[w] = 0;
    }

    for (int j = 0; j < d; j++) {
        float val = std::tanh(vec[j] * scale);
        if (val >= 0.0f) {
            int word_idx = j / 64;
            int bit_idx = j % 64;
            sig[word_idx] |= (1ULL << bit_idx);
        }
    }
}

uint64_t IndexNeuroMultiScaleIntersection::compute_bucket_key(
        const std::vector<std::vector<uint64_t>>& sigs) const {
    // XOR fold all signatures into a single 64-bit key
    uint64_t key = 0;
    for (const auto& sig : sigs) {
        for (uint64_t word : sig) {
            key ^= word;
            // Rotate to spread bits
            key = (key << 7) | (key >> 57);
        }
    }
    return key;
}

int IndexNeuroMultiScaleIntersection::hamming_distance(
        const uint64_t* sig1,
        const uint64_t* sig2) const {
    int dist = 0;
    for (int w = 0; w < words_per_vec; w++) {
        dist += __builtin_popcountll(sig1[w] ^ sig2[w]);
    }
    return dist;
}

void IndexNeuroMultiScaleIntersection::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(n > 0);

    if (!inner_index) {
        inner_index = new IndexFlatL2(d);
        own_inner = true;
    }

    // Clear existing data
    for (auto& sigs : signatures) {
        sigs.clear();
    }
    buckets.clear();

    // Compute signatures and build bucket index
    std::vector<uint64_t> sig(words_per_vec);
    std::vector<std::vector<uint64_t>> vec_sigs(scales.size());

    for (idx_t i = 0; i < n; i++) {
        const float* vec = x + i * d;

        // Compute signatures at all scales
        for (size_t s = 0; s < scales.size(); s++) {
            vec_sigs[s].resize(words_per_vec);
            compute_signature(vec, scales[s], vec_sigs[s].data());
            signatures[s].insert(
                    signatures[s].end(),
                    vec_sigs[s].begin(),
                    vec_sigs[s].end());
        }

        // Add to bucket
        uint64_t key = compute_bucket_key(vec_sigs);
        buckets[key].push_back(i);
    }

    inner_index->add(n, x);
    ntotal = n;
    is_trained = true;
}

void IndexNeuroMultiScaleIntersection::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "Index must be trained first");

    std::vector<uint64_t> sig(words_per_vec);
    std::vector<std::vector<uint64_t>> vec_sigs(scales.size());

    for (idx_t i = 0; i < n; i++) {
        const float* vec = x + i * d;
        idx_t vec_id = ntotal + i;

        for (size_t s = 0; s < scales.size(); s++) {
            vec_sigs[s].resize(words_per_vec);
            compute_signature(vec, scales[s], vec_sigs[s].data());
            signatures[s].insert(
                    signatures[s].end(),
                    vec_sigs[s].begin(),
                    vec_sigs[s].end());
        }

        uint64_t key = compute_bucket_key(vec_sigs);
        buckets[key].push_back(vec_id);
    }

    if (inner_index) {
        inner_index->add(n, x);
    }
    ntotal += n;
}

void IndexNeuroMultiScaleIntersection::search(
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
    int fb_k = fallback_k;

    auto* msp = dynamic_cast<const NeuroMultiScaleIntersectionParams*>(params);
    if (msp) {
        fb_k = msp->fallback_k;
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

        // Compute query signatures
        std::vector<std::vector<uint64_t>> query_sigs(scales.size());
        for (size_t s = 0; s < scales.size(); s++) {
            query_sigs[s].resize(words_per_vec);
            compute_signature(query, scales[s], query_sigs[s].data());
        }

        // Look up bucket
        uint64_t key = compute_bucket_key(query_sigs);
        std::vector<idx_t> candidates;

        auto it = buckets.find(key);
        if (it != buckets.end() && !it->second.empty()) {
            candidates = it->second;
        }

        // Fallback: if bucket empty, find closest by Hamming distance
        if (candidates.empty()) {
            std::vector<std::pair<int, idx_t>> hamming_dists;
            hamming_dists.reserve(ntotal);

            for (idx_t i = 0; i < ntotal; i++) {
                int total_hamming = 0;
                for (size_t s = 0; s < scales.size(); s++) {
                    const uint64_t* vec_sig =
                            signatures[s].data() + i * words_per_vec;
                    total_hamming +=
                            hamming_distance(query_sigs[s].data(), vec_sig);
                }
                hamming_dists.push_back({total_hamming, i});
            }

            std::partial_sort(
                    hamming_dists.begin(),
                    hamming_dists.begin() + std::min(static_cast<idx_t>(fb_k), ntotal),
                    hamming_dists.end());

            for (int i = 0; i < std::min(fb_k, static_cast<int>(ntotal)); i++) {
                candidates.push_back(hamming_dists[i].second);
            }
        }

        // Compute L2 distances
        std::vector<std::pair<float, idx_t>> l2_dists;
        l2_dists.reserve(candidates.size());

        for (idx_t cand : candidates) {
            float dist = fvec_L2sqr(query, xb + cand * d, d);
            l2_dists.push_back({dist, cand});
        }
        total_calcs += candidates.size() * d;

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

void IndexNeuroMultiScaleIntersection::reset() {
    for (auto& sigs : signatures) {
        sigs.clear();
    }
    buckets.clear();
    if (inner_index) {
        inner_index->reset();
    }
    ntotal = 0;
}

} // namespace faiss
