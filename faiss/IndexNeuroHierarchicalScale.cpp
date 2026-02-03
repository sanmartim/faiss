/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroHierarchicalScale.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace faiss {

IndexNeuroHierarchicalScale::IndexNeuroHierarchicalScale(int d)
        : IndexNeuro(nullptr, false) {
    this->d = d;
    words_per_vec = (d + 63) / 64;
    signatures.resize(cascade_scales.size());
    is_trained = false;
}

void IndexNeuroHierarchicalScale::compute_signature(
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

int IndexNeuroHierarchicalScale::hamming_distance(
        const uint64_t* sig1,
        const uint64_t* sig2) const {
    int dist = 0;
    for (int w = 0; w < words_per_vec; w++) {
        dist += __builtin_popcountll(sig1[w] ^ sig2[w]);
    }
    return dist;
}

void IndexNeuroHierarchicalScale::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(n > 0);

    if (!inner_index) {
        inner_index = new IndexFlatL2(d);
        own_inner = true;
    }

    // Ensure signatures vector has correct size
    signatures.resize(cascade_scales.size());
    for (auto& sigs : signatures) {
        sigs.clear();
    }

    // Compute signatures at all cascade levels
    std::vector<uint64_t> sig(words_per_vec);
    for (idx_t i = 0; i < n; i++) {
        const float* vec = x + i * d;
        for (size_t s = 0; s < cascade_scales.size(); s++) {
            compute_signature(vec, cascade_scales[s], sig.data());
            signatures[s].insert(signatures[s].end(), sig.begin(), sig.end());
        }
    }

    inner_index->add(n, x);
    ntotal = n;
    is_trained = true;
}

void IndexNeuroHierarchicalScale::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "Index must be trained first");

    std::vector<uint64_t> sig(words_per_vec);
    for (idx_t i = 0; i < n; i++) {
        const float* vec = x + i * d;
        for (size_t s = 0; s < cascade_scales.size(); s++) {
            compute_signature(vec, cascade_scales[s], sig.data());
            signatures[s].insert(signatures[s].end(), sig.begin(), sig.end());
        }
    }

    if (inner_index) {
        inner_index->add(n, x);
    }
    ntotal += n;
}

void IndexNeuroHierarchicalScale::search(
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
    std::vector<float> kr = keep_ratios;
    int target = target_candidates;

    auto* hsp = dynamic_cast<const NeuroHierarchicalScaleParams*>(params);
    if (hsp) {
        if (!hsp->keep_ratios.empty()) {
            kr = hsp->keep_ratios;
        }
        target = hsp->target_candidates;
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

        // Compute query signatures at all levels
        std::vector<std::vector<uint64_t>> query_sigs(cascade_scales.size());
        for (size_t s = 0; s < cascade_scales.size(); s++) {
            query_sigs[s].resize(words_per_vec);
            compute_signature(query, cascade_scales[s], query_sigs[s].data());
        }

        // Start with all candidates
        std::vector<idx_t> candidates(ntotal);
        for (idx_t i = 0; i < ntotal; i++) {
            candidates[i] = i;
        }

        // Cascade through levels
        for (size_t level = 0; level < cascade_scales.size() && !candidates.empty(); level++) {
            // Check early termination
            if (target > 0 && static_cast<int>(candidates.size()) <= target) {
                break;
            }

            // Compute Hamming distances for current candidates
            std::vector<std::pair<int, idx_t>> hamming_dists;
            hamming_dists.reserve(candidates.size());

            for (idx_t cand : candidates) {
                const uint64_t* vec_sig =
                        signatures[level].data() + cand * words_per_vec;
                int hdist = hamming_distance(query_sigs[level].data(), vec_sig);
                hamming_dists.push_back({hdist, cand});
            }

            // Sort by Hamming distance
            std::sort(hamming_dists.begin(), hamming_dists.end());

            // Keep top ratio
            float ratio = (level < kr.size()) ? kr[level] : 0.5f;
            size_t keep_n = std::max(
                    static_cast<size_t>(candidates.size() * ratio),
                    static_cast<size_t>(k));

            candidates.clear();
            for (size_t i = 0; i < std::min(keep_n, hamming_dists.size()); i++) {
                candidates.push_back(hamming_dists[i].second);
            }
        }

        // Compute L2 distances for final candidates
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

void IndexNeuroHierarchicalScale::reset() {
    for (auto& sigs : signatures) {
        sigs.clear();
    }
    if (inner_index) {
        inner_index->reset();
    }
    ntotal = 0;
}

} // namespace faiss
