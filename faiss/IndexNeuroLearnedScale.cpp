/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroLearnedScale.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <set>
#include <vector>

namespace faiss {

IndexNeuroLearnedScale::IndexNeuroLearnedScale(int d, int num_scales)
        : IndexNeuro(nullptr, false), num_scales(num_scales) {
    this->d = d;
    words_per_vec = (d + 63) / 64;
    is_trained = false;
}

void IndexNeuroLearnedScale::compute_signature(
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

int IndexNeuroLearnedScale::hamming_distance(
        const uint64_t* sig1,
        const uint64_t* sig2) const {
    int dist = 0;
    for (int w = 0; w < words_per_vec; w++) {
        dist += __builtin_popcountll(sig1[w] ^ sig2[w]);
    }
    return dist;
}

float IndexNeuroLearnedScale::evaluate_recall(
        const float* data,
        idx_t n,
        const float* queries,
        idx_t nq,
        idx_t k,
        const std::vector<float>& scales) const {
    // Compute signatures for data
    std::vector<std::vector<uint64_t>> data_sigs(scales.size());
    std::vector<uint64_t> sig(words_per_vec);

    for (size_t s = 0; s < scales.size(); s++) {
        data_sigs[s].resize(n * words_per_vec);
        for (idx_t i = 0; i < n; i++) {
            compute_signature(data + i * d, scales[s], sig.data());
            std::copy(
                    sig.begin(),
                    sig.end(),
                    data_sigs[s].begin() + i * words_per_vec);
        }
    }

    // Compute ground truth with brute force
    std::vector<std::set<idx_t>> ground_truth(nq);
    for (idx_t q = 0; q < nq; q++) {
        const float* query = queries + q * d;
        std::vector<std::pair<float, idx_t>> dists(n);
        for (idx_t i = 0; i < n; i++) {
            dists[i] = {fvec_L2sqr(query, data + i * d, d), i};
        }
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
        for (idx_t i = 0; i < k; i++) {
            ground_truth[q].insert(dists[i].second);
        }
    }

    // Evaluate recall with signature filtering
    int max_hamming = d / 2;
    float total_recall = 0.0f;

    for (idx_t q = 0; q < nq; q++) {
        const float* query = queries + q * d;

        // Compute query signatures
        std::vector<std::vector<uint64_t>> query_sigs(scales.size());
        for (size_t s = 0; s < scales.size(); s++) {
            query_sigs[s].resize(words_per_vec);
            compute_signature(query, scales[s], query_sigs[s].data());
        }

        // Filter candidates
        std::vector<idx_t> candidates;
        for (idx_t i = 0; i < n; i++) {
            bool passes = true;
            for (size_t s = 0; s < scales.size(); s++) {
                const uint64_t* vec_sig =
                        data_sigs[s].data() + i * words_per_vec;
                int hdist = hamming_distance(query_sigs[s].data(), vec_sig);
                if (hdist > max_hamming) {
                    passes = false;
                    break;
                }
            }
            if (passes) {
                candidates.push_back(i);
            }
        }

        // If no candidates, fallback
        if (candidates.empty()) {
            candidates.resize(n);
            for (idx_t i = 0; i < n; i++) {
                candidates[i] = i;
            }
        }

        // Compute L2 and find top-k
        std::vector<std::pair<float, idx_t>> l2_dists;
        for (idx_t cand : candidates) {
            float dist = fvec_L2sqr(query, data + cand * d, d);
            l2_dists.push_back({dist, cand});
        }

        size_t actual_k = std::min(static_cast<size_t>(k), l2_dists.size());
        std::partial_sort(
                l2_dists.begin(), l2_dists.begin() + actual_k, l2_dists.end());

        // Compute recall
        int hits = 0;
        for (size_t i = 0; i < actual_k; i++) {
            if (ground_truth[q].count(l2_dists[i].second)) {
                hits++;
            }
        }
        total_recall += static_cast<float>(hits) / k;
    }

    return total_recall / nq;
}

void IndexNeuroLearnedScale::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(n > 0);

    if (!inner_index) {
        inner_index = new IndexFlatL2(d);
        own_inner = true;
    }

    // Sample validation queries from data
    std::mt19937 rng(random_seed);
    idx_t nq = std::min(static_cast<idx_t>(validation_queries), n / 10);
    nq = std::max(nq, static_cast<idx_t>(10));

    std::vector<idx_t> query_indices(n);
    std::iota(query_indices.begin(), query_indices.end(), 0);
    std::shuffle(query_indices.begin(), query_indices.end(), rng);

    std::vector<float> validation_data(nq * d);
    for (idx_t i = 0; i < nq; i++) {
        std::copy(
                x + query_indices[i] * d,
                x + query_indices[i] * d + d,
                validation_data.data() + i * d);
    }

    // Random search for optimal scales
    float best_recall = 0.0f;
    std::vector<float> best_scales;

    std::uniform_real_distribution<float> scale_dist(0.1f, 4.0f);

    for (int iter = 0; iter < search_iterations; iter++) {
        // Generate random scales
        std::vector<float> candidate_scales(num_scales);
        for (int i = 0; i < num_scales; i++) {
            candidate_scales[i] = scale_dist(rng);
        }
        std::sort(candidate_scales.begin(), candidate_scales.end());

        // Evaluate recall
        float recall = evaluate_recall(
                x, n, validation_data.data(), nq, validation_k, candidate_scales);

        if (recall > best_recall) {
            best_recall = recall;
            best_scales = candidate_scales;
        }
    }

    // Use learned scales
    learned_scales = best_scales.empty()
            ? std::vector<float>{0.5f, 1.0f, 2.0f}
            : best_scales;

    // Resize signatures
    signatures.resize(learned_scales.size());
    for (auto& sigs : signatures) {
        sigs.clear();
    }

    // Compute signatures for all data
    std::vector<uint64_t> sig(words_per_vec);
    for (idx_t i = 0; i < n; i++) {
        const float* vec = x + i * d;
        for (size_t s = 0; s < learned_scales.size(); s++) {
            compute_signature(vec, learned_scales[s], sig.data());
            signatures[s].insert(signatures[s].end(), sig.begin(), sig.end());
        }
    }

    inner_index->add(n, x);
    ntotal = n;
    is_trained = true;
}

void IndexNeuroLearnedScale::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "Index must be trained first");

    std::vector<uint64_t> sig(words_per_vec);
    for (idx_t i = 0; i < n; i++) {
        const float* vec = x + i * d;
        for (size_t s = 0; s < learned_scales.size(); s++) {
            compute_signature(vec, learned_scales[s], sig.data());
            signatures[s].insert(signatures[s].end(), sig.begin(), sig.end());
        }
    }

    if (inner_index) {
        inner_index->add(n, x);
    }
    ntotal += n;
}

void IndexNeuroLearnedScale::search(
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

    int max_hamming = max_hamming_distance;
    if (max_hamming <= 0) {
        max_hamming = d / 2;
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
        std::vector<std::vector<uint64_t>> query_sigs(learned_scales.size());
        for (size_t s = 0; s < learned_scales.size(); s++) {
            query_sigs[s].resize(words_per_vec);
            compute_signature(query, learned_scales[s], query_sigs[s].data());
        }

        // Filter candidates
        std::vector<idx_t> candidates;
        for (idx_t i = 0; i < ntotal; i++) {
            bool passes = true;
            for (size_t s = 0; s < learned_scales.size(); s++) {
                const uint64_t* vec_sig =
                        signatures[s].data() + i * words_per_vec;
                int hdist = hamming_distance(query_sigs[s].data(), vec_sig);
                if (hdist > max_hamming) {
                    passes = false;
                    break;
                }
            }
            if (passes) {
                candidates.push_back(i);
            }
        }

        // Fallback if no candidates
        if (candidates.empty()) {
            candidates.resize(ntotal);
            for (idx_t i = 0; i < ntotal; i++) {
                candidates[i] = i;
            }
        }

        // Compute L2
        std::vector<std::pair<float, idx_t>> l2_dists;
        l2_dists.reserve(candidates.size());

        for (idx_t cand : candidates) {
            float dist = fvec_L2sqr(query, xb + cand * d, d);
            l2_dists.push_back({dist, cand});
        }
        total_calcs += candidates.size() * d;

        // Sort and output
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

void IndexNeuroLearnedScale::reset() {
    for (auto& sigs : signatures) {
        sigs.clear();
    }
    if (inner_index) {
        inner_index->reset();
    }
    ntotal = 0;
}

} // namespace faiss
