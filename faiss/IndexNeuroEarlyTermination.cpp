/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroEarlyTermination.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace faiss {

IndexNeuroEarlyTermination::IndexNeuroEarlyTermination(
        Index* sub_index,
        float confidence_threshold,
        int min_candidates)
        : IndexNeuro(sub_index, false),
          sub_index(sub_index),
          confidence_threshold(confidence_threshold),
          min_candidates(min_candidates) {}

void IndexNeuroEarlyTermination::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");
    sub_index->train(n, x);
    is_trained = sub_index->is_trained;
}

void IndexNeuroEarlyTermination::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");
    sub_index->add(n, x);
    ntotal = sub_index->ntotal;
}

void IndexNeuroEarlyTermination::reset() {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");
    sub_index->reset();
    ntotal = 0;
}

void IndexNeuroEarlyTermination::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");
    sub_index->reconstruct(key, recons);
}

void IndexNeuroEarlyTermination::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");

    // Resolve parameters
    float conf_thresh = confidence_threshold;
    int min_cand = min_candidates;

    auto etp = dynamic_cast<const NeuroEarlyTermParams*>(params);
    if (etp) {
        if (etp->confidence_threshold >= 0) conf_thresh = etp->confidence_threshold;
        if (etp->min_candidates > 0) min_cand = etp->min_candidates;
    }

    // Try to get raw vectors from sub_index for manual search
    IndexFlat* flat = dynamic_cast<IndexFlat*>(sub_index);
    if (!flat) {
        // Fall back to regular search
        sub_index->search(n, x, k, distances, labels, params);
        return;
    }

    const float* xb = flat->get_xb();
    if (!xb) {
        sub_index->search(n, x, k, distances, labels, params);
        return;
    }

    int64_t early_stops = 0;

#pragma omp parallel for reduction(+ : early_stops)
    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * d;

        // Priority queue for top-k (max-heap)
        std::vector<std::pair<float, idx_t>> topk;
        topk.reserve(k + 1);

        // Process candidates with early termination
        idx_t candidates_checked = 0;
        float k_th_dist = std::numeric_limits<float>::max();
        bool stopped_early = false;

        for (idx_t i = 0; i < ntotal && !stopped_early; i++) {
            float dist = fvec_L2sqr(query, xb + i * d, d);
            candidates_checked++;

            if (topk.size() < static_cast<size_t>(k)) {
                topk.push_back({dist, i});
                std::push_heap(topk.begin(), topk.end());
                if (topk.size() == static_cast<size_t>(k)) {
                    k_th_dist = topk[0].first;
                }
            } else if (dist < topk[0].first) {
                std::pop_heap(topk.begin(), topk.end());
                topk.back() = {dist, i};
                std::push_heap(topk.begin(), topk.end());
                k_th_dist = topk[0].first;
            }

            // Check early termination condition
            if (candidates_checked >= min_cand && topk.size() == static_cast<size_t>(k)) {
                float best_dist = std::numeric_limits<float>::max();
                for (const auto& p : topk) {
                    best_dist = std::min(best_dist, p.first);
                }

                // Gap ratio: how much worse is k-th compared to best
                if (best_dist > 1e-10f) {
                    float gap_ratio = (k_th_dist - best_dist) / best_dist;
                    if (gap_ratio > conf_thresh) {
                        stopped_early = true;
                        early_stops++;
                    }
                }
            }
        }

        // Sort top-k results
        std::sort_heap(topk.begin(), topk.end());

        // Output results
        size_t actual_k = std::min(static_cast<size_t>(k), topk.size());
        for (size_t i = 0; i < actual_k; i++) {
            distances[q * k + i] = topk[i].first;
            labels[q * k + i] = topk[i].second;
        }
        for (size_t i = actual_k; i < static_cast<size_t>(k); i++) {
            distances[q * k + i] = std::numeric_limits<float>::max();
            labels[q * k + i] = -1;
        }
    }

    // Update stats
    if (params) {
        auto nsp = dynamic_cast<const NeuroSearchParameters*>(params);
        if (nsp && nsp->collect_stats) {
            last_stats.calculations_performed = ntotal * n - early_stops * (ntotal - min_cand) / 2;
        }
    }
}

} // namespace faiss
