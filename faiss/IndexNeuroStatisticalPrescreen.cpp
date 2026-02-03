/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroStatisticalPrescreen.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace faiss {

IndexNeuroStatisticalPrescreen::IndexNeuroStatisticalPrescreen(
        int d,
        float keep_ratio)
        : IndexNeuro(nullptr, false), keep_ratio(keep_ratio) {
    this->d = d;
    is_trained = false;
}

void IndexNeuroStatisticalPrescreen::compute_stats(
        const float* vec,
        float& norm,
        float& mean) const {
    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int j = 0; j < d; j++) {
        sum += vec[j];
        sum_sq += vec[j] * vec[j];
    }

    norm = std::sqrt(sum_sq);
    mean = sum / d;
}

void IndexNeuroStatisticalPrescreen::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(n > 0);

    // Store vectors
    vectors.resize(n * d);
    std::copy(x, x + n * d, vectors.data());

    // Compute statistics
    norms.resize(n);
    means.resize(n);

    for (idx_t i = 0; i < n; i++) {
        compute_stats(x + i * d, norms[i], means[i]);
    }

    ntotal = n;
    is_trained = true;
}

void IndexNeuroStatisticalPrescreen::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "Index must be trained first");

    idx_t old_ntotal = ntotal;

    // Extend storage
    vectors.resize((old_ntotal + n) * d);
    std::copy(x, x + n * d, vectors.data() + old_ntotal * d);

    norms.resize(old_ntotal + n);
    means.resize(old_ntotal + n);

    // Compute statistics for new vectors
    for (idx_t i = 0; i < n; i++) {
        compute_stats(x + i * d, norms[old_ntotal + i], means[old_ntotal + i]);
    }

    ntotal += n;
}

void IndexNeuroStatisticalPrescreen::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(ntotal > 0);

    // Parse parameters
    float kr = keep_ratio;
    float nw = norm_weight;
    float mw = mean_weight;

    auto* spp = dynamic_cast<const NeuroStatisticalPrescreenParams*>(params);
    if (spp) {
        kr = spp->keep_ratio;
        nw = spp->norm_weight;
        mw = spp->mean_weight;
    }

    bool collect = false;
    if (params) {
        auto* nsp = dynamic_cast<const NeuroSearchParameters*>(params);
        if (nsp) {
            collect = nsp->collect_stats;
        }
    }

    // How many to keep
    idx_t keep_n = std::max(
            static_cast<idx_t>(ntotal * kr),
            std::min(k * 2, ntotal));

    float sqrt_d = std::sqrt(static_cast<float>(d));

    int64_t total_calcs = 0;

#pragma omp parallel for reduction(+ : total_calcs)
    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * d;

        // Compute query statistics
        float query_norm, query_mean;
        compute_stats(query, query_norm, query_mean);

        // Compute statistical scores for all vectors
        std::vector<std::pair<float, idx_t>> scores(ntotal);
        for (idx_t i = 0; i < ntotal; i++) {
            float norm_diff = std::abs(query_norm - norms[i]);
            float mean_diff = std::abs(query_mean - means[i]);
            float score = nw * norm_diff + mw * mean_diff * sqrt_d;
            scores[i] = {score, i};
        }
        // Only 2 comparisons per candidate (very cheap)

        // Partial sort to get top keep_n by score (ascending)
        std::partial_sort(
                scores.begin(),
                scores.begin() + keep_n,
                scores.end());

        // Compute L2 distances for candidates
        std::vector<std::pair<float, idx_t>> l2_dists;
        l2_dists.reserve(keep_n);

        for (idx_t i = 0; i < keep_n; i++) {
            idx_t cand = scores[i].second;
            float dist = fvec_L2sqr(query, vectors.data() + cand * d, d);
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

void IndexNeuroStatisticalPrescreen::reset() {
    vectors.clear();
    norms.clear();
    means.clear();
    ntotal = 0;
}

} // namespace faiss
