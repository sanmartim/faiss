/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroMissingValue.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace faiss {

IndexNeuroMissingValue::IndexNeuroMissingValue(
        Index* inner,
        NeuroMissingStrategy strategy,
        bool own_inner)
        : IndexNeuro(inner, own_inner), missing_strategy(strategy) {}

void IndexNeuroMissingValue::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");
    inner_index->train(n, x);
    is_trained = inner_index->is_trained;
}

namespace {

/// Get the raw float data pointer from the inner index.
const float* get_flat_data_mv(const Index* index) {
    auto flat = dynamic_cast<const IndexFlat*>(index);
    FAISS_THROW_IF_NOT_MSG(
            flat,
            "IndexNeuroMissingValue requires inner_index to be IndexFlat");
    return flat->get_xb();
}

/// Compute NaN-aware distance with the PROPORTIONAL strategy.
/// weight = (1 - missing_rate)
float nan_dist_proportional(
        const float* x,
        const float* y,
        int d) {
    float accu = 0.0f;
    int present = 0;
    for (int i = 0; i < d; i++) {
        if (!std::isnan(x[i]) && !std::isnan(y[i])) {
            float diff = x[i] - y[i];
            accu += diff * diff;
            present++;
        }
    }
    if (present == 0) {
        return std::numeric_limits<float>::max();
    }
    float missing_rate = 1.0f - static_cast<float>(present) / d;
    float weight = 1.0f - missing_rate; // = present/d
    return (static_cast<float>(d) / present) * accu * weight;
}

/// Compute NaN-aware distance with the THRESHOLD strategy.
/// Ignores dimensions where the per-pair missing rate exceeds threshold.
/// For this pair, a dimension is "missing" if either is NaN.
/// If the overall missing rate > threshold, return max distance.
float nan_dist_threshold(
        const float* x,
        const float* y,
        int d,
        float ignore_threshold) {
    float accu = 0.0f;
    int present = 0;
    for (int i = 0; i < d; i++) {
        if (!std::isnan(x[i]) && !std::isnan(y[i])) {
            float diff = x[i] - y[i];
            accu += diff * diff;
            present++;
        }
    }
    if (present == 0) {
        return std::numeric_limits<float>::max();
    }
    float missing_rate = 1.0f - static_cast<float>(present) / d;
    if (missing_rate > ignore_threshold) {
        return std::numeric_limits<float>::max();
    }
    // Simple renormalization, no additional weight reduction
    return (static_cast<float>(d) / present) * accu;
}

/// Compute NaN-aware distance with the HYBRID strategy.
/// weight = (1 - missing_rate)^2
float nan_dist_hybrid(
        const float* x,
        const float* y,
        int d) {
    float accu = 0.0f;
    int present = 0;
    for (int i = 0; i < d; i++) {
        if (!std::isnan(x[i]) && !std::isnan(y[i])) {
            float diff = x[i] - y[i];
            accu += diff * diff;
            present++;
        }
    }
    if (present == 0) {
        return std::numeric_limits<float>::max();
    }
    float missing_rate = 1.0f - static_cast<float>(present) / d;
    float weight = (1.0f - missing_rate) * (1.0f - missing_rate);
    return (static_cast<float>(d) / present) * accu * weight;
}

/// Search a single query with NaN-aware distances.
void search_one_missing(
        const float* query,
        const float* data,
        idx_t ntotal,
        int d,
        idx_t k,
        float* out_distances,
        idx_t* out_labels,
        NeuroMissingStrategy strategy,
        float ignore_threshold) {
    std::vector<std::pair<float, idx_t>> scores(ntotal);

    for (idx_t i = 0; i < ntotal; i++) {
        float dist;
        const float* vec = data + i * d;
        switch (strategy) {
            case NEURO_MISSING_PROPORTIONAL:
                dist = nan_dist_proportional(query, vec, d);
                break;
            case NEURO_MISSING_THRESHOLD:
                dist = nan_dist_threshold(query, vec, d, ignore_threshold);
                break;
            case NEURO_MISSING_HYBRID:
                dist = nan_dist_hybrid(query, vec, d);
                break;
        }
        scores[i] = {dist, i};
    }

    idx_t actual_k = std::min(k, ntotal);
    std::partial_sort(
            scores.begin(),
            scores.begin() + actual_k,
            scores.end());

    for (idx_t i = 0; i < actual_k; i++) {
        out_distances[i] = scores[i].first;
        out_labels[i] = scores[i].second;
    }
    for (idx_t i = actual_k; i < k; i++) {
        out_distances[i] = std::numeric_limits<float>::max();
        out_labels[i] = -1;
    }
}

} // anonymous namespace

void IndexNeuroMissingValue::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");

    const float* data = get_flat_data_mv(inner_index);
    idx_t nb = inner_index->ntotal;

    NeuroMissingStrategy strat = missing_strategy;
    float thresh = ignore_threshold;

    auto mp = dynamic_cast<const NeuroMissingValueParams*>(params);
    if (mp) {
        strat = mp->missing_strategy;
        thresh = mp->ignore_threshold;
    }

#pragma omp parallel for if (n > 1)
    for (idx_t i = 0; i < n; i++) {
        search_one_missing(
                x + i * d,
                data,
                nb,
                d,
                k,
                distances + i * k,
                labels + i * k,
                strat,
                thresh);
    }
}

} // namespace faiss
