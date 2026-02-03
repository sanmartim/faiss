/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/NeuroDistance.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>

namespace faiss {

/*************************************************************
 * MT-00: NeuroMetric implementations
 *************************************************************/

void NeuroMetric::distance_batch(
        const float* query,
        const float* data,
        idx_t n,
        int d,
        float* out) const {
    // Default implementation: call distance() for each vector
    for (idx_t i = 0; i < n; i++) {
        out[i] = distance(query, data + i * d, d);
    }
}

// NeuroMetricL2

float NeuroMetricL2::distance(const float* x1, const float* x2, int d) const {
    return fvec_L2sqr(x1, x2, d);
}

void NeuroMetricL2::distance_batch(
        const float* query,
        const float* data,
        idx_t n,
        int d,
        float* out) const {
    // Use FAISS optimized batch L2
    fvec_L2sqr_ny(out, query, data, d, n);
}

// NeuroMetricCosine

float NeuroMetricCosine::distance(
        const float* x1,
        const float* x2,
        int d) const {
    float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
    for (int i = 0; i < d; i++) {
        dot += x1[i] * x2[i];
        norm1 += x1[i] * x1[i];
        norm2 += x2[i] * x2[i];
    }
    float denom = std::sqrt(norm1) * std::sqrt(norm2);
    if (denom < 1e-10f) {
        return 1.0f;  // Maximum distance for zero vectors
    }
    float cosine_sim = dot / denom;
    // Clamp to [-1, 1] to handle numerical errors
    cosine_sim = std::max(-1.0f, std::min(1.0f, cosine_sim));
    return 1.0f - cosine_sim;
}

// NeuroMetricDot

float NeuroMetricDot::distance(const float* x1, const float* x2, int d) const {
    float dot = 0.0f;
    for (int i = 0; i < d; i++) {
        dot += x1[i] * x2[i];
    }
    return dot;  // Note: higher is better for this metric
}

// NeuroMetricMahalanobis

NeuroMetricMahalanobis::NeuroMetricMahalanobis(
        const std::vector<float>& variances) {
    inv_variances.resize(variances.size());
    for (size_t i = 0; i < variances.size(); i++) {
        float var = std::max(variances[i], 1e-10f);  // Prevent division by zero
        inv_variances[i] = 1.0f / var;
    }
}

void NeuroMetricMahalanobis::fit(const float* data, idx_t n, int d) {
    if (n == 0) {
        inv_variances.assign(d, 1.0f);
        return;
    }

    // Compute mean per dimension
    std::vector<double> mean(d, 0.0);
    for (idx_t i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            mean[j] += data[i * d + j];
        }
    }
    for (int j = 0; j < d; j++) {
        mean[j] /= n;
    }

    // Compute variance per dimension
    std::vector<double> var(d, 0.0);
    for (idx_t i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            double diff = data[i * d + j] - mean[j];
            var[j] += diff * diff;
        }
    }

    inv_variances.resize(d);
    for (int j = 0; j < d; j++) {
        var[j] /= n;
        float v = std::max(static_cast<float>(var[j]), 1e-10f);
        inv_variances[j] = 1.0f / v;
    }
}

float NeuroMetricMahalanobis::distance(
        const float* x1,
        const float* x2,
        int d) const {
    FAISS_THROW_IF_NOT_MSG(
            static_cast<int>(inv_variances.size()) == d,
            "Mahalanobis metric not initialized or dimension mismatch");
    float sum = 0.0f;
    for (int i = 0; i < d; i++) {
        float diff = x1[i] - x2[i];
        sum += inv_variances[i] * diff * diff;
    }
    return sum;
}

// NeuroMetricJaccard

float NeuroMetricJaccard::distance(
        const float* x1,
        const float* x2,
        int d) const {
    // Jaccard for binary vectors: 1 - |intersection| / |union|
    int intersection = 0;
    int union_count = 0;
    for (int i = 0; i < d; i++) {
        bool b1 = x1[i] > 0.5f;
        bool b2 = x2[i] > 0.5f;
        if (b1 && b2) {
            intersection++;
        }
        if (b1 || b2) {
            union_count++;
        }
    }
    if (union_count == 0) {
        return 0.0f;  // Both vectors are zero
    }
    return 1.0f - static_cast<float>(intersection) / union_count;
}

/*************************************************************
 * IndexNeuro base class
 *************************************************************/

IndexNeuro::IndexNeuro(Index* inner_index, bool own_inner)
        : inner_index(inner_index),
          own_inner(own_inner) {
    if (inner_index) {
        d = inner_index->d;
        metric_type = inner_index->metric_type;
        ntotal = inner_index->ntotal;
        is_trained = inner_index->is_trained;
        metric_arg = inner_index->metric_arg;
    }
}

void IndexNeuro::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");
    inner_index->add(n, x);
    ntotal = inner_index->ntotal;
    is_trained = inner_index->is_trained;
}

void IndexNeuro::reset() {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");
    inner_index->reset();
    ntotal = 0;
}

void IndexNeuro::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");
    inner_index->reconstruct(key, recons);
}

IndexNeuro::~IndexNeuro() {
    if (own_inner) {
        delete inner_index;
    }
}

} // namespace faiss
