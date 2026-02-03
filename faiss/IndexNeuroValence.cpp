/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroValence.h>
#include <faiss/impl/FaissAssert.h>

#include <algorithm>
#include <cmath>

namespace faiss {

IndexNeuroValence::IndexNeuroValence(
        Index* sub_index,
        int n_valences,
        bool own_fields)
        : Index(sub_index->d, sub_index->metric_type),
          sub_index(sub_index),
          own_fields(own_fields),
          n_valences(n_valences) {
    ntotal = sub_index->ntotal;
    is_trained = sub_index->is_trained;

    // Initialize all valences to neutral (uniform weights of 1.0)
    valence_weights.assign(n_valences * d, 1.0f);
}

IndexNeuroValence::~IndexNeuroValence() {
    if (own_fields && sub_index) {
        delete sub_index;
    }
}

void IndexNeuroValence::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");
    sub_index->train(n, x);
    is_trained = sub_index->is_trained;
}

void IndexNeuroValence::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");
    sub_index->add(n, x);
    ntotal = sub_index->ntotal;
}

void IndexNeuroValence::reset() {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");
    sub_index->reset();
    ntotal = 0;
}

void IndexNeuroValence::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");
    sub_index->reconstruct(key, recons);
}

void IndexNeuroValence::set_active_valence(int valence) {
    FAISS_THROW_IF_NOT_FMT(
            valence >= 0 && valence < n_valences,
            "valence %d out of range [0, %d)",
            valence,
            n_valences);
    active_valence = valence;
}

float* IndexNeuroValence::get_valence_weights(int valence) {
    FAISS_THROW_IF_NOT_FMT(
            valence >= 0 && valence < n_valences,
            "valence %d out of range [0, %d)",
            valence,
            n_valences);
    return valence_weights.data() + valence * d;
}

const float* IndexNeuroValence::get_valence_weights(int valence) const {
    FAISS_THROW_IF_NOT_FMT(
            valence >= 0 && valence < n_valences,
            "valence %d out of range [0, %d)",
            valence,
            n_valences);
    return valence_weights.data() + valence * d;
}

void IndexNeuroValence::apply_valence(
        const float* queries,
        idx_t n,
        std::vector<float>& transformed) const {
    const float* weights = get_valence_weights(active_valence);

    transformed.resize(n * d);

    for (idx_t i = 0; i < n; i++) {
        const float* q = queries + i * d;
        float* t = transformed.data() + i * d;

        if (multiply_mode) {
            for (int j = 0; j < d; j++) {
                t[j] = q[j] * weights[j];
            }
        } else {
            for (int j = 0; j < d; j++) {
                t[j] = q[j] + weights[j];
            }
        }
    }
}

void IndexNeuroValence::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");

    // Transform queries with active valence
    std::vector<float> transformed;
    apply_valence(x, n, transformed);

    // Delegate to sub_index
    sub_index->search(n, transformed.data(), k, distances, labels, params);
}

void IndexNeuroValence::learn_valence(
        int valence,
        idx_t n,
        const float* positive,
        const float* negative) {
    FAISS_THROW_IF_NOT_FMT(
            valence >= 0 && valence < n_valences,
            "valence %d out of range [0, %d)",
            valence,
            n_valences);

    float* weights = get_valence_weights(valence);

    // Compute mean of positive and negative examples
    std::vector<double> pos_mean(d, 0.0);
    std::vector<double> neg_mean(d, 0.0);

    if (positive && n > 0) {
        for (idx_t i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                pos_mean[j] += positive[i * d + j];
            }
        }
        for (int j = 0; j < d; j++) {
            pos_mean[j] /= n;
        }
    }

    if (negative && n > 0) {
        for (idx_t i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                neg_mean[j] += negative[i * d + j];
            }
        }
        for (int j = 0; j < d; j++) {
            neg_mean[j] /= n;
        }
    }

    // Update weights: dimensions where positive > negative get increased
    for (int j = 0; j < d; j++) {
        // Apply weight decay
        weights[j] *= weight_decay;

        // Gradient: positive mean - negative mean
        float grad = static_cast<float>(pos_mean[j] - neg_mean[j]);
        weights[j] += learning_rate * grad;

        // Keep weights positive (for multiply mode)
        if (multiply_mode) {
            weights[j] = std::max(0.01f, weights[j]);
        }
    }
}

void IndexNeuroValence::update_valence(
        int valence,
        const float* example,
        bool is_positive) {
    FAISS_THROW_IF_NOT_FMT(
            valence >= 0 && valence < n_valences,
            "valence %d out of range [0, %d)",
            valence,
            n_valences);

    float* weights = get_valence_weights(valence);
    float sign = is_positive ? 1.0f : -1.0f;

    for (int j = 0; j < d; j++) {
        // Apply weight decay
        weights[j] *= weight_decay;

        // Update based on example value
        // For positive: high example values increase weight
        // For negative: high example values decrease weight
        weights[j] += sign * learning_rate * std::abs(example[j]);

        // Keep weights positive (for multiply mode)
        if (multiply_mode) {
            weights[j] = std::max(0.01f, weights[j]);
        }
    }
}

void IndexNeuroValence::reset_valence(int valence) {
    FAISS_THROW_IF_NOT_FMT(
            valence >= 0 && valence < n_valences,
            "valence %d out of range [0, %d)",
            valence,
            n_valences);

    float* weights = get_valence_weights(valence);
    for (int j = 0; j < d; j++) {
        weights[j] = 1.0f;
    }
}

void IndexNeuroValence::reset_all_valences() {
    for (int v = 0; v < n_valences; v++) {
        reset_valence(v);
    }
}

} // namespace faiss
