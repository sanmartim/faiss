/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroErrorDriven.h>
#include <faiss/impl/FaissAssert.h>

#include <algorithm>
#include <cmath>

namespace faiss {

IndexNeuroErrorDriven::IndexNeuroErrorDriven(Index* sub_index, bool own_fields)
        : Index(sub_index->d, sub_index->metric_type),
          sub_index(sub_index),
          own_fields(own_fields) {
    ntotal = sub_index->ntotal;
    is_trained = sub_index->is_trained;

    // Initialize weights to uniform 1.0
    refinement_weights.assign(d, 1.0f);
}

IndexNeuroErrorDriven::~IndexNeuroErrorDriven() {
    if (own_fields && sub_index) {
        delete sub_index;
    }
}

void IndexNeuroErrorDriven::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");
    sub_index->train(n, x);
    is_trained = sub_index->is_trained;
}

void IndexNeuroErrorDriven::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");
    sub_index->add(n, x);
    ntotal = sub_index->ntotal;
}

void IndexNeuroErrorDriven::reset() {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");
    sub_index->reset();
    ntotal = 0;
}

void IndexNeuroErrorDriven::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");
    sub_index->reconstruct(key, recons);
}

void IndexNeuroErrorDriven::apply_weights(
        const float* queries,
        idx_t n,
        std::vector<float>& weighted) const {
    weighted.resize(n * d);

    for (idx_t i = 0; i < n; i++) {
        const float* q = queries + i * d;
        float* w = weighted.data() + i * d;

        for (int j = 0; j < d; j++) {
            w[j] = q[j] * std::sqrt(refinement_weights[j]);
        }
    }
}

void IndexNeuroErrorDriven::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");

    // Apply refinement weights to queries
    std::vector<float> weighted;
    apply_weights(x, n, weighted);

    // Delegate to sub_index
    sub_index->search(n, weighted.data(), k, distances, labels, params);
}

void IndexNeuroErrorDriven::feedback(
        idx_t n_queries,
        const float* queries,
        const float* expected,
        const float* actual) {
    // For each query, compare dimensions between expected and actual
    // Increase weight for dimensions where expected was closer to query
    // Decrease weight for dimensions where actual was closer but incorrect

    for (idx_t q = 0; q < n_queries; q++) {
        const float* query = queries + q * d;
        const float* exp = expected + q * d;
        const float* act = actual + q * d;

        for (int j = 0; j < d; j++) {
            float diff_exp = std::abs(query[j] - exp[j]);
            float diff_act = std::abs(query[j] - act[j]);

            // If expected was closer, increase weight
            // If actual was closer (but wrong), decrease weight
            if (diff_exp < diff_act) {
                // Expected closer: good dimension for discrimination
                refinement_weights[j] += learning_rate;
            } else if (diff_exp > diff_act) {
                // Actual closer: misleading dimension
                refinement_weights[j] -= learning_rate;
            }

            // Apply bounds
            refinement_weights[j] = std::max(min_weight, refinement_weights[j]);
            refinement_weights[j] = std::min(max_weight, refinement_weights[j]);
        }
    }

    // Apply weight decay
    for (int j = 0; j < d; j++) {
        refinement_weights[j] *= weight_decay;
        // Re-apply floor after decay
        refinement_weights[j] = std::max(min_weight, refinement_weights[j]);
    }

    feedback_count++;
}

void IndexNeuroErrorDriven::feedback_binary(
        const float* query,
        const float* result,
        bool correct) {
    // For binary feedback:
    // If correct: slightly increase weights for matching dimensions
    // If incorrect: decrease weights where result differed from query

    float delta = correct ? learning_rate : -learning_rate;

    for (int j = 0; j < d; j++) {
        float diff = std::abs(query[j] - result[j]);

        // Dimensions with small difference get stronger signal
        float signal = 1.0f / (1.0f + diff);

        refinement_weights[j] += delta * signal;

        // Apply bounds
        refinement_weights[j] = std::max(min_weight, refinement_weights[j]);
        refinement_weights[j] = std::min(max_weight, refinement_weights[j]);
    }

    // Apply weight decay
    for (int j = 0; j < d; j++) {
        refinement_weights[j] *= weight_decay;
        refinement_weights[j] = std::max(min_weight, refinement_weights[j]);
    }

    feedback_count++;
}

void IndexNeuroErrorDriven::reset_weights() {
    refinement_weights.assign(d, 1.0f);
    feedback_count = 0;
}

} // namespace faiss
