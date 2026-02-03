/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroPatternCompletion.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>

namespace faiss {

namespace {

const float* get_flat_data_hpc(const Index* index) {
    auto flat = dynamic_cast<const IndexFlat*>(index);
    FAISS_THROW_IF_NOT_MSG(
            flat,
            "IndexNeuroPatternCompletion requires inner_index to be IndexFlat");
    return flat->get_xb();
}

} // anonymous namespace

IndexNeuroPatternCompletion::IndexNeuroPatternCompletion(
        Index* inner,
        int n_iterations,
        bool own_inner)
        : IndexNeuro(inner, own_inner), n_iterations(n_iterations) {}

void IndexNeuroPatternCompletion::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");
    FAISS_THROW_IF_NOT_MSG(d > 0, "dimension must be positive");

    // Compute mean and std per dimension
    dim_means.assign(d, 0.0f);
    dim_stds.assign(d, 1.0f);

    if (n > 0 && x != nullptr) {
        // Compute means
        for (idx_t i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                dim_means[j] += x[i * d + j];
            }
        }
        for (int j = 0; j < d; j++) {
            dim_means[j] /= n;
        }

        // Compute stds
        for (idx_t i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                float diff = x[i * d + j] - dim_means[j];
                dim_stds[j] += diff * diff;
            }
        }
        for (int j = 0; j < d; j++) {
            dim_stds[j] = std::sqrt(dim_stds[j] / n);
            if (dim_stds[j] < 1e-10f) {
                dim_stds[j] = 1.0f;
            }
        }

        // Compute association matrix (correlation-based Hopfield weights)
        // W[i,j] = sum_k (x_k[i] - mean[i]) * (x_k[j] - mean[j]) / (n * std[i] * std[j])
        association_matrix.assign(d * d, 0.0f);

        for (idx_t k = 0; k < n; k++) {
            for (int i = 0; i < d; i++) {
                float xi = (x[k * d + i] - dim_means[i]) / dim_stds[i];
                for (int j = 0; j < d; j++) {
                    float xj = (x[k * d + j] - dim_means[j]) / dim_stds[j];
                    association_matrix[i * d + j] += xi * xj;
                }
            }
        }

        // Normalize
        for (int i = 0; i < d * d; i++) {
            association_matrix[i] /= n;
        }

        // Zero diagonal (no self-connections in Hopfield)
        for (int i = 0; i < d; i++) {
            association_matrix[i * d + i] = 0.0f;
        }
    } else {
        // Identity-like initialization
        association_matrix.assign(d * d, 0.0f);
    }

    is_trained = true;
}

float IndexNeuroPatternCompletion::apply_activation(float x) const {
    if (activation == "tanh") {
        return std::tanh(x);
    } else if (activation == "relu") {
        return std::max(0.0f, x);
    } else {
        return x;  // linear
    }
}

void IndexNeuroPatternCompletion::complete_pattern(
        const float* x,
        const std::vector<bool>& mask,
        float* out,
        int n_iter) const {
    if (n_iter < 0) {
        n_iter = n_iterations;
    }

    // Initialize output
    // Known dims: use input value (normalized)
    // Unknown dims: use mean
    std::vector<float> state(d);
    for (int j = 0; j < d; j++) {
        if (mask.empty() || (j < static_cast<int>(mask.size()) && mask[j])) {
            // Known dimension
            if (normalize_input) {
                state[j] = (x[j] - dim_means[j]) / dim_stds[j];
            } else {
                state[j] = x[j];
            }
        } else {
            // Unknown dimension
            state[j] = 0.0f;  // Mean in normalized space
        }
    }

    // Iterative completion
    std::vector<float> new_state(d);
    for (int iter = 0; iter < n_iter; iter++) {
        for (int i = 0; i < d; i++) {
            // Skip known dimensions - keep their original values
            if (!mask.empty() && i < static_cast<int>(mask.size()) && mask[i]) {
                new_state[i] = state[i];
                continue;
            }

            // Update unknown dimension via association matrix
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += association_matrix[i * d + j] * state[j];
            }
            new_state[i] = apply_activation(sum);
        }
        state = new_state;
    }

    // Denormalize output
    for (int j = 0; j < d; j++) {
        if (normalize_input) {
            out[j] = state[j] * dim_stds[j] + dim_means[j];
        } else {
            out[j] = state[j];
        }
    }
}

void IndexNeuroPatternCompletion::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained");

    const float* data = get_flat_data_hpc(inner_index);
    idx_t nb = inner_index->ntotal;

    // Resolve parameters
    int n_iter = n_iterations;
    std::vector<bool> default_mask;  // empty = all known

    auto pp = dynamic_cast<const NeuroPatternCompletionParams*>(params);
    if (pp) {
        if (pp->n_iterations >= 0) {
            n_iter = pp->n_iterations;
        }
        if (!pp->mask.empty()) {
            default_mask = pp->mask;
        }
    }

    bool collect = false;
    if (params) {
        auto nsp = dynamic_cast<const NeuroSearchParameters*>(params);
        if (nsp) {
            collect = nsp->collect_stats;
        }
    }

    int64_t total_calcs = 0;

#pragma omp parallel for reduction(+ : total_calcs)
    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * d;

        // Complete the query pattern
        std::vector<float> completed(d);
        complete_pattern(query, default_mask, completed.data(), n_iter);

        // Search with completed query
        std::vector<std::pair<float, idx_t>> scored(nb);
        for (idx_t i = 0; i < nb; i++) {
            float dist;
            if (metric) {
                dist = metric->distance(completed.data(), data + i * d, d);
            } else {
                dist = fvec_L2sqr(completed.data(), data + i * d, d);
            }
            scored[i] = {dist, i};
            total_calcs += d;
        }

        // Sort and output
        size_t actual_k = std::min(static_cast<size_t>(k), static_cast<size_t>(nb));
        std::partial_sort(
                scored.begin(),
                scored.begin() + actual_k,
                scored.end());

        for (size_t i = 0; i < actual_k; i++) {
            distances[q * k + i] = scored[i].first;
            labels[q * k + i] = scored[i].second;
        }
        for (size_t i = actual_k; i < static_cast<size_t>(k); i++) {
            distances[q * k + i] = std::numeric_limits<float>::max();
            labels[q * k + i] = -1;
        }
    }

    if (collect) {
        last_stats.calculations_performed = total_calcs;
    }
}

} // namespace faiss
