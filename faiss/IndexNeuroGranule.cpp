/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroGranule.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <random>

namespace faiss {

namespace {

const float* get_flat_data_gr(const Index* index) {
    auto flat = dynamic_cast<const IndexFlat*>(index);
    FAISS_THROW_IF_NOT_MSG(
            flat, "IndexNeuroGranule requires inner_index to be IndexFlat");
    return flat->get_xb();
}

/// Sparse dot product
float sparse_dot(
        const std::vector<std::pair<int, float>>& a,
        const std::vector<std::pair<int, float>>& b) {
    float result = 0.0f;
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i].first < b[j].first) {
            i++;
        } else if (a[i].first > b[j].first) {
            j++;
        } else {
            result += a[i].second * b[j].second;
            i++;
            j++;
        }
    }
    return result;
}

} // anonymous namespace

IndexNeuroGranule::IndexNeuroGranule(Index* inner, int expansion, bool own_inner)
        : IndexNeuro(inner, own_inner), expansion(expansion) {}

void IndexNeuroGranule::train(idx_t /*n*/, const float* /*x*/) {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");
    FAISS_THROW_IF_NOT_MSG(d > 0, "dimension must be positive");

    n_granule = expansion * d;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> idx_dist(0, d - 1);
    std::exponential_distribution<float> weight_dist(1.0f);  // Non-negative

    // Initialize mossy fiber → granule projection
    mf_to_granule_indices.resize(n_granule * connections_per_granule);
    mf_to_granule_weights.resize(n_granule * connections_per_granule);

    for (int g = 0; g < n_granule; g++) {
        // Select random mossy fibers
        std::vector<int> selected;
        while (static_cast<int>(selected.size()) < connections_per_granule) {
            int idx = idx_dist(rng);
            if (std::find(selected.begin(), selected.end(), idx) == selected.end()) {
                selected.push_back(idx);
            }
        }

        for (int c = 0; c < connections_per_granule; c++) {
            mf_to_granule_indices[g * connections_per_granule + c] = selected[c];
            // Non-negative weights (exponential distribution)
            mf_to_granule_weights[g * connections_per_granule + c] = weight_dist(rng);
        }
    }

    // Normalize weights per granule
    for (int g = 0; g < n_granule; g++) {
        float sum = 0.0f;
        for (int c = 0; c < connections_per_granule; c++) {
            sum += mf_to_granule_weights[g * connections_per_granule + c];
        }
        if (sum > 0) {
            for (int c = 0; c < connections_per_granule; c++) {
                mf_to_granule_weights[g * connections_per_granule + c] /= sum;
            }
        }
    }

    // Initialize Purkinje weights uniformly
    purkinje_weights.assign(n_purkinje * n_granule, 1.0f / n_granule);

    granule_codes.clear();
    is_trained = true;
}

void IndexNeuroGranule::compute_granule_code(
        const float* x,
        std::vector<std::pair<int, float>>& code) const {
    code.clear();

    for (int g = 0; g < n_granule; g++) {
        float activation = 0.0f;
        for (int c = 0; c < connections_per_granule; c++) {
            int mf_idx = mf_to_granule_indices[g * connections_per_granule + c];
            float w = mf_to_granule_weights[g * connections_per_granule + c];
            activation += w * x[mf_idx];
        }

        // Apply threshold (ReLU-like)
        if (activation > threshold) {
            code.emplace_back(g, activation);
        }
    }

    // Sort by granule index for efficient sparse operations
    std::sort(code.begin(), code.end());
}

void IndexNeuroGranule::compute_purkinje_output(
        const std::vector<std::pair<int, float>>& code,
        std::vector<float>& output) const {
    output.assign(n_purkinje, 0.0f);

    for (int p = 0; p < n_purkinje; p++) {
        const float* weights = purkinje_weights.data() + p * n_granule;
        for (const auto& [g, act] : code) {
            output[p] += weights[g] * act;
        }
    }
}

void IndexNeuroGranule::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained before adding");

    IndexNeuro::add(n, x);

    for (idx_t i = 0; i < n; i++) {
        std::vector<std::pair<int, float>> code;
        compute_granule_code(x + i * d, code);
        granule_codes.push_back(std::move(code));
    }
}

void IndexNeuroGranule::reset() {
    IndexNeuro::reset();
    granule_codes.clear();
}

void IndexNeuroGranule::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained");

    const float* data = get_flat_data_gr(inner_index);
    idx_t nb = inner_index->ntotal;

    // Resolve parameters
    bool do_rerank = rerank;

    auto gp = dynamic_cast<const NeuroGranuleParams*>(params);
    if (gp) {
        do_rerank = gp->rerank;
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

        // Compute query granule code
        std::vector<std::pair<int, float>> query_code;
        compute_granule_code(query, query_code);

        // Compute Purkinje output for query
        std::vector<float> query_purkinje;
        compute_purkinje_output(query_code, query_purkinje);

        // Score all database vectors using sparse granule similarity
        std::vector<std::pair<float, idx_t>> scored(nb);

        for (idx_t i = 0; i < nb; i++) {
            // Sparse dot product of granule codes
            float granule_sim = sparse_dot(query_code, granule_codes[i]);

            // Purkinje output similarity
            std::vector<float> data_purkinje;
            compute_purkinje_output(granule_codes[i], data_purkinje);

            float purkinje_sim = 0.0f;
            for (int p = 0; p < n_purkinje; p++) {
                purkinje_sim += query_purkinje[p] * data_purkinje[p];
            }

            // Combined score (negative for sorting)
            scored[i] = {-(granule_sim + purkinje_sim), i};
        }

        // Get top candidates
        size_t n_cand = std::min(static_cast<size_t>(k * 3), static_cast<size_t>(nb));
        std::partial_sort(
                scored.begin(),
                scored.begin() + n_cand,
                scored.end());

        // Rerank if requested
        if (do_rerank) {
            for (size_t i = 0; i < n_cand; i++) {
                idx_t idx = scored[i].second;
                float dist;
                if (metric) {
                    dist = metric->distance(query, data + idx * d, d);
                } else {
                    dist = fvec_L2sqr(query, data + idx * d, d);
                }
                scored[i].first = dist;
                total_calcs += d;
            }
            std::partial_sort(
                    scored.begin(),
                    scored.begin() + std::min(n_cand, static_cast<size_t>(k)),
                    scored.end());
        }

        // Output results
        size_t actual_k = std::min(static_cast<size_t>(k), n_cand);
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

void IndexNeuroGranule::train_purkinje(
        idx_t n_samples,
        const float* x,
        const float* targets,
        int n_epochs) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained first");

    // Compute granule codes for all training samples
    std::vector<std::vector<std::pair<int, float>>> train_codes(n_samples);
    for (idx_t i = 0; i < n_samples; i++) {
        compute_granule_code(x + i * d, train_codes[i]);
    }

    // Gradient descent on Purkinje weights
    for (int epoch = 0; epoch < n_epochs; epoch++) {
        for (idx_t i = 0; i < n_samples; i++) {
            // Compute current output
            std::vector<float> output;
            compute_purkinje_output(train_codes[i], output);

            // Compute error and update weights
            for (int p = 0; p < n_purkinje; p++) {
                float error = targets[i * n_purkinje + p] - output[p];
                float* weights = purkinje_weights.data() + p * n_granule;

                for (const auto& [g, act] : train_codes[i]) {
                    weights[g] += learning_rate * error * act;
                }
            }
        }
    }
}

} // namespace faiss
