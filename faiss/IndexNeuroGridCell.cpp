/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroGridCell.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <random>

namespace faiss {

namespace {

const float* get_flat_data_gc(const Index* index) {
    auto flat = dynamic_cast<const IndexFlat*>(index);
    FAISS_THROW_IF_NOT_MSG(
            flat, "IndexNeuroGridCell requires inner_index to be IndexFlat");
    return flat->get_xb();
}

} // anonymous namespace

IndexNeuroGridCell::IndexNeuroGridCell(
        Index* inner,
        int n_scales,
        bool own_inner)
        : IndexNeuro(inner, own_inner), n_scales(n_scales) {
    // Default scale factors: 1, 2, 4, 8 (progressively coarser)
    scale_factors.resize(n_scales);
    for (int s = 0; s < n_scales; s++) {
        scale_factors[s] = static_cast<float>(1 << s);
    }

    // Default uniform weights
    scale_weights.assign(n_scales, 1.0f / n_scales);
}

void IndexNeuroGridCell::set_scale_factors(const std::vector<float>& factors) {
    FAISS_THROW_IF_NOT_MSG(
            static_cast<int>(factors.size()) == n_scales,
            "factors size must match n_scales");
    scale_factors = factors;
}

void IndexNeuroGridCell::train(idx_t /*n*/, const float* /*x*/) {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");
    FAISS_THROW_IF_NOT_MSG(d > 0, "dimension must be positive");

    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    scale_projections.resize(n_scales);
    scale_data.resize(n_scales);
    scale_dims.resize(n_scales);

    for (int s = 0; s < n_scales; s++) {
        // Compute output dimension for this scale
        int out_d = std::max(1, static_cast<int>(d / scale_factors[s]));
        scale_dims[s] = out_d;

        // Generate random projection matrix: out_d * d
        scale_projections[s].resize(out_d * d);
        for (int i = 0; i < out_d * d; i++) {
            scale_projections[s][i] = normal(rng);
        }

        // Normalize rows if requested
        if (normalize_projections) {
            for (int r = 0; r < out_d; r++) {
                float* row = scale_projections[s].data() + r * d;
                float norm = 0.0f;
                for (int j = 0; j < d; j++) {
                    norm += row[j] * row[j];
                }
                norm = std::sqrt(norm);
                if (norm > 1e-10f) {
                    for (int j = 0; j < d; j++) {
                        row[j] /= norm;
                    }
                }
            }
        }

        scale_data[s].clear();
    }

    is_trained = true;
}

void IndexNeuroGridCell::project_to_scale(
        int scale,
        const float* x,
        float* out) const {
    int out_d = scale_dims[scale];
    const float* proj = scale_projections[scale].data();

    for (int r = 0; r < out_d; r++) {
        float sum = 0.0f;
        const float* row = proj + r * d;
        for (int j = 0; j < d; j++) {
            sum += row[j] * x[j];
        }
        out[r] = sum;
    }
}

void IndexNeuroGridCell::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained before adding");

    IndexNeuro::add(n, x);

    // Project all new vectors to each scale
    for (int s = 0; s < n_scales; s++) {
        int out_d = scale_dims[s];
        size_t old_size = scale_data[s].size();
        scale_data[s].resize(old_size + n * out_d);

        for (idx_t i = 0; i < n; i++) {
            project_to_scale(
                    s,
                    x + i * d,
                    scale_data[s].data() + old_size + i * out_d);
        }
    }
}

void IndexNeuroGridCell::reset() {
    IndexNeuro::reset();
    for (auto& data : scale_data) {
        data.clear();
    }
}

void IndexNeuroGridCell::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained");

    idx_t nb = inner_index->ntotal;

    // Resolve parameters
    std::vector<float> weights = scale_weights;

    auto gp = dynamic_cast<const NeuroGridCellParams*>(params);
    if (gp && !gp->scale_weights.empty()) {
        FAISS_THROW_IF_NOT_MSG(
                static_cast<int>(gp->scale_weights.size()) == n_scales,
                "scale_weights size must match n_scales");
        weights = gp->scale_weights;
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

        // Project query to all scales
        std::vector<std::vector<float>> query_proj(n_scales);
        for (int s = 0; s < n_scales; s++) {
            query_proj[s].resize(scale_dims[s]);
            project_to_scale(s, query, query_proj[s].data());
        }

        // Compute weighted combined distance to all database vectors
        std::vector<std::pair<float, idx_t>> scored(nb);

        for (idx_t i = 0; i < nb; i++) {
            float combined_dist = 0.0f;

            for (int s = 0; s < n_scales; s++) {
                int out_d = scale_dims[s];
                const float* data_proj = scale_data[s].data() + i * out_d;

                // L2 distance in projected space
                float dist = 0.0f;
                for (int j = 0; j < out_d; j++) {
                    float diff = query_proj[s][j] - data_proj[j];
                    dist += diff * diff;
                }

                combined_dist += weights[s] * dist;
                total_calcs += out_d;
            }

            scored[i] = {combined_dist, i};
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
