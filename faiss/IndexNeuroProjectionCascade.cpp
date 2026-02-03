/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroProjectionCascade.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace faiss {

IndexNeuroProjectionCascade::IndexNeuroProjectionCascade(int d)
        : IndexNeuro(nullptr, false) {
    this->d = d;
    is_trained = false;
}

void IndexNeuroProjectionCascade::init_projection_matrices() {
    std::mt19937 rng(random_seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    projection_matrices.resize(projection_dims.size());
    projected_data.resize(projection_dims.size());

    for (size_t level = 0; level < projection_dims.size(); level++) {
        int target_d = projection_dims[level];
        projection_matrices[level].resize(d * target_d);

        // Random Gaussian projection matrix
        for (int i = 0; i < d * target_d; i++) {
            projection_matrices[level][i] = normal(rng) / std::sqrt(static_cast<float>(target_d));
        }
    }
}

void IndexNeuroProjectionCascade::project(
        const float* vec,
        int level,
        float* out) const {
    int target_d = projection_dims[level];
    const float* matrix = projection_matrices[level].data();

    // out = matrix^T * vec (target_d outputs)
    for (int i = 0; i < target_d; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            sum += matrix[j * target_d + i] * vec[j];
        }
        out[i] = sum;
    }
}

void IndexNeuroProjectionCascade::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(n > 0);

    // Initialize projection matrices
    init_projection_matrices();

    // Store original vectors
    vectors.resize(n * d);
    std::copy(x, x + n * d, vectors.data());

    // Project all data at each level
    for (size_t level = 0; level < projection_dims.size(); level++) {
        int target_d = projection_dims[level];
        projected_data[level].resize(n * target_d);

        for (idx_t i = 0; i < n; i++) {
            project(x + i * d, level, projected_data[level].data() + i * target_d);
        }
    }

    ntotal = n;
    is_trained = true;
}

void IndexNeuroProjectionCascade::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "Index must be trained first");

    idx_t old_ntotal = ntotal;

    // Extend vectors
    vectors.resize((old_ntotal + n) * d);
    std::copy(x, x + n * d, vectors.data() + old_ntotal * d);

    // Project new data
    for (size_t level = 0; level < projection_dims.size(); level++) {
        int target_d = projection_dims[level];
        projected_data[level].resize((old_ntotal + n) * target_d);

        for (idx_t i = 0; i < n; i++) {
            project(
                    x + i * d,
                    level,
                    projected_data[level].data() + (old_ntotal + i) * target_d);
        }
    }

    ntotal += n;
}

void IndexNeuroProjectionCascade::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(ntotal > 0);

    // Parse parameters
    std::vector<float> kr = keep_ratios;

    auto* pcp = dynamic_cast<const NeuroProjectionCascadeParams*>(params);
    if (pcp) {
        if (!pcp->keep_ratios.empty()) {
            kr = pcp->keep_ratios;
        }
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

        // Project query at each level
        std::vector<std::vector<float>> query_proj(projection_dims.size());
        for (size_t level = 0; level < projection_dims.size(); level++) {
            query_proj[level].resize(projection_dims[level]);
            project(query, level, query_proj[level].data());
        }

        // Start with all candidates
        std::vector<idx_t> candidates(ntotal);
        for (idx_t i = 0; i < ntotal; i++) {
            candidates[i] = i;
        }

        // Cascade through projection levels
        for (size_t level = 0; level < projection_dims.size() && !candidates.empty(); level++) {
            int target_d = projection_dims[level];
            float ratio = (level < kr.size()) ? kr[level] : 0.1f;

            // Compute projected distances
            std::vector<std::pair<float, idx_t>> proj_dists;
            proj_dists.reserve(candidates.size());

            for (idx_t cand : candidates) {
                const float* proj_vec =
                        projected_data[level].data() + cand * target_d;
                float dist = fvec_L2sqr(query_proj[level].data(), proj_vec, target_d);
                proj_dists.push_back({dist, cand});
            }
            total_calcs += candidates.size() * target_d;

            // Sort and keep top ratio
            size_t keep_n = std::max(
                    static_cast<size_t>(candidates.size() * ratio),
                    static_cast<size_t>(k));

            std::partial_sort(
                    proj_dists.begin(),
                    proj_dists.begin() + std::min(keep_n, proj_dists.size()),
                    proj_dists.end());

            candidates.clear();
            for (size_t i = 0; i < std::min(keep_n, proj_dists.size()); i++) {
                candidates.push_back(proj_dists[i].second);
            }
        }

        // Final L2 on survivors
        std::vector<std::pair<float, idx_t>> l2_dists;
        l2_dists.reserve(candidates.size());

        for (idx_t cand : candidates) {
            float dist = fvec_L2sqr(query, vectors.data() + cand * d, d);
            l2_dists.push_back({dist, cand});
        }
        total_calcs += candidates.size() * d;

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

void IndexNeuroProjectionCascade::reset() {
    vectors.clear();
    for (auto& proj : projected_data) {
        proj.clear();
    }
    ntotal = 0;
}

} // namespace faiss
