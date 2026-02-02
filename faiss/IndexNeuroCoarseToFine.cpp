/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroCoarseToFine.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace faiss {

IndexNeuroCoarseToFine::IndexNeuroCoarseToFine(
        Index* inner,
        int num_levels,
        bool own_inner)
        : IndexNeuro(inner, own_inner), num_levels(num_levels) {
    // Default cutoffs: keep 30% at level 0, 50% at level 1, all at level 2
    cutoff_per_level = {0.3f, 0.5f, 1.0f};
    if (num_levels > 3) {
        cutoff_per_level.resize(num_levels, 1.0f);
        for (int i = 0; i < num_levels - 1; i++) {
            cutoff_per_level[i] =
                    0.3f + 0.7f * i / (num_levels - 1);
        }
    }
}

void IndexNeuroCoarseToFine::compute_coarse(
        const float* vec,
        int level,
        std::vector<float>& out) const {
    int cd = coarse_dims[level];
    out.resize(cd);

    if (cd == d) {
        // Full resolution
        std::copy(vec, vec + d, out.data());
        return;
    }

    // Average groups of dimensions
    int group_size = d / cd;
    int remainder = d % cd;

    int src = 0;
    for (int i = 0; i < cd; i++) {
        int gs = group_size + (i < remainder ? 1 : 0);
        float sum = 0.0f;
        for (int j = 0; j < gs; j++) {
            sum += vec[src++];
        }
        out[i] = sum / gs;
    }
}

void IndexNeuroCoarseToFine::train(idx_t /*n*/, const float* /*x*/) {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is null");

    auto* flat = dynamic_cast<IndexFlat*>(inner_index);
    FAISS_THROW_IF_NOT_MSG(flat, "inner_index must be IndexFlat");

    idx_t nb = inner_index->ntotal;
    FAISS_THROW_IF_NOT_MSG(nb > 0, "inner_index must have data");

    const float* xb = flat->get_xb();

    // Compute coarse dimensions per level
    // Level 0: d/4 (or min 2), Level 1: d/2 (or min 4), Level 2: d
    coarse_dims.resize(num_levels);
    for (int level = 0; level < num_levels; level++) {
        if (level == num_levels - 1) {
            coarse_dims[level] = d;
        } else {
            int divisor = 1 << (num_levels - 1 - level);
            coarse_dims[level] = std::max(d / divisor, 2);
        }
    }

    // Ensure cutoffs vector is right size
    while ((int)cutoff_per_level.size() < num_levels) {
        cutoff_per_level.push_back(1.0f);
    }

    // Precompute coarse representations for the database
    coarse_data.resize(num_levels);
    for (int level = 0; level < num_levels - 1; level++) {
        int cd = coarse_dims[level];
        coarse_data[level].resize(nb * cd);

        std::vector<float> coarse_vec;
        for (idx_t i = 0; i < nb; i++) {
            compute_coarse(xb + i * d, level, coarse_vec);
            std::copy(
                    coarse_vec.begin(),
                    coarse_vec.end(),
                    coarse_data[level].data() + i * cd);
        }
    }
    // Level num_levels-1 uses raw data (no precomputation needed)

    is_trained = true;
}

void IndexNeuroCoarseToFine::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is null");
    FAISS_THROW_IF_NOT_MSG(!coarse_data.empty(), "train() must be called first");

    auto* flat = dynamic_cast<IndexFlat*>(inner_index);
    FAISS_THROW_IF_NOT_MSG(flat, "inner_index must be IndexFlat");

    const float* xb = flat->get_xb();
    idx_t nb = inner_index->ntotal;

    int nl = num_levels;
    std::vector<float> cutoffs = cutoff_per_level;

    if (params_in) {
        auto* cfp = dynamic_cast<const NeuroCoarseToFineParams*>(params_in);
        if (cfp) {
            if (cfp->num_levels > 0)
                nl = cfp->num_levels;
            if (!cfp->cutoff_per_level.empty())
                cutoffs = cfp->cutoff_per_level;
        }
    }
    nl = std::min(nl, num_levels);

    bool collect = false;
    if (params_in) {
        auto* nsp = dynamic_cast<const NeuroSearchParameters*>(params_in);
        if (nsp)
            collect = nsp->collect_stats;
    }

    int64_t total_calcs = 0;

#pragma omp parallel for reduction(+ : total_calcs)
    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * d;

        // Start with all candidates
        std::vector<idx_t> candidates(nb);
        std::iota(candidates.begin(), candidates.end(), 0);

        // Progressive refinement through levels
        for (int level = 0; level < nl; level++) {
            int cd = coarse_dims[level];
            float cutoff = (level < (int)cutoffs.size()) ? cutoffs[level] : 1.0f;
            int keep = std::max(
                    (int)(candidates.size() * cutoff), (int)k);

            if (level < nl - 1) {
                // Coarse level: use precomputed coarse representations
                std::vector<float> q_coarse;
                compute_coarse(query, level, q_coarse);

                std::vector<std::pair<float, idx_t>> dists(candidates.size());
                for (size_t ci = 0; ci < candidates.size(); ci++) {
                    idx_t id = candidates[ci];
                    const float* cv =
                            coarse_data[level].data() + id * cd;
                    float dist = 0.0f;
                    for (int j = 0; j < cd; j++) {
                        float diff = q_coarse[j] - cv[j];
                        dist += diff * diff;
                    }
                    dists[ci] = {dist, id};
                }
                total_calcs += (int64_t)candidates.size() * cd;

                std::partial_sort(
                        dists.begin(),
                        dists.begin() + std::min(keep, (int)dists.size()),
                        dists.end());

                candidates.resize(std::min(keep, (int)dists.size()));
                for (int i = 0; i < (int)candidates.size(); i++) {
                    candidates[i] = dists[i].second;
                }
            } else {
                // Final level: full resolution L2
                std::vector<std::pair<float, idx_t>> dists(candidates.size());
                for (size_t ci = 0; ci < candidates.size(); ci++) {
                    idx_t id = candidates[ci];
                    float dist = fvec_L2sqr(query, xb + id * d, d);
                    dists[ci] = {dist, id};
                }
                total_calcs += (int64_t)candidates.size() * d;

                idx_t nk = std::min(k, (idx_t)dists.size());
                std::partial_sort(
                        dists.begin(),
                        dists.begin() + nk,
                        dists.end());

                for (idx_t i = 0; i < nk; i++) {
                    distances[q * k + i] = dists[i].first;
                    labels[q * k + i] = dists[i].second;
                }
                for (idx_t i = nk; i < k; i++) {
                    distances[q * k + i] =
                            std::numeric_limits<float>::max();
                    labels[q * k + i] = -1;
                }
            }
        }
    }

    if (collect) {
        last_stats.calculations_performed = total_calcs;
        last_stats.columns_used = d;
    }
}

} // namespace faiss
