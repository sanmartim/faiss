/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroParallelVoting.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <vector>

namespace faiss {

IndexNeuroParallelVoting::IndexNeuroParallelVoting(
        Index* inner,
        int num_groups,
        bool own_inner)
        : IndexNeuro(inner, own_inner), num_groups(num_groups) {}

void IndexNeuroParallelVoting::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is null");

    auto* flat = dynamic_cast<IndexFlat*>(inner_index);
    FAISS_THROW_IF_NOT_MSG(flat, "inner_index must be IndexFlat");

    const float* xb = flat->get_xb();
    idx_t nb = inner_index->ntotal;

    // Resolve params
    int ng = num_groups;
    int topk_pg = top_k_per_group;
    NeuroGroupingMethod grp = grouping;
    NeuroIntegrationMethod integ = integration;

    if (params_in) {
        auto* pvp = dynamic_cast<const NeuroParallelVotingParams*>(params_in);
        if (pvp) {
            if (pvp->num_groups > 0)
                ng = pvp->num_groups;
            if (pvp->top_k_per_group > 0)
                topk_pg = pvp->top_k_per_group;
            grp = pvp->grouping;
            integ = pvp->integration;
        }
    }

    ng = std::min(ng, (int)d);
    if (ng < 1)
        ng = 1;
    if (topk_pg <= 0)
        topk_pg = std::max((int)k * 3, 20);

    // Build dimension groups
    std::vector<std::vector<int>> groups(ng);
    for (int j = 0; j < d; j++) {
        int g;
        if (grp == NEURO_GROUP_INTERLEAVED) {
            g = j % ng;
        } else {
            g = j * ng / d;
        }
        groups[g].push_back(j);
    }

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

        // Per-group: compute partial distances, get top-k candidates
        // candidates[candidate_id] -> (votes, borda_score, mean_dist_sum,
        // count)
        struct CandInfo {
            int votes = 0;
            float borda_score = 0;
            float dist_sum = 0;
            int count = 0;
        };
        std::unordered_map<idx_t, CandInfo> candidates;

        for (int g = 0; g < ng; g++) {
            const auto& dims = groups[g];
            if (dims.empty())
                continue;

            // Compute partial L2 for this group
            std::vector<std::pair<float, idx_t>> partial(nb);
            for (idx_t i = 0; i < nb; i++) {
                const float* vec = xb + i * d;
                float dist = 0.0f;
                for (int dim_idx : dims) {
                    float diff = query[dim_idx] - vec[dim_idx];
                    dist += diff * diff;
                }
                partial[i] = {dist, i};
            }
            total_calcs += nb * (int64_t)dims.size();

            int topk = std::min(topk_pg, (int)nb);
            std::partial_sort(
                    partial.begin(), partial.begin() + topk, partial.end());

            for (int i = 0; i < topk; i++) {
                auto& c = candidates[partial[i].second];
                c.votes++;
                c.borda_score += (float)(topk - i); // higher rank = more
                c.dist_sum += partial[i].first;
                c.count++;
            }
        }

        // Integrate and select top-k overall
        std::vector<std::pair<float, idx_t>> final_cands;
        final_cands.reserve(candidates.size());

        if (integ == NEURO_INTEGRATE_FULL_RERANK) {
            // Full rerank: compute exact L2 on all candidates
            for (auto& [id, info] : candidates) {
                const float* vec = xb + id * d;
                float dist = fvec_L2sqr(query, vec, d);
                final_cands.push_back({dist, id});
            }
            total_calcs += (int64_t)candidates.size() * d;
        } else {
            for (auto& [id, info] : candidates) {
                float score;
                switch (integ) {
                    case NEURO_INTEGRATE_VOTING:
                        score = -(float)info.votes; // negate: more votes =
                                                     // smaller score
                        break;
                    case NEURO_INTEGRATE_BORDA:
                        score = -info.borda_score;
                        break;
                    case NEURO_INTEGRATE_MEAN_DIST:
                        score = info.dist_sum /
                                std::max(info.count, 1);
                        break;
                    default:
                        score = -info.borda_score;
                        break;
                }
                final_cands.push_back({score, id});
            }
        }

        idx_t nk = std::min(k, (idx_t)final_cands.size());
        std::partial_sort(
                final_cands.begin(),
                final_cands.begin() + nk,
                final_cands.end());

        // For non-rerank methods, recompute exact L2 for final distances
        for (idx_t i = 0; i < nk; i++) {
            labels[q * k + i] = final_cands[i].second;
            if (integ == NEURO_INTEGRATE_FULL_RERANK) {
                distances[q * k + i] = final_cands[i].first;
            } else {
                const float* vec = xb + final_cands[i].second * d;
                distances[q * k + i] = fvec_L2sqr(query, vec, d);
            }
        }
        // Fill remaining
        for (idx_t i = nk; i < k; i++) {
            distances[q * k + i] = std::numeric_limits<float>::max();
            labels[q * k + i] = -1;
        }

        // Re-sort by actual distance
        // (integration order may differ from distance order)
        std::vector<std::pair<float, idx_t>> final_sorted(k);
        for (idx_t i = 0; i < k; i++) {
            final_sorted[i] = {distances[q * k + i], labels[q * k + i]};
        }
        std::sort(final_sorted.begin(), final_sorted.end());
        for (idx_t i = 0; i < k; i++) {
            distances[q * k + i] = final_sorted[i].first;
            labels[q * k + i] = final_sorted[i].second;
        }
    }

    if (collect) {
        last_stats.calculations_performed = total_calcs;
        last_stats.columns_used = d;
    }
}

} // namespace faiss
