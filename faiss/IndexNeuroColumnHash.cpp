/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroColumnHash.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <random>

namespace faiss {

namespace {

/// Count set bits in a uint64
inline int popcount64(uint64_t x) {
#ifdef __GNUC__
    return __builtin_popcountll(x);
#else
    int count = 0;
    while (x) {
        count += x & 1;
        x >>= 1;
    }
    return count;
#endif
}

/// Get raw data pointer from IndexFlat
const float* get_flat_data_col(const Index* index) {
    auto flat = dynamic_cast<const IndexFlat*>(index);
    FAISS_THROW_IF_NOT_MSG(
            flat, "IndexNeuroColumnHash requires inner_index to be IndexFlat");
    return flat->get_xb();
}

} // anonymous namespace

IndexNeuroColumnHash::IndexNeuroColumnHash(Index* inner, bool own_inner)
        : IndexNeuro(inner, own_inner) {
    // Default: split dimensions into 4 equal groups
    set_equal_groups(4);
}

void IndexNeuroColumnHash::set_groups(
        const std::vector<std::vector<int>>& new_groups) {
    groups = new_groups;
    // Validate indices
    for (const auto& g : groups) {
        for (int idx : g) {
            FAISS_THROW_IF_NOT_FMT(
                    idx >= 0 && idx < d,
                    "group index %d out of range [0, %d)",
                    idx,
                    d);
        }
    }
}

void IndexNeuroColumnHash::set_equal_groups(int n_groups) {
    FAISS_THROW_IF_NOT_MSG(n_groups > 0, "n_groups must be positive");
    FAISS_THROW_IF_NOT_MSG(d > 0, "dimension must be set");

    groups.resize(n_groups);
    for (int g = 0; g < n_groups; g++) {
        groups[g].clear();
    }

    // Round-robin assignment
    for (int i = 0; i < d; i++) {
        groups[i % n_groups].push_back(i);
    }
}

void IndexNeuroColumnHash::train(idx_t /*n*/, const float* /*x*/) {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");
    FAISS_THROW_IF_NOT_MSG(!groups.empty(), "groups must be set");

    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    int n_groups = static_cast<int>(groups.size());
    group_hyperplanes.resize(n_groups);
    group_codes.resize(n_groups);

    for (int g = 0; g < n_groups; g++) {
        int group_d = static_cast<int>(groups[g].size());
        if (group_d == 0) continue;

        group_hyperplanes[g].resize(bits_per_group * group_d);

        for (int b = 0; b < bits_per_group; b++) {
            float* hp = group_hyperplanes[g].data() + b * group_d;
            float norm = 0.0f;
            for (int i = 0; i < group_d; i++) {
                hp[i] = normal(rng);
                norm += hp[i] * hp[i];
            }
            norm = std::sqrt(norm);
            if (norm > 1e-10f) {
                for (int i = 0; i < group_d; i++) {
                    hp[i] /= norm;
                }
            }
        }
    }

    // Initialize uniform weights if not set
    if (group_weights.empty()) {
        group_weights.assign(n_groups, 1.0f);
    }

    is_trained = true;
}

void IndexNeuroColumnHash::compute_group_code(
        int group,
        const float* x,
        uint64_t& code) const {
    code = 0;
    const auto& group_cols = groups[group];
    int group_d = static_cast<int>(group_cols.size());
    if (group_d == 0) return;

    const float* hps = group_hyperplanes[group].data();

    for (int b = 0; b < bits_per_group && b < 64; b++) {
        const float* hp = hps + b * group_d;
        float dot = 0.0f;
        for (int i = 0; i < group_d; i++) {
            dot += hp[i] * x[group_cols[i]];
        }
        if (dot > 0) {
            code |= (1ULL << b);
        }
    }
}

void IndexNeuroColumnHash::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained before adding");

    IndexNeuro::add(n, x);

    int n_groups = static_cast<int>(groups.size());
    idx_t old_ntotal = group_codes.empty() || group_codes[0].empty()
                               ? 0
                               : group_codes[0].size();

    for (int g = 0; g < n_groups; g++) {
        group_codes[g].resize(old_ntotal + n);
        for (idx_t i = 0; i < n; i++) {
            compute_group_code(g, x + i * d, group_codes[g][old_ntotal + i]);
        }
    }
}

void IndexNeuroColumnHash::reset() {
    IndexNeuro::reset();
    for (auto& codes : group_codes) {
        codes.clear();
    }
}

void IndexNeuroColumnHash::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained");

    const float* data = get_flat_data_col(inner_index);
    idx_t nb = inner_index->ntotal;
    int n_groups = static_cast<int>(groups.size());

    // Resolve parameters
    int min_match = min_groups_match;
    bool do_rerank = rerank;

    auto cp = dynamic_cast<const NeuroColumnHashParams*>(params);
    if (cp) {
        if (cp->min_groups_match >= 0) {
            min_match = cp->min_groups_match;
        }
        do_rerank = cp->rerank;
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

        // Compute query codes for all groups
        std::vector<uint64_t> query_codes(n_groups);
        for (int g = 0; g < n_groups; g++) {
            compute_group_code(g, query, query_codes[g]);
        }

        // Find candidates that match on >= min_match groups
        std::vector<std::pair<float, idx_t>> candidates;
        candidates.reserve(nb);

        for (idx_t i = 0; i < nb; i++) {
            int matches = 0;
            float weighted_score = 0.0f;

            for (int g = 0; g < n_groups; g++) {
                int ham = popcount64(query_codes[g] ^ group_codes[g][i]);
                if (ham <= group_hamming_threshold) {
                    matches++;
                    weighted_score += group_weights[g];
                }
            }

            if (matches >= min_match) {
                // Use negative weighted score (more matches = lower score)
                candidates.emplace_back(-weighted_score, i);
            }
        }

        // Sort by score (best matches first)
        std::sort(candidates.begin(), candidates.end());

        // Limit candidates and optionally rerank
        size_t n_cand = std::min(candidates.size(), static_cast<size_t>(k * 3));
        std::vector<std::pair<float, idx_t>> scored(n_cand);

        for (size_t i = 0; i < n_cand; i++) {
            idx_t idx = candidates[i].second;
            float dist;
            if (do_rerank) {
                if (metric) {
                    dist = metric->distance(query, data + idx * d, d);
                } else {
                    dist = fvec_L2sqr(query, data + idx * d, d);
                }
                total_calcs += d;
            } else {
                dist = candidates[i].first;  // Use hash-based score
            }
            scored[i] = {dist, idx};
        }

        // Final sort and output
        size_t actual_k = std::min(static_cast<size_t>(k), n_cand);
        if (actual_k > 0) {
            std::partial_sort(
                    scored.begin(),
                    scored.begin() + actual_k,
                    scored.end());
        }

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
