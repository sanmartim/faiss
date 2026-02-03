/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroAnchor.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <random>

namespace faiss {

namespace {

const float* get_flat_data_anchor(const Index* index) {
    auto flat = dynamic_cast<const IndexFlat*>(index);
    FAISS_THROW_IF_NOT_MSG(
            flat, "IndexNeuroAnchor requires inner_index to be IndexFlat");
    return flat->get_xb();
}

} // anonymous namespace

IndexNeuroAnchor::IndexNeuroAnchor(Index* inner, int n_anchors, bool own_inner)
        : IndexNeuro(inner, own_inner), n_anchors(n_anchors) {}

void IndexNeuroAnchor::farthest_point_sampling(idx_t n, const float* x) {
    if (n == 0) return;

    std::vector<float> min_dists(n, std::numeric_limits<float>::max());
    std::vector<bool> selected(n, false);

    // Start with random point
    std::mt19937 rng(42);
    std::uniform_int_distribution<idx_t> dist(0, n - 1);
    idx_t first = dist(rng);
    selected[first] = true;
    std::copy(x + first * d, x + (first + 1) * d, anchors.data());

    // Update distances to first anchor
    for (idx_t i = 0; i < n; i++) {
        if (!selected[i]) {
            min_dists[i] = fvec_L2sqr(x + i * d, anchors.data(), d);
        }
    }

    // Select remaining anchors
    for (int a = 1; a < n_anchors; a++) {
        // Find farthest point
        idx_t farthest = 0;
        float max_dist = -1.0f;
        for (idx_t i = 0; i < n; i++) {
            if (!selected[i] && min_dists[i] > max_dist) {
                max_dist = min_dists[i];
                farthest = i;
            }
        }

        // Add as anchor
        selected[farthest] = true;
        std::copy(x + farthest * d, x + (farthest + 1) * d,
                  anchors.data() + a * d);

        // Update distances
        for (idx_t i = 0; i < n; i++) {
            if (!selected[i]) {
                float dist = fvec_L2sqr(x + i * d, anchors.data() + a * d, d);
                min_dists[i] = std::min(min_dists[i], dist);
            }
        }
    }
}

void IndexNeuroAnchor::kmeans_anchors(idx_t n, const float* x) {
    if (n == 0) return;

    // Simple k-means: initialize with farthest point sampling
    farthest_point_sampling(n, x);

    // K-means iterations
    int max_iter = 20;
    std::vector<int> assignments(n);
    std::vector<int> counts(n_anchors);

    for (int iter = 0; iter < max_iter; iter++) {
        // Assign points to nearest anchor
        std::fill(counts.begin(), counts.end(), 0);
        for (idx_t i = 0; i < n; i++) {
            float min_dist = std::numeric_limits<float>::max();
            int best_a = 0;
            for (int a = 0; a < n_anchors; a++) {
                float dist = fvec_L2sqr(x + i * d, anchors.data() + a * d, d);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_a = a;
                }
            }
            assignments[i] = best_a;
            counts[best_a]++;
        }

        // Update anchor positions
        std::vector<float> new_anchors(n_anchors * d, 0.0f);
        for (idx_t i = 0; i < n; i++) {
            int a = assignments[i];
            for (int j = 0; j < d; j++) {
                new_anchors[a * d + j] += x[i * d + j];
            }
        }

        for (int a = 0; a < n_anchors; a++) {
            if (counts[a] > 0) {
                for (int j = 0; j < d; j++) {
                    anchors[a * d + j] = new_anchors[a * d + j] / counts[a];
                }
            }
        }
    }
}

void IndexNeuroAnchor::optimize_anchors(idx_t n, const float* x) {
    // MT-02: Gradient-based optimization to minimize approximation error
    // Start from k-means initialization
    kmeans_anchors(n, x);

    if (n < 100) return;  // Not enough data for optimization

    std::mt19937 rng(42);
    std::uniform_int_distribution<idx_t> sample_dist(0, n - 1);

    for (int step = 0; step < n_optimization_steps; step++) {
        // Sample a pair of points
        idx_t i = sample_dist(rng);
        idx_t j = sample_dist(rng);
        while (j == i) j = sample_dist(rng);

        const float* xi = x + i * d;
        const float* xj = x + j * d;

        // Compute true distance
        float true_dist = std::sqrt(fvec_L2sqr(xi, xj, d));

        // Compute profile-based distance estimate
        std::vector<float> pi(n_anchors), pj(n_anchors);
        compute_profile(xi, pi.data());
        compute_profile(xj, pj.data());

        float approx_dist = 0.0f;
        for (int a = 0; a < n_anchors; a++) {
            float diff = pi[a] - pj[a];
            approx_dist += diff * diff;
        }
        approx_dist = std::sqrt(approx_dist / n_anchors);

        // Compute gradient to reduce error
        float error = approx_dist - true_dist;

        for (int a = 0; a < n_anchors; a++) {
            float* anchor = anchors.data() + a * d;
            float di = std::sqrt(fvec_L2sqr(xi, anchor, d));
            float dj = std::sqrt(fvec_L2sqr(xj, anchor, d));

            if (di < 1e-10f || dj < 1e-10f) continue;

            // Gradient w.r.t. anchor position
            for (int k = 0; k < d; k++) {
                float grad_i = (anchor[k] - xi[k]) / di;
                float grad_j = (anchor[k] - xj[k]) / dj;
                float grad = error * (grad_i - grad_j) * (pi[a] - pj[a]);

                anchor[k] -= learning_rate * grad;
            }
        }
    }
}

void IndexNeuroAnchor::select_anchors(idx_t n, const float* x) {
    anchors.resize(n_anchors * d);

    switch (selection) {
        case NEURO_ANCHOR_RANDOM: {
            std::mt19937 rng(42);
            std::uniform_int_distribution<idx_t> dist(0, n - 1);
            for (int a = 0; a < n_anchors; a++) {
                idx_t idx = dist(rng);
                std::copy(x + idx * d, x + (idx + 1) * d, anchors.data() + a * d);
            }
            break;
        }
        case NEURO_ANCHOR_FARTHEST:
            farthest_point_sampling(n, x);
            break;
        case NEURO_ANCHOR_KMEANS:
            kmeans_anchors(n, x);
            break;
        case NEURO_ANCHOR_LEARNED:
            optimize_anchors(n, x);
            break;
    }
}

void IndexNeuroAnchor::set_hierarchical(
        const std::vector<int>& anc_per_level,
        const std::vector<int>& cand_per_level) {
    FAISS_THROW_IF_NOT_MSG(
            anc_per_level.size() == cand_per_level.size(),
            "anchors and candidates per level must have same size");
    hierarchical = true;
    anchors_per_level = anc_per_level;
    candidates_per_level = cand_per_level;

    // Total anchors
    n_anchors = 0;
    for (int a : anchors_per_level) {
        n_anchors += a;
    }
}

void IndexNeuroAnchor::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");
    FAISS_THROW_IF_NOT_MSG(d > 0, "dimension must be positive");

    if (hierarchical && !anchors_per_level.empty()) {
        // Train hierarchical anchors
        int total_anchors = 0;
        for (int a : anchors_per_level) {
            total_anchors += a;
        }
        anchors.resize(total_anchors * d);

        int offset = 0;
        for (size_t level = 0; level < anchors_per_level.size(); level++) {
            int n_level = anchors_per_level[level];

            // For each level, select anchors
            int old_n_anchors = n_anchors;
            n_anchors = n_level;
            std::vector<float> level_anchors(n_level * d);

            // Temporarily use class members for selection
            std::vector<float> temp_anchors;
            std::swap(anchors, temp_anchors);
            anchors.resize(n_level * d);

            select_anchors(n, x);

            std::copy(anchors.begin(), anchors.end(),
                      temp_anchors.begin() + offset * d);
            offset += n_level;

            std::swap(anchors, temp_anchors);
            n_anchors = old_n_anchors;
        }

        level_profiles.resize(anchors_per_level.size());
    } else {
        select_anchors(n, x);
    }

    profiles.clear();
    is_trained = true;
}

void IndexNeuroAnchor::compute_profile(const float* x, float* profile) const {
    for (int a = 0; a < n_anchors; a++) {
        const float* anchor = anchors.data() + a * d;
        profile[a] = std::sqrt(fvec_L2sqr(x, anchor, d));
    }
}

void IndexNeuroAnchor::compute_hierarchical_profile(
        const float* x,
        std::vector<std::vector<float>>& profs) const {
    int n_levels = static_cast<int>(anchors_per_level.size());
    profs.resize(n_levels);

    int offset = 0;
    for (int level = 0; level < n_levels; level++) {
        int n_level = anchors_per_level[level];
        profs[level].resize(n_level);

        for (int a = 0; a < n_level; a++) {
            const float* anchor = anchors.data() + (offset + a) * d;
            profs[level][a] = std::sqrt(fvec_L2sqr(x, anchor, d));
        }
        offset += n_level;
    }
}

float IndexNeuroAnchor::profile_distance(
        const float* p1,
        const float* p2,
        int n) const {
    float dist = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = p1[i] - p2[i];
        dist += diff * diff;
    }
    return dist;
}

void IndexNeuroAnchor::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained before adding");

    IndexNeuro::add(n, x);

    // Compute profiles for new vectors
    size_t old_size = profiles.size();
    profiles.resize(old_size + n * n_anchors);

    for (idx_t i = 0; i < n; i++) {
        compute_profile(x + i * d, profiles.data() + old_size + i * n_anchors);
    }
}

void IndexNeuroAnchor::reset() {
    IndexNeuro::reset();
    profiles.clear();
    for (auto& lp : level_profiles) {
        lp.clear();
    }
}

void IndexNeuroAnchor::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained");

    const float* data = get_flat_data_anchor(inner_index);
    idx_t nb = inner_index->ntotal;

    // Resolve parameters
    bool do_rerank = rerank;
    int n_cand = n_candidates;

    auto ap = dynamic_cast<const NeuroAnchorParams*>(params);
    if (ap) {
        do_rerank = ap->rerank;
        if (ap->candidates_per_anchor > 0) {
            n_cand = ap->candidates_per_anchor * n_anchors;
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

        // Compute query profile
        std::vector<float> query_profile(n_anchors);
        compute_profile(query, query_profile.data());

        // Score all database vectors by profile distance
        std::vector<std::pair<float, idx_t>> scored(nb);
        for (idx_t i = 0; i < nb; i++) {
            const float* data_profile = profiles.data() + i * n_anchors;
            float pdist = profile_distance(
                    query_profile.data(), data_profile, n_anchors);
            scored[i] = {pdist, i};
        }

        // Get top candidates by profile distance
        size_t actual_cand = std::min(static_cast<size_t>(n_cand),
                                       static_cast<size_t>(nb));
        std::partial_sort(
                scored.begin(),
                scored.begin() + actual_cand,
                scored.end());

        // Rerank if requested
        if (do_rerank) {
            for (size_t i = 0; i < actual_cand; i++) {
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
                    scored.begin() + std::min(actual_cand, static_cast<size_t>(k)),
                    scored.end());
        }

        // Output results
        size_t actual_k = std::min(static_cast<size_t>(k), actual_cand);
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
