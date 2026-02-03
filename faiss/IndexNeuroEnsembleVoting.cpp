/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroEnsembleVoting.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

namespace faiss {

IndexNeuroEnsembleVoting::IndexNeuroEnsembleVoting(int d)
        : IndexNeuro(nullptr, false) {
    this->d = d;
    hamming_words_per_vec = (d + 63) / 64;
    hamming_thresholds.resize(d, 0.0f);
    is_trained = false;
}

void IndexNeuroEnsembleVoting::encode_hamming(
        const float* vec,
        uint64_t* code) const {
    for (int w = 0; w < hamming_words_per_vec; w++) {
        code[w] = 0;
    }
    for (int j = 0; j < d; j++) {
        if (vec[j] > hamming_thresholds[j]) {
            int word_idx = j / 64;
            int bit_idx = j % 64;
            code[word_idx] |= (1ULL << bit_idx);
        }
    }
}

int IndexNeuroEnsembleVoting::hamming_distance(
        const uint64_t* c1,
        const uint64_t* c2) const {
    int dist = 0;
    for (int w = 0; w < hamming_words_per_vec; w++) {
        dist += __builtin_popcountll(c1[w] ^ c2[w]);
    }
    return dist;
}

void IndexNeuroEnsembleVoting::compute_stats(
        const float* vec,
        float& norm,
        float& mean) const {
    float sum = 0.0f, sum_sq = 0.0f;
    for (int j = 0; j < d; j++) {
        sum += vec[j];
        sum_sq += vec[j] * vec[j];
    }
    norm = std::sqrt(sum_sq);
    mean = sum / d;
}

void IndexNeuroEnsembleVoting::project(const float* vec, float* out) const {
    for (int i = 0; i < projection_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            sum += projection_matrix[j * projection_dim + i] * vec[j];
        }
        out[i] = sum;
    }
}

void IndexNeuroEnsembleVoting::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(n > 0);

    // Store vectors
    vectors.resize(n * d);
    std::copy(x, x + n * d, vectors.data());

    // Train Hamming filter - compute medians
    std::vector<float> col_values(n);
    for (int j = 0; j < d; j++) {
        for (idx_t i = 0; i < n; i++) {
            col_values[i] = x[i * d + j];
        }
        std::nth_element(
                col_values.begin(),
                col_values.begin() + n / 2,
                col_values.end());
        hamming_thresholds[j] = col_values[n / 2];
    }

    // Encode Hamming codes
    hamming_codes.resize(n * hamming_words_per_vec);
    for (idx_t i = 0; i < n; i++) {
        encode_hamming(x + i * d, hamming_codes.data() + i * hamming_words_per_vec);
    }

    // Compute statistics
    stat_norms.resize(n);
    stat_means.resize(n);
    for (idx_t i = 0; i < n; i++) {
        compute_stats(x + i * d, stat_norms[i], stat_means[i]);
    }

    // Initialize random projection
    std::mt19937 rng(random_seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    projection_matrix.resize(d * projection_dim);
    for (int i = 0; i < d * projection_dim; i++) {
        projection_matrix[i] = normal(rng) / std::sqrt(static_cast<float>(projection_dim));
    }

    // Project all data
    projected_data.resize(n * projection_dim);
    for (idx_t i = 0; i < n; i++) {
        project(x + i * d, projected_data.data() + i * projection_dim);
    }

    ntotal = n;
    is_trained = true;
}

void IndexNeuroEnsembleVoting::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "Index must be trained first");

    idx_t old_ntotal = ntotal;

    // Extend vectors
    vectors.resize((old_ntotal + n) * d);
    std::copy(x, x + n * d, vectors.data() + old_ntotal * d);

    // Extend Hamming codes
    hamming_codes.resize((old_ntotal + n) * hamming_words_per_vec);
    for (idx_t i = 0; i < n; i++) {
        encode_hamming(
                x + i * d,
                hamming_codes.data() + (old_ntotal + i) * hamming_words_per_vec);
    }

    // Extend stats
    stat_norms.resize(old_ntotal + n);
    stat_means.resize(old_ntotal + n);
    for (idx_t i = 0; i < n; i++) {
        compute_stats(x + i * d, stat_norms[old_ntotal + i], stat_means[old_ntotal + i]);
    }

    // Extend projections
    projected_data.resize((old_ntotal + n) * projection_dim);
    for (idx_t i = 0; i < n; i++) {
        project(x + i * d, projected_data.data() + (old_ntotal + i) * projection_dim);
    }

    ntotal += n;
}

void IndexNeuroEnsembleVoting::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(ntotal > 0);

    // Parse parameters
    int mv = min_votes;
    bool uh = use_hamming;
    bool us = use_stats;
    bool up = use_projection;

    auto* evp = dynamic_cast<const NeuroEnsembleVotingParams*>(params);
    if (evp) {
        mv = evp->min_votes;
        uh = evp->use_hamming;
        us = evp->use_stats;
        up = evp->use_projection;
    }

    bool collect = false;
    if (params) {
        auto* nsp = dynamic_cast<const NeuroSearchParameters*>(params);
        if (nsp) {
            collect = nsp->collect_stats;
        }
    }

    idx_t filter_keep_n = std::max(
            static_cast<idx_t>(ntotal * filter_keep_ratio),
            std::min(k * 4, ntotal));

    int64_t total_calcs = 0;

#pragma omp parallel for reduction(+ : total_calcs)
    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * d;

        // Vote counting
        std::vector<int> votes(ntotal, 0);

        // Hamming filter
        if (uh) {
            std::vector<uint64_t> query_code(hamming_words_per_vec);
            encode_hamming(query, query_code.data());

            std::vector<std::pair<int, idx_t>> hdists(ntotal);
            for (idx_t i = 0; i < ntotal; i++) {
                const uint64_t* vec_code =
                        hamming_codes.data() + i * hamming_words_per_vec;
                hdists[i] = {hamming_distance(query_code.data(), vec_code), i};
            }

            std::partial_sort(
                    hdists.begin(),
                    hdists.begin() + filter_keep_n,
                    hdists.end());

            for (idx_t i = 0; i < filter_keep_n; i++) {
                votes[hdists[i].second]++;
            }
        }

        // Statistical filter
        if (us) {
            float query_norm, query_mean;
            compute_stats(query, query_norm, query_mean);
            float sqrt_d = std::sqrt(static_cast<float>(d));

            std::vector<std::pair<float, idx_t>> scores(ntotal);
            for (idx_t i = 0; i < ntotal; i++) {
                float score = std::abs(query_norm - stat_norms[i]) +
                              std::abs(query_mean - stat_means[i]) * sqrt_d;
                scores[i] = {score, i};
            }

            std::partial_sort(
                    scores.begin(),
                    scores.begin() + filter_keep_n,
                    scores.end());

            for (idx_t i = 0; i < filter_keep_n; i++) {
                votes[scores[i].second]++;
            }
        }

        // Projection filter
        if (up) {
            std::vector<float> query_proj(projection_dim);
            project(query, query_proj.data());

            std::vector<std::pair<float, idx_t>> proj_dists(ntotal);
            for (idx_t i = 0; i < ntotal; i++) {
                const float* vec_proj =
                        projected_data.data() + i * projection_dim;
                float dist = fvec_L2sqr(query_proj.data(), vec_proj, projection_dim);
                proj_dists[i] = {dist, i};
            }

            std::partial_sort(
                    proj_dists.begin(),
                    proj_dists.begin() + filter_keep_n,
                    proj_dists.end());

            for (idx_t i = 0; i < filter_keep_n; i++) {
                votes[proj_dists[i].second]++;
            }
        }

        // Collect candidates with sufficient votes
        std::vector<idx_t> candidates;
        for (idx_t i = 0; i < ntotal; i++) {
            if (votes[i] >= mv) {
                candidates.push_back(i);
            }
        }

        // Fallback if no candidates
        if (candidates.empty()) {
            // Take top by vote count
            std::vector<std::pair<int, idx_t>> vote_pairs(ntotal);
            for (idx_t i = 0; i < ntotal; i++) {
                vote_pairs[i] = {-votes[i], i};  // negative for descending sort
            }
            std::partial_sort(
                    vote_pairs.begin(),
                    vote_pairs.begin() + std::min(filter_keep_n, ntotal),
                    vote_pairs.end());

            for (idx_t i = 0; i < std::min(filter_keep_n, ntotal); i++) {
                candidates.push_back(vote_pairs[i].second);
            }
        }

        // Compute L2 distances
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

void IndexNeuroEnsembleVoting::reset() {
    vectors.clear();
    hamming_codes.clear();
    stat_norms.clear();
    stat_means.clear();
    projected_data.clear();
    ntotal = 0;
}

} // namespace faiss
