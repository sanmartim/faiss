/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroRecommendedPipeline.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace faiss {

IndexNeuroRecommendedPipeline::IndexNeuroRecommendedPipeline(int d)
        : IndexNeuro(nullptr, false) {
    this->d = d;
    ms_words_per_vec = (d + 63) / 64;
    fs_words_per_vec = (d + 63) / 64;
    fs_thresholds.resize(d, 0.0f);
    ms_signatures.resize(scales.size());
    is_trained = false;
}

void IndexNeuroRecommendedPipeline::compute_ms_signature(
        const float* vec,
        float scale,
        uint64_t* sig) const {
    for (int w = 0; w < ms_words_per_vec; w++) {
        sig[w] = 0;
    }

    for (int j = 0; j < d; j++) {
        float val = std::tanh(vec[j] * scale);
        if (val >= 0.0f) {
            int word_idx = j / 64;
            int bit_idx = j % 64;
            sig[word_idx] |= (1ULL << bit_idx);
        }
    }
}

void IndexNeuroRecommendedPipeline::compute_fs_code(
        const float* vec,
        uint64_t* code) const {
    for (int w = 0; w < fs_words_per_vec; w++) {
        code[w] = 0;
    }

    for (int j = 0; j < d; j++) {
        if (vec[j] > fs_thresholds[j]) {
            int word_idx = j / 64;
            int bit_idx = j % 64;
            code[word_idx] |= (1ULL << bit_idx);
        }
    }
}

int IndexNeuroRecommendedPipeline::hamming_distance(
        const uint64_t* sig1,
        const uint64_t* sig2,
        int words) const {
    int dist = 0;
    for (int w = 0; w < words; w++) {
        dist += __builtin_popcountll(sig1[w] ^ sig2[w]);
    }
    return dist;
}

void IndexNeuroRecommendedPipeline::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(n > 0);

    // Store vectors
    vectors.resize(n * d);
    std::copy(x, x + n * d, vectors.data());

    // Compute FS thresholds (medians)
    std::vector<float> col_values(n);
    for (int j = 0; j < d; j++) {
        for (idx_t i = 0; i < n; i++) {
            col_values[i] = x[i * d + j];
        }
        std::nth_element(
                col_values.begin(),
                col_values.begin() + n / 2,
                col_values.end());
        fs_thresholds[j] = col_values[n / 2];
    }

    // Clear existing data
    for (auto& sigs : ms_signatures) {
        sigs.clear();
    }
    fs_codes.clear();

    // Compute MS signatures
    std::vector<uint64_t> sig(ms_words_per_vec);
    for (idx_t i = 0; i < n; i++) {
        const float* vec = x + i * d;
        for (size_t s = 0; s < scales.size(); s++) {
            compute_ms_signature(vec, scales[s], sig.data());
            ms_signatures[s].insert(
                    ms_signatures[s].end(), sig.begin(), sig.end());
        }
    }

    // Compute FS codes
    std::vector<uint64_t> code(fs_words_per_vec);
    for (idx_t i = 0; i < n; i++) {
        compute_fs_code(x + i * d, code.data());
        fs_codes.insert(fs_codes.end(), code.begin(), code.end());
    }

    ntotal = n;
    is_trained = true;
}

void IndexNeuroRecommendedPipeline::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "Index must be trained first");

    idx_t old_ntotal = ntotal;

    // Extend vectors
    vectors.resize((old_ntotal + n) * d);
    std::copy(x, x + n * d, vectors.data() + old_ntotal * d);

    // Compute MS signatures for new vectors
    std::vector<uint64_t> sig(ms_words_per_vec);
    for (idx_t i = 0; i < n; i++) {
        const float* vec = x + i * d;
        for (size_t s = 0; s < scales.size(); s++) {
            compute_ms_signature(vec, scales[s], sig.data());
            ms_signatures[s].insert(
                    ms_signatures[s].end(), sig.begin(), sig.end());
        }
    }

    // Compute FS codes for new vectors
    std::vector<uint64_t> code(fs_words_per_vec);
    for (idx_t i = 0; i < n; i++) {
        compute_fs_code(x + i * d, code.data());
        fs_codes.insert(fs_codes.end(), code.begin(), code.end());
    }

    ntotal += n;
}

void IndexNeuroRecommendedPipeline::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(ntotal > 0);

    // Parse parameters
    float ms_kr = ms_keep_ratio;
    float fs_kr = fs_keep_ratio;

    auto* rpp = dynamic_cast<const NeuroRecommendedPipelineParams*>(params);
    if (rpp) {
        ms_kr = rpp->ms_keep_ratio;
        fs_kr = rpp->fs_keep_ratio;
    }

    bool collect = false;
    if (params) {
        auto* nsp = dynamic_cast<const NeuroSearchParameters*>(params);
        if (nsp) {
            collect = nsp->collect_stats;
        }
    }

    // Auto-compute max Hamming distance - more permissive for better recall
    int ms_max_hamming = d / 4;

    int64_t total_calcs = 0;

#pragma omp parallel for reduction(+ : total_calcs)
    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * d;

        // ========== Stage 1: Multi-Scale Sign Filter ==========
        // Compute query MS signatures
        std::vector<std::vector<uint64_t>> query_ms_sigs(scales.size());
        for (size_t s = 0; s < scales.size(); s++) {
            query_ms_sigs[s].resize(ms_words_per_vec);
            compute_ms_signature(query, scales[s], query_ms_sigs[s].data());
        }

        // Filter by MS (intersection across all scales)
        std::vector<std::pair<int, idx_t>> ms_candidates;
        for (idx_t i = 0; i < ntotal; i++) {
            int total_hamming = 0;
            bool passes = true;

            for (size_t s = 0; s < scales.size(); s++) {
                const uint64_t* vec_sig =
                        ms_signatures[s].data() + i * ms_words_per_vec;
                int hdist = hamming_distance(
                        query_ms_sigs[s].data(), vec_sig, ms_words_per_vec);
                total_hamming += hdist;

                if (hdist > ms_max_hamming) {
                    passes = false;
                    break;
                }
            }

            if (passes) {
                ms_candidates.push_back({total_hamming, i});
            }
        }

        // Keep top ms_keep_ratio by total Hamming
        idx_t ms_keep_n = std::max(
                static_cast<idx_t>(ntotal * ms_kr),
                std::min(k * 4, ntotal));

        if (ms_candidates.size() > static_cast<size_t>(ms_keep_n)) {
            std::partial_sort(
                    ms_candidates.begin(),
                    ms_candidates.begin() + ms_keep_n,
                    ms_candidates.end());
            ms_candidates.resize(ms_keep_n);
        }

        // Fallback if no MS candidates
        if (ms_candidates.empty()) {
            for (idx_t i = 0; i < ntotal; i++) {
                int total_hamming = 0;
                for (size_t s = 0; s < scales.size(); s++) {
                    const uint64_t* vec_sig =
                            ms_signatures[s].data() + i * ms_words_per_vec;
                    total_hamming += hamming_distance(
                            query_ms_sigs[s].data(), vec_sig, ms_words_per_vec);
                }
                ms_candidates.push_back({total_hamming, i});
            }
            std::partial_sort(
                    ms_candidates.begin(),
                    ms_candidates.begin() + std::min(ms_keep_n, ntotal),
                    ms_candidates.end());
            ms_candidates.resize(std::min(ms_keep_n, ntotal));
        }

        // ========== Stage 2: Hamming Prefilter ==========
        // Compute query FS code
        std::vector<uint64_t> query_fs_code(fs_words_per_vec);
        compute_fs_code(query, query_fs_code.data());

        // Filter MS candidates by FS Hamming
        std::vector<std::pair<int, idx_t>> fs_candidates;
        for (const auto& mc : ms_candidates) {
            idx_t i = mc.second;
            const uint64_t* vec_code = fs_codes.data() + i * fs_words_per_vec;
            int hdist = hamming_distance(
                    query_fs_code.data(), vec_code, fs_words_per_vec);
            fs_candidates.push_back({hdist, i});
        }

        // Keep top fs_keep_ratio
        idx_t fs_keep_n = std::max(
                static_cast<idx_t>(ms_candidates.size() * fs_kr / ms_kr),
                std::min(k * 2, static_cast<idx_t>(fs_candidates.size())));

        if (fs_candidates.size() > static_cast<size_t>(fs_keep_n)) {
            std::partial_sort(
                    fs_candidates.begin(),
                    fs_candidates.begin() + fs_keep_n,
                    fs_candidates.end());
            fs_candidates.resize(fs_keep_n);
        }

        // ========== Stage 3: Precise L2 ==========
        std::vector<std::pair<float, idx_t>> l2_dists;
        l2_dists.reserve(fs_candidates.size());

        for (const auto& fc : fs_candidates) {
            idx_t cand = fc.second;
            float dist = fvec_L2sqr(query, vectors.data() + cand * d, d);
            l2_dists.push_back({dist, cand});
        }
        total_calcs += fs_candidates.size() * d;

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

void IndexNeuroRecommendedPipeline::reset() {
    vectors.clear();
    for (auto& sigs : ms_signatures) {
        sigs.clear();
    }
    fs_codes.clear();
    ntotal = 0;
}

} // namespace faiss
