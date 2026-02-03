/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroAdaptiveScale.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace faiss {

IndexNeuroAdaptiveScale::IndexNeuroAdaptiveScale(int d, int num_scales)
        : IndexNeuro(nullptr, false), num_scales(num_scales) {
    this->d = d;
    words_per_vec = (d + 63) / 64;

    // Initialize default scales
    scales.resize(num_scales);
    for (int i = 0; i < num_scales; i++) {
        scales[i] = std::pow(2.0f, static_cast<float>(i) - 1.0f);
        // Produces: 0.5, 1.0, 2.0 for num_scales=3
    }

    thresholds.resize(num_scales);
    signatures.resize(num_scales);
    is_trained = false;
}

float IndexNeuroAdaptiveScale::compute_entropy(float p) const {
    if (p <= 0.0f || p >= 1.0f) {
        return 0.0f;
    }
    return -p * std::log2(p) - (1.0f - p) * std::log2(1.0f - p);
}

float IndexNeuroAdaptiveScale::find_optimal_threshold(
        const std::vector<float>& values,
        float scale) const {
    if (values.empty()) {
        return 0.0f;
    }

    // Transform values through tanh(x * scale)
    std::vector<float> transformed(values.size());
    for (size_t i = 0; i < values.size(); i++) {
        transformed[i] = std::tanh(values[i] * scale);
    }

    // Sort transformed values
    std::vector<float> sorted = transformed;
    std::sort(sorted.begin(), sorted.end());

    // Try different thresholds and find one maximizing entropy
    float best_threshold = 0.0f;
    float best_entropy = 0.0f;

    // Try percentile-based thresholds
    int n = sorted.size();
    for (int pct = 10; pct <= 90; pct += 5) {
        int idx = (n * pct) / 100;
        float threshold = sorted[idx];

        // Count how many values are above threshold
        int count_above = 0;
        for (float v : transformed) {
            if (v > threshold) {
                count_above++;
            }
        }

        float p = static_cast<float>(count_above) / n;
        float entropy = compute_entropy(p);

        if (entropy > best_entropy) {
            best_entropy = entropy;
            best_threshold = threshold;
        }
    }

    return best_threshold;
}

void IndexNeuroAdaptiveScale::compute_signature(
        const float* vec,
        size_t scale_idx,
        uint64_t* sig) const {
    // Initialize signature to zero
    for (int w = 0; w < words_per_vec; w++) {
        sig[w] = 0;
    }

    float scale = scales[scale_idx];
    const std::vector<float>& thresh = thresholds[scale_idx];

    // Set bits based on adaptive threshold comparison
    for (int j = 0; j < d; j++) {
        float val = std::tanh(vec[j] * scale);
        if (val > thresh[j]) {
            int word_idx = j / 64;
            int bit_idx = j % 64;
            sig[word_idx] |= (1ULL << bit_idx);
        }
    }
}

int IndexNeuroAdaptiveScale::hamming_distance(
        const uint64_t* sig1,
        const uint64_t* sig2) const {
    int dist = 0;
    for (int w = 0; w < words_per_vec; w++) {
        dist += __builtin_popcountll(sig1[w] ^ sig2[w]);
    }
    return dist;
}

void IndexNeuroAdaptiveScale::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(n > 0);

    // Create inner index if not provided
    if (!inner_index) {
        inner_index = new IndexFlatL2(d);
        own_inner = true;
    }

    // Learn adaptive thresholds per dimension per scale
    for (size_t s = 0; s < scales.size(); s++) {
        thresholds[s].resize(d);
        std::vector<float> col_values(n);

        for (int j = 0; j < d; j++) {
            // Collect values for this dimension
            for (idx_t i = 0; i < n; i++) {
                col_values[i] = x[i * d + j];
            }

            // Find optimal threshold
            thresholds[s][j] = find_optimal_threshold(col_values, scales[s]);
        }
    }

    // Clear existing signatures
    for (auto& sigs : signatures) {
        sigs.clear();
    }

    // Compute signatures for all training vectors
    std::vector<uint64_t> sig(words_per_vec);
    for (idx_t i = 0; i < n; i++) {
        const float* vec = x + i * d;
        for (size_t s = 0; s < scales.size(); s++) {
            compute_signature(vec, s, sig.data());
            signatures[s].insert(signatures[s].end(), sig.begin(), sig.end());
        }
    }

    // Add to inner index
    inner_index->add(n, x);
    ntotal = n;
    is_trained = true;
}

void IndexNeuroAdaptiveScale::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "Index must be trained first");

    // Compute signatures for new vectors
    std::vector<uint64_t> sig(words_per_vec);
    for (idx_t i = 0; i < n; i++) {
        const float* vec = x + i * d;
        for (size_t s = 0; s < scales.size(); s++) {
            compute_signature(vec, s, sig.data());
            signatures[s].insert(signatures[s].end(), sig.begin(), sig.end());
        }
    }

    // Add to inner index
    if (inner_index) {
        inner_index->add(n, x);
    }
    ntotal += n;
}

void IndexNeuroAdaptiveScale::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(ntotal > 0);
    FAISS_THROW_IF_NOT(inner_index);

    auto* flat = dynamic_cast<IndexFlat*>(inner_index);
    FAISS_THROW_IF_NOT_MSG(flat, "inner_index must be IndexFlat");

    const float* xb = flat->get_xb();

    // Parse parameters
    int max_hamming = max_hamming_distance;

    auto* asp = dynamic_cast<const NeuroAdaptiveScaleParams*>(params);
    if (asp) {
        // Could add more parameter handling here
    }

    // Auto-compute max Hamming distance if not set
    if (max_hamming <= 0) {
        max_hamming = d / 4;  // ~25% of bits can differ for better recall
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

        // Compute query signatures for all scales
        std::vector<std::vector<uint64_t>> query_sigs(scales.size());
        for (size_t s = 0; s < scales.size(); s++) {
            query_sigs[s].resize(words_per_vec);
            compute_signature(query, s, query_sigs[s].data());
        }

        // Filter candidates by Hamming distance (intersection across scales)
        std::vector<idx_t> candidates;
        for (idx_t i = 0; i < ntotal; i++) {
            bool passes_all = true;

            for (size_t s = 0; s < scales.size(); s++) {
                const uint64_t* vec_sig =
                        signatures[s].data() + i * words_per_vec;
                int hdist = hamming_distance(query_sigs[s].data(), vec_sig);

                if (hdist > max_hamming) {
                    passes_all = false;
                    break;
                }
            }

            if (passes_all) {
                candidates.push_back(i);
            }
        }

        // Fallback if no candidates
        if (candidates.empty()) {
            candidates.resize(ntotal);
            for (idx_t i = 0; i < ntotal; i++) {
                candidates[i] = i;
            }
        }

        // Compute L2 distances for candidates
        std::vector<std::pair<float, idx_t>> dists;
        dists.reserve(candidates.size());

        for (idx_t cand : candidates) {
            float dist = fvec_L2sqr(query, xb + cand * d, d);
            dists.push_back({dist, cand});
        }
        total_calcs += candidates.size() * d;

        // Sort and output top-k
        size_t actual_k = std::min(static_cast<size_t>(k), dists.size());
        std::partial_sort(
                dists.begin(), dists.begin() + actual_k, dists.end());

        for (size_t i = 0; i < actual_k; i++) {
            distances[q * k + i] = dists[i].first;
            labels[q * k + i] = dists[i].second;
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

void IndexNeuroAdaptiveScale::reset() {
    for (auto& sigs : signatures) {
        sigs.clear();
    }
    if (inner_index) {
        inner_index->reset();
    }
    ntotal = 0;
}

} // namespace faiss
