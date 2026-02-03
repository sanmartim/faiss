/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroProductQuantizationTiered.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace faiss {

IndexNeuroProductQuantizationTiered::IndexNeuroProductQuantizationTiered(
        int d,
        int M,
        int rerank_k)
        : IndexNeuro(nullptr, false), rerank_k(rerank_k) {
    this->d = d;

    // Default 2-tier configuration: 4-bit coarse, 8-bit fine
    tier_configs.push_back({M, 4, 0.1f});   // Tier 0: 4-bit, keep 10%
    tier_configs.push_back({M, 8, 0.5f});   // Tier 1: 8-bit, keep 50%

    // Create PQs
    for (const auto& config : tier_configs) {
        pqs.push_back(new ProductQuantizer(d, config.M, config.nbits));
    }
    codes_per_tier.resize(pqs.size());

    is_trained = false;
}

IndexNeuroProductQuantizationTiered::IndexNeuroProductQuantizationTiered(
        int d,
        const std::vector<NeuroPQTierConfig>& configs,
        int rerank_k)
        : IndexNeuro(nullptr, false), tier_configs(configs), rerank_k(rerank_k) {
    this->d = d;

    for (const auto& config : tier_configs) {
        pqs.push_back(new ProductQuantizer(d, config.M, config.nbits));
    }
    codes_per_tier.resize(pqs.size());

    is_trained = false;
}

IndexNeuroProductQuantizationTiered::~IndexNeuroProductQuantizationTiered() {
    for (auto* pq : pqs) {
        delete pq;
    }
}

void IndexNeuroProductQuantizationTiered::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(!pqs.empty(), "no PQ tiers configured");
    FAISS_THROW_IF_NOT_MSG(d > 0, "dimension must be positive");
    FAISS_THROW_IF_NOT_MSG(n > 0, "need training data");

    // Train each tier's PQ
    for (auto* pq : pqs) {
        if (pq->centroids.empty()) {
            pq->train(n, x);
        }
    }

    is_trained = true;
}

void IndexNeuroProductQuantizationTiered::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained before adding");

    // Encode at each tier
    for (size_t t = 0; t < pqs.size(); t++) {
        size_t old_size = codes_per_tier[t].size();
        codes_per_tier[t].resize(old_size + n * pqs[t]->code_size);
        pqs[t]->compute_codes(x, codes_per_tier[t].data() + old_size, n);
    }

    // Store original vectors for reranking
    size_t old_orig = orig_vectors.size();
    orig_vectors.resize(old_orig + n * d);
    std::copy(x, x + n * d, orig_vectors.data() + old_orig);

    ntotal += n;
}

void IndexNeuroProductQuantizationTiered::reset() {
    for (auto& codes : codes_per_tier) {
        codes.clear();
    }
    orig_vectors.clear();
    ntotal = 0;
}

void IndexNeuroProductQuantizationTiered::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT(key >= 0 && key < ntotal);
    std::copy(
            orig_vectors.data() + key * d,
            orig_vectors.data() + (key + 1) * d,
            recons);
}

float IndexNeuroProductQuantizationTiered::get_compression_ratio() const {
    if (pqs.empty()) return 1.0f;
    // Use the coarsest (first) tier for compression ratio
    return static_cast<float>(d * sizeof(float)) / pqs[0]->code_size;
}

void IndexNeuroProductQuantizationTiered::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(!pqs.empty(), "no PQ tiers configured");
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained");

    // Resolve parameters
    int n_rerank = rerank_k;
    bool should_rerank = do_rerank;

    auto tp = dynamic_cast<const NeuroPQTieredParams*>(params);
    if (tp) {
        if (tp->rerank_k > 0) n_rerank = tp->rerank_k;
        should_rerank = tp->do_rerank;
    }

#pragma omp parallel for
    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * d;

        // Start with all candidates
        std::vector<idx_t> candidates(ntotal);
        for (idx_t i = 0; i < ntotal; i++) {
            candidates[i] = i;
        }

        // Cascade through tiers
        for (size_t t = 0; t < pqs.size() && candidates.size() > static_cast<size_t>(n_rerank); t++) {
            ProductQuantizer* pq = pqs[t];
            const auto& codes = codes_per_tier[t];

            // Compute distance table for this tier
            std::vector<float> dis_table(pq->M * pq->ksub);
            pq->compute_distance_table(query, dis_table.data());

            // Score candidates
            std::vector<std::pair<float, idx_t>> scored;
            scored.reserve(candidates.size());
            for (idx_t idx : candidates) {
                const uint8_t* code = codes.data() + idx * pq->code_size;
                float dist = 0.0f;
                for (size_t m = 0; m < pq->M; m++) {
                    dist += dis_table[m * pq->ksub + code[m]];
                }
                scored.push_back({dist, idx});
            }

            // Keep top fraction
            size_t keep = std::max(
                    static_cast<size_t>(n_rerank),
                    static_cast<size_t>(candidates.size() * tier_configs[t].keep_ratio));
            keep = std::min(keep, scored.size());

            std::partial_sort(scored.begin(), scored.begin() + keep, scored.end());

            candidates.clear();
            candidates.reserve(keep);
            for (size_t i = 0; i < keep; i++) {
                candidates.push_back(scored[i].second);
            }
        }

        // Final reranking with original vectors
        std::vector<std::pair<float, idx_t>> final_scored;
        final_scored.reserve(candidates.size());

        if (should_rerank && !orig_vectors.empty()) {
            for (idx_t idx : candidates) {
                const float* orig = orig_vectors.data() + idx * d;
                float dist;
                if (metric) {
                    dist = metric->distance(query, orig, d);
                } else {
                    dist = fvec_L2sqr(query, orig, d);
                }
                final_scored.push_back({dist, idx});
            }
        } else {
            // Use last tier's distances
            ProductQuantizer* pq = pqs.back();
            const auto& codes = codes_per_tier.back();
            std::vector<float> dis_table(pq->M * pq->ksub);
            pq->compute_distance_table(query, dis_table.data());

            for (idx_t idx : candidates) {
                const uint8_t* code = codes.data() + idx * pq->code_size;
                float dist = 0.0f;
                for (size_t m = 0; m < pq->M; m++) {
                    dist += dis_table[m * pq->ksub + code[m]];
                }
                final_scored.push_back({dist, idx});
            }
        }

        // Sort and output
        size_t actual_k = std::min(static_cast<size_t>(k), final_scored.size());
        std::partial_sort(
                final_scored.begin(),
                final_scored.begin() + actual_k,
                final_scored.end());

        for (size_t i = 0; i < actual_k; i++) {
            distances[q * k + i] = final_scored[i].first;
            labels[q * k + i] = final_scored[i].second;
        }
        for (size_t i = actual_k; i < static_cast<size_t>(k); i++) {
            distances[q * k + i] = std::numeric_limits<float>::max();
            labels[q * k + i] = -1;
        }
    }
}

} // namespace faiss
