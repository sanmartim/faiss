/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroMultiResolutionBinary.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/hamming.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace faiss {

IndexNeuroMultiResolutionBinary::IndexNeuroMultiResolutionBinary(int d, int rerank_k)
        : IndexNeuro(nullptr, false), rerank_k(rerank_k) {
    this->d = d;

    // Default 4-level configuration
    num_levels = 4;
    level_bits = {1, 2, 4, 8};
    keep_ratios = {0.5f, 0.3f, 0.2f, 0.1f};  // Progressive filtering

    level_codes.resize(num_levels);
    is_trained = true;
}

size_t IndexNeuroMultiResolutionBinary::code_size_at_level(int level) const {
    int bits = level_bits[level];
    // bits per dimension * dimensions / 8 bits per byte
    return (d * bits + 7) / 8;
}

void IndexNeuroMultiResolutionBinary::encode_at_level(
        const float* vec,
        int level,
        uint8_t* code) const {
    int bits = level_bits[level];
    size_t code_sz = code_size_at_level(level);
    std::fill(code, code + code_sz, 0);

    if (bits == 1) {
        // Binary: 1 if > 0, else 0
        for (int i = 0; i < d; i++) {
            if (vec[i] > 0) {
                code[i / 8] |= (1 << (i % 8));
            }
        }
    } else if (bits == 2) {
        // 2-bit quantization: 4 levels
        for (int i = 0; i < d; i++) {
            int q = static_cast<int>((vec[i] + 3.0f) / 1.5f);  // Map [-3,3] to [0,3]
            q = std::max(0, std::min(3, q));
            int byte_idx = (i * 2) / 8;
            int bit_offset = (i * 2) % 8;
            code[byte_idx] |= (q << bit_offset);
        }
    } else if (bits == 4) {
        // 4-bit quantization: 16 levels
        for (int i = 0; i < d; i++) {
            int q = static_cast<int>((vec[i] + 3.0f) / 0.375f);  // Map [-3,3] to [0,15]
            q = std::max(0, std::min(15, q));
            int byte_idx = (i * 4) / 8;
            int bit_offset = (i * 4) % 8;
            code[byte_idx] |= (q << bit_offset);
        }
    } else {
        // 8-bit quantization: 256 levels
        for (int i = 0; i < d; i++) {
            int q = static_cast<int>((vec[i] + 4.0f) / 8.0f * 255);
            q = std::max(0, std::min(255, q));
            code[i] = static_cast<uint8_t>(q);
        }
    }
}

int IndexNeuroMultiResolutionBinary::hamming_distance(
        const uint8_t* a,
        const uint8_t* b,
        int level) const {
    size_t code_sz = code_size_at_level(level);
    int dist = 0;

    for (size_t i = 0; i < code_sz; i++) {
        dist += __builtin_popcount(a[i] ^ b[i]);
    }
    return dist;
}

void IndexNeuroMultiResolutionBinary::train(idx_t /*n*/, const float* /*x*/) {
    // No training needed for simple threshold-based encoding
    is_trained = true;
}

void IndexNeuroMultiResolutionBinary::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained");

    // Encode at each level
    for (int level = 0; level < num_levels; level++) {
        size_t code_sz = code_size_at_level(level);
        size_t old_size = level_codes[level].size();
        level_codes[level].resize(old_size + n * code_sz);

        for (idx_t i = 0; i < n; i++) {
            encode_at_level(
                    x + i * d,
                    level,
                    level_codes[level].data() + old_size + i * code_sz);
        }
    }

    // Store original vectors
    size_t old_orig = orig_vectors.size();
    orig_vectors.resize(old_orig + n * d);
    std::copy(x, x + n * d, orig_vectors.data() + old_orig);

    ntotal += n;
}

void IndexNeuroMultiResolutionBinary::reset() {
    for (auto& codes : level_codes) {
        codes.clear();
    }
    orig_vectors.clear();
    ntotal = 0;
}

void IndexNeuroMultiResolutionBinary::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT(key >= 0 && key < ntotal);
    std::copy(
            orig_vectors.data() + key * d,
            orig_vectors.data() + (key + 1) * d,
            recons);
}

std::vector<float> IndexNeuroMultiResolutionBinary::get_compression_ratios() const {
    std::vector<float> ratios(num_levels);
    float orig_size = d * sizeof(float);
    for (int level = 0; level < num_levels; level++) {
        ratios[level] = orig_size / code_size_at_level(level);
    }
    return ratios;
}

void IndexNeuroMultiResolutionBinary::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(ntotal > 0);

    // Resolve parameters
    int n_rerank = rerank_k;
    auto bp = dynamic_cast<const NeuroMultiResBinaryParams*>(params);
    if (bp && bp->rerank_k > 0) {
        n_rerank = bp->rerank_k;
    }

#pragma omp parallel for
    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * d;

        // Encode query at each level
        std::vector<std::vector<uint8_t>> query_codes(num_levels);
        for (int level = 0; level < num_levels; level++) {
            query_codes[level].resize(code_size_at_level(level));
            encode_at_level(query, level, query_codes[level].data());
        }

        // Start with all candidates
        std::vector<idx_t> candidates(ntotal);
        for (idx_t i = 0; i < ntotal; i++) {
            candidates[i] = i;
        }

        // Cascade through levels
        for (int level = 0; level < num_levels && !candidates.empty(); level++) {
            size_t code_sz = code_size_at_level(level);

            // Score candidates at this level
            std::vector<std::pair<int, idx_t>> scored;
            scored.reserve(candidates.size());

            for (idx_t idx : candidates) {
                const uint8_t* code = level_codes[level].data() + idx * code_sz;
                int dist = hamming_distance(query_codes[level].data(), code, level);
                scored.push_back({dist, idx});
            }

            // Keep top fraction
            size_t keep = std::max(
                    static_cast<size_t>(n_rerank),
                    static_cast<size_t>(candidates.size() * keep_ratios[level]));
            keep = std::min(keep, scored.size());

            std::partial_sort(scored.begin(), scored.begin() + keep, scored.end());

            candidates.clear();
            candidates.reserve(keep);
            for (size_t i = 0; i < keep; i++) {
                candidates.push_back(scored[i].second);
            }
        }

        // Final reranking with full precision
        std::vector<std::pair<float, idx_t>> final_scored;
        final_scored.reserve(candidates.size());

        if (do_rerank && !orig_vectors.empty()) {
            for (idx_t idx : candidates) {
                const float* vec = orig_vectors.data() + idx * d;
                float dist = fvec_L2sqr(query, vec, d);
                final_scored.push_back({dist, idx});
            }
        } else {
            // Use last level's Hamming distance as approximation
            for (idx_t idx : candidates) {
                size_t code_sz = code_size_at_level(num_levels - 1);
                const uint8_t* code = level_codes[num_levels - 1].data() + idx * code_sz;
                int dist = hamming_distance(query_codes[num_levels - 1].data(), code, num_levels - 1);
                final_scored.push_back({static_cast<float>(dist), idx});
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
