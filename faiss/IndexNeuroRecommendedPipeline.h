/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/NeuroDistance.h>
#include <vector>

namespace faiss {

/** FS-06: Recommended Pipeline Index
 *
 * Battle-tested pipeline combining MS-01 (multi-scale sign) + FS-01
 * (Hamming refinement) for optimal recall/speed tradeoff.
 *
 * Pipeline:
 * 1. Stage 1: Multi-scale sign filtering (~15% candidates remain)
 * 2. Stage 2: Hamming prefilter refinement (~5% candidates remain)
 * 3. Stage 3: Precise L2 on survivors
 *
 * Typical performance: 96-99% recall at 10-20x speedup
 * No tuning required for default configuration.
 */
struct IndexNeuroRecommendedPipeline : IndexNeuro {
    /// Keep ratio after MS stage
    float ms_keep_ratio = 0.15f;

    /// Keep ratio after FS stage
    float fs_keep_ratio = 0.05f;

    /// Multi-scale sign scales
    std::vector<float> scales = {0.5f, 1.0f, 2.0f};

    /// MS signatures per scale
    std::vector<std::vector<uint64_t>> ms_signatures;
    int ms_words_per_vec = 0;

    /// FS Hamming thresholds and codes
    std::vector<float> fs_thresholds;
    std::vector<uint64_t> fs_codes;
    int fs_words_per_vec = 0;

    /// Original vectors
    std::vector<float> vectors;

    /// Default constructor
    IndexNeuroRecommendedPipeline() = default;

    /** Construct with dimension.
     * @param d vector dimensionality
     */
    explicit IndexNeuroRecommendedPipeline(int d);

    void train(idx_t n, const float* x) override;
    void add(idx_t n, const float* x) override;
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;
    void reset() override;

protected:
    /// Compute MS signature
    void compute_ms_signature(
            const float* vec,
            float scale,
            uint64_t* sig) const;

    /// Compute FS code
    void compute_fs_code(const float* vec, uint64_t* code) const;

    /// Hamming distance
    int hamming_distance(
            const uint64_t* sig1,
            const uint64_t* sig2,
            int words) const;
};

} // namespace faiss
