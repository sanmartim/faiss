/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/NeuroDistance.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <memory>
#include <vector>

namespace faiss {

/// Calibration method for scalar quantization
enum NeuroSQCalibration {
    NEURO_SQ_CALIB_MINMAX = 0,
    NEURO_SQ_CALIB_PERCENTILE,
    NEURO_SQ_CALIB_OPTIM,
};

/** QT-01: Scalar Quantization with Calibration and Rerank.
 *
 * Applies per-dimension scalar quantization (int8 or int4) with
 * configurable calibration methods and optional reranking with
 * full-precision vectors.
 *
 * Key features:
 *   - Wraps FAISS ScalarQuantizer (int8, int4, fp16)
 *   - Calibration: minmax, percentile, optimized
 *   - Configurable rerank candidates
 *   - Pluggable NeuroMetric for reranking
 *   - ~4x compression with >=98% recall
 */
struct IndexNeuroScalarQuantization : IndexNeuro {
    /// Scalar quantizer (owned)
    ScalarQuantizer* sq = nullptr;

    /// Quantized codes storage
    std::vector<uint8_t> codes;

    /// Original vectors for reranking
    std::vector<float> orig_vectors;

    /// Calibration method
    NeuroSQCalibration calibration = NEURO_SQ_CALIB_PERCENTILE;

    /// Percentile args for calibration (lower bound)
    float percentile_low = 0.001f;

    /// Percentile args for calibration (upper bound)
    float percentile_high = 0.999f;

    /// Number of candidates before rerank
    int rerank_k = 100;

    /// Whether to rerank with original vectors
    bool do_rerank = true;

    /// Optional pluggable metric for reranking
    NeuroMetric* metric = nullptr;

    IndexNeuroScalarQuantization() = default;

    /** Construct with dimension and quantizer type.
     * @param d      dimension
     * @param nbits  bits per code (8 for int8, 4 for int4)
     */
    explicit IndexNeuroScalarQuantization(int d, int nbits = 8);

    /** Construct wrapping existing inner index.
     * @param inner_index  the index to wrap
     * @param nbits        bits per code
     * @param rerank_k     number of candidates for reranking
     */
    IndexNeuroScalarQuantization(
            Index* inner_index,
            int nbits = 8,
            int rerank_k = 100);

    ~IndexNeuroScalarQuantization() override;

    void train(idx_t n, const float* x) override;
    void add(idx_t n, const float* x) override;
    void reset() override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /// Reconstruct vector from original storage
    void reconstruct(idx_t key, float* recons) const override;

    /// Get code for a specific vector
    const uint8_t* get_code(idx_t i) const;

    /// Get compression ratio
    float get_compression_ratio() const;

private:
    /// Compute quantized distance between query and code
    float compute_quantized_distance(
            const float* query,
            const uint8_t* code) const;
};

/// Parameters for SQ search
struct NeuroSQParams : NeuroSearchParameters {
    int rerank_k = -1;
    bool do_rerank = true;

    ~NeuroSQParams() override = default;
};

} // namespace faiss
