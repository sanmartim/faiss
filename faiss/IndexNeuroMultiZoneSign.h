/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/NeuroDistance.h>
#include <vector>
#include <functional>

namespace faiss {

/// Trigonometric function types for zone encoding
enum class TrigFunction {
    TANH,       // tanh(x * scale) - standard
    SIGMOID,    // 1 / (1 + exp(-x * scale))
    SIN,        // sin(x * scale)
    ATAN,       // atan(x * scale) / (pi/2)
    ERF,        // erf(x * scale)
    SOFTSIGN    // x / (1 + |x|)
};

/// Parameters for MultiZoneSign search
struct NeuroMultiZoneSignParams : NeuroSearchParameters {
    float keep_ratio = 0.2f;
    ~NeuroMultiZoneSignParams() override = default;
};

/** V5-MZS: Multi-Zone Sign Index.
 *
 * Experimental index testing different configurations:
 * - Many scales (100, 300, 1000)
 * - Different trig functions (tanh, sigmoid, sin, atan, erf)
 * - 2-bit quantization of trig output
 *
 * For each scale s and dimension j:
 *   code[s][j] = quantize(trig(x[j] * scale[s]))
 *
 * Where quantize maps [-1, 1] to {0, 1, 2, 3} (2 bits)
 */
struct IndexNeuroMultiZoneSign : IndexNeuro {
    /// Number of scales
    int n_scales = 100;

    /// Bits per zone (2 = 4 zones, 3 = 8 zones)
    int bits_per_zone = 2;

    /// Trigonometric function to use
    TrigFunction trig_func = TrigFunction::TANH;

    /// Scale range (min, max) - scales are logarithmically spaced
    float scale_min = 0.01f;
    float scale_max = 10.0f;

    /// Keep ratio for candidate filtering
    float keep_ratio = 0.2f;

    // Computed values
    std::vector<float> scales;
    int n_zones = 4;
    int bytes_per_vec = 0;

    // Per-scale zone codes: n_scales vectors of (ntotal * bytes_per_vec)
    std::vector<std::vector<uint8_t>> zone_codes;

    // Original vectors for L2 reranking
    std::vector<float> vectors;

    // Per-dimension, per-scale zone boundaries (learned from data)
    // Shape: n_scales * d * (n_zones + 1)
    std::vector<float> zone_boundaries;

    IndexNeuroMultiZoneSign() = default;

    /** Construct MultiZoneSign index.
     * @param d dimension
     * @param n_scales number of scales (100, 300, 1000)
     * @param trig_func trigonometric function to use
     * @param bits_per_zone bits per zone (2 or 3)
     */
    IndexNeuroMultiZoneSign(
            int d,
            int n_scales = 100,
            TrigFunction trig_func = TrigFunction::TANH,
            int bits_per_zone = 2);

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

    /// Apply the trigonometric function
    float apply_trig(float x, float scale) const;

    /// Compute zone code for a vector at given scale index
    void compute_zone_code(
            const float* vec,
            int scale_idx,
            uint8_t* code) const;

    /// Compute zone distance between two codes
    int zone_distance(const uint8_t* code1, const uint8_t* code2) const;
};

} // namespace faiss
