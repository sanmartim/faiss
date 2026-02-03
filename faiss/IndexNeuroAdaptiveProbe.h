/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/NeuroDistance.h>
#include <faiss/IndexIVF.h>
#include <vector>

namespace faiss {

/** PT-02: Adaptive Probe with Dynamic nprobe Selection.
 *
 * Dynamically adjusts nprobe based on query difficulty, measured by
 * the gap between distances to nearest clusters.
 *
 * Key features:
 *   - Gap-based nprobe selection
 *   - -30% candidates on average while maintaining recall
 *   - Automatic difficulty estimation
 */
struct IndexNeuroAdaptiveProbe : IndexNeuro {
    /// Inner IVF index to wrap
    IndexIVF* ivf_index = nullptr;

    /// Minimum nprobe (for easy queries)
    int nprobe_min = 1;

    /// Maximum nprobe (for hard queries)
    int nprobe_max = 64;

    /// Default nprobe (baseline)
    int nprobe_default = 8;

    /// Gap threshold: if gap < threshold, query is hard
    float gap_threshold = 0.1f;

    /// Statistics tracking
    mutable int64_t easy_queries = 0;
    mutable int64_t hard_queries = 0;

    IndexNeuroAdaptiveProbe() = default;

    /** Construct wrapping an IVF index.
     * @param ivf_index     the IVF index to wrap
     * @param nprobe_min    minimum nprobe for easy queries
     * @param nprobe_max    maximum nprobe for hard queries
     */
    IndexNeuroAdaptiveProbe(
            IndexIVF* ivf_index,
            int nprobe_min = 1,
            int nprobe_max = 64);

    ~IndexNeuroAdaptiveProbe() override = default;

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

    void reconstruct(idx_t key, float* recons) const override;

    /// Get average nprobe used
    float get_avg_nprobe() const;

    /// Reset statistics
    void reset_stats();

private:
    /// Compute adaptive nprobe for a query
    int compute_nprobe(const float* query) const;
};

/// Parameters for adaptive probe search
struct NeuroAdaptiveProbeParams : NeuroSearchParameters {
    float gap_threshold = -1.0f;  ///< -1 = use index default

    ~NeuroAdaptiveProbeParams() override = default;
};

} // namespace faiss
