/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroAdaptiveProbe.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace faiss {

IndexNeuroAdaptiveProbe::IndexNeuroAdaptiveProbe(
        IndexIVF* ivf_index,
        int nprobe_min,
        int nprobe_max)
        : IndexNeuro(nullptr, false),
          ivf_index(ivf_index),
          nprobe_min(nprobe_min),
          nprobe_max(nprobe_max) {
    if (ivf_index) {
        this->d = ivf_index->d;
        this->ntotal = ivf_index->ntotal;
        this->is_trained = ivf_index->is_trained;
        nprobe_default = ivf_index->nprobe;
    }
}

void IndexNeuroAdaptiveProbe::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(ivf_index, "ivf_index is not set");
    ivf_index->train(n, x);
    is_trained = ivf_index->is_trained;
}

void IndexNeuroAdaptiveProbe::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(ivf_index, "ivf_index is not set");
    ivf_index->add(n, x);
    ntotal = ivf_index->ntotal;
}

void IndexNeuroAdaptiveProbe::reset() {
    FAISS_THROW_IF_NOT_MSG(ivf_index, "ivf_index is not set");
    ivf_index->reset();
    ntotal = 0;
    reset_stats();
}

void IndexNeuroAdaptiveProbe::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT_MSG(ivf_index, "ivf_index is not set");
    ivf_index->reconstruct(key, recons);
}

int IndexNeuroAdaptiveProbe::compute_nprobe(const float* query) const {
    if (!ivf_index || !ivf_index->quantizer) {
        return nprobe_default;
    }

    // Get distances to top clusters
    int nlist = ivf_index->nlist;
    int check_k = std::min(nprobe_max + 1, nlist);

    std::vector<float> distances(check_k);
    std::vector<idx_t> labels(check_k);

    ivf_index->quantizer->search(1, query, check_k, distances.data(), labels.data());

    if (check_k < 2) {
        return nprobe_default;
    }

    // Compute gap ratio between first and second cluster
    float d1 = distances[0];
    float d2 = distances[1];

    if (d1 < 1e-10f) {
        // Query is very close to a centroid - easy query
        return nprobe_min;
    }

    float gap_ratio = (d2 - d1) / d1;

    // Large gap = easy query (clearly belongs to one cluster)
    // Small gap = hard query (on boundary between clusters)
    if (gap_ratio > gap_threshold * 2) {
        return nprobe_min;
    } else if (gap_ratio < gap_threshold / 2) {
        return nprobe_max;
    } else {
        // Linear interpolation
        float t = (gap_ratio - gap_threshold / 2) / (gap_threshold * 1.5f);
        t = std::max(0.0f, std::min(1.0f, t));
        return static_cast<int>(nprobe_max - t * (nprobe_max - nprobe_min));
    }
}

void IndexNeuroAdaptiveProbe::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(ivf_index, "ivf_index is not set");

    // Resolve parameters
    float thresh = gap_threshold;
    auto ap = dynamic_cast<const NeuroAdaptiveProbeParams*>(params);
    if (ap && ap->gap_threshold >= 0) {
        thresh = ap->gap_threshold;
    }

    // Process each query with adaptive nprobe
    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * d;

        // Compute adaptive nprobe for this query
        int adaptive_nprobe = compute_nprobe(query);

        // Track statistics
        if (adaptive_nprobe <= nprobe_min + (nprobe_max - nprobe_min) / 3) {
            easy_queries++;
        } else {
            hard_queries++;
        }

        // Temporarily set nprobe and search
        int old_nprobe = ivf_index->nprobe;
        ivf_index->nprobe = adaptive_nprobe;

        ivf_index->search(
                1,
                query,
                k,
                distances + q * k,
                labels + q * k,
                params);

        ivf_index->nprobe = old_nprobe;
    }
}

float IndexNeuroAdaptiveProbe::get_avg_nprobe() const {
    int64_t total = easy_queries + hard_queries;
    if (total == 0) return static_cast<float>(nprobe_default);

    // Estimate average based on easy/hard ratio
    float easy_ratio = static_cast<float>(easy_queries) / total;
    return easy_ratio * nprobe_min + (1 - easy_ratio) * nprobe_max;
}

void IndexNeuroAdaptiveProbe::reset_stats() {
    easy_queries = 0;
    hard_queries = 0;
}

} // namespace faiss
