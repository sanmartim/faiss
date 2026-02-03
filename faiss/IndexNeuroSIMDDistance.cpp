/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroSIMDDistance.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace faiss {

IndexNeuroSIMDDistance::IndexNeuroSIMDDistance(Index* sub_index, int batch_size)
        : IndexNeuro(sub_index, false), sub_index(sub_index), batch_size(batch_size) {
    if (sub_index) {
        use_inner_product = (sub_index->metric_type == METRIC_INNER_PRODUCT);
    }
}

void IndexNeuroSIMDDistance::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");
    sub_index->train(n, x);
    is_trained = sub_index->is_trained;
}

void IndexNeuroSIMDDistance::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");
    sub_index->add(n, x);
    ntotal = sub_index->ntotal;

    // Invalidate precomputed norms
    norms_computed = false;
}

void IndexNeuroSIMDDistance::reset() {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");
    sub_index->reset();
    ntotal = 0;
    db_norms.clear();
    norms_computed = false;
}

void IndexNeuroSIMDDistance::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");
    sub_index->reconstruct(key, recons);
}

const float* IndexNeuroSIMDDistance::get_xb() const {
    // Try to get raw vectors from IndexFlat
    auto flat = dynamic_cast<IndexFlat*>(sub_index);
    if (flat) {
        return flat->get_xb();
    }

    // Try IndexNeuro wrapping IndexFlat
    auto neuro = dynamic_cast<IndexNeuro*>(sub_index);
    if (neuro && neuro->inner_index) {
        auto inner_flat = dynamic_cast<IndexFlat*>(neuro->inner_index);
        if (inner_flat) {
            return inner_flat->get_xb();
        }
    }

    return nullptr;
}

void IndexNeuroSIMDDistance::precompute_norms() {
    const float* xb = get_xb();
    if (!xb || !use_inner_product) {
        return;
    }

    db_norms.resize(ntotal);
    for (idx_t i = 0; i < ntotal; i++) {
        db_norms[i] = fvec_norm_L2sqr(xb + i * d, d);
    }
    norms_computed = true;
}

void IndexNeuroSIMDDistance::search_batch_simd(
        idx_t nq,
        const float* xq,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    const float* xb = get_xb();

    if (!xb) {
        // Fallback to sub_index search if we can't access raw vectors
        sub_index->search(nq, xq, k, distances, labels);
        return;
    }

    // Allocate temporary storage for all distances
    std::vector<float> all_distances(nq * ntotal);

    if (use_inner_product) {
        // Compute inner products: batch query × all database
        // IP(q, d) for all pairs
        for (idx_t q = 0; q < nq; q++) {
            fvec_inner_products_ny(
                    all_distances.data() + q * ntotal,
                    xq + q * d,
                    xb,
                    d,
                    ntotal);
        }
    } else {
        // Compute L2 squared distances: batch
        for (idx_t q = 0; q < nq; q++) {
            fvec_L2sqr_ny(
                    all_distances.data() + q * ntotal,
                    xq + q * d,
                    xb,
                    d,
                    ntotal);
        }
    }

    // Find top-k for each query
#pragma omp parallel for
    for (idx_t q = 0; q < nq; q++) {
        float* q_distances = all_distances.data() + q * ntotal;

        // Create (distance, index) pairs
        std::vector<std::pair<float, idx_t>> scored(ntotal);
        for (idx_t i = 0; i < ntotal; i++) {
            float dist = q_distances[i];
            // For inner product, negate for min-heap behavior
            if (use_inner_product) {
                dist = -dist;
            }
            scored[i] = {dist, i};
        }

        // Partial sort to get top-k
        size_t actual_k = std::min(static_cast<size_t>(k), static_cast<size_t>(ntotal));
        std::partial_sort(
                scored.begin(), scored.begin() + actual_k, scored.end());

        // Output results
        for (size_t i = 0; i < actual_k; i++) {
            float dist = scored[i].first;
            if (use_inner_product) {
                dist = -dist;  // Restore original sign
            }
            distances[q * k + i] = dist;
            labels[q * k + i] = scored[i].second;
        }
        for (size_t i = actual_k; i < static_cast<size_t>(k); i++) {
            distances[q * k + i] = use_inner_product
                    ? -std::numeric_limits<float>::max()
                    : std::numeric_limits<float>::max();
            labels[q * k + i] = -1;
        }
    }
}

void IndexNeuroSIMDDistance::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");

    // Resolve parameters
    int bs = batch_size;

    auto sp = dynamic_cast<const NeuroSIMDParams*>(params);
    if (sp && sp->batch_size > 0) {
        bs = sp->batch_size;
    }

    bool collect = false;
    if (params) {
        auto nsp = dynamic_cast<const NeuroSearchParameters*>(params);
        if (nsp) {
            collect = nsp->collect_stats;
        }
    }

    int64_t total_calcs = 0;

    // Check if we can use SIMD optimization
    const float* xb = get_xb();
    if (!xb || ntotal == 0) {
        // Fallback to sub_index search
        sub_index->search(n, x, k, distances, labels, params);
        return;
    }

    // Process queries in batches for better cache utilization
    for (idx_t start = 0; start < n; start += bs) {
        idx_t batch_n = std::min(static_cast<idx_t>(bs), n - start);

        search_batch_simd(
                batch_n,
                x + start * d,
                k,
                distances + start * k,
                labels + start * k);

        total_calcs += batch_n * ntotal * d;
    }

    if (collect) {
        last_stats.calculations_performed = total_calcs;
    }
}

} // namespace faiss
