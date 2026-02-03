/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroBatchedQueries.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace faiss {

IndexNeuroBatchedQueries::IndexNeuroBatchedQueries(Index* sub_index, int batch_size)
        : IndexNeuro(sub_index, false),
          sub_index(sub_index),
          batch_size(batch_size) {}

void IndexNeuroBatchedQueries::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");
    sub_index->train(n, x);
    is_trained = sub_index->is_trained;
}

void IndexNeuroBatchedQueries::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");
    sub_index->add(n, x);
    ntotal = sub_index->ntotal;
    norms_computed = false;  // Invalidate precomputed norms
}

void IndexNeuroBatchedQueries::reset() {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");
    sub_index->reset();
    ntotal = 0;
    xb_norms.clear();
    norms_computed = false;
}

void IndexNeuroBatchedQueries::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");
    sub_index->reconstruct(key, recons);
}

const float* IndexNeuroBatchedQueries::get_xb() const {
    auto flat = dynamic_cast<IndexFlat*>(sub_index);
    if (flat) {
        return flat->get_xb();
    }
    auto neuro = dynamic_cast<IndexNeuro*>(sub_index);
    if (neuro && neuro->inner_index) {
        auto inner_flat = dynamic_cast<IndexFlat*>(neuro->inner_index);
        if (inner_flat) {
            return inner_flat->get_xb();
        }
    }
    return nullptr;
}

void IndexNeuroBatchedQueries::precompute_norms() {
    const float* xb = get_xb();
    if (!xb) return;

    xb_norms.resize(ntotal);
    for (idx_t i = 0; i < ntotal; i++) {
        xb_norms[i] = fvec_norm_L2sqr(xb + i * d, d);
    }
    norms_computed = true;
}

void IndexNeuroBatchedQueries::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");

    // Resolve parameters
    int bs = batch_size;

    auto bp = dynamic_cast<const NeuroBatchedParams*>(params);
    if (bp && bp->batch_size > 0) {
        bs = bp->batch_size;
    }

    const float* xb = get_xb();
    if (!xb || ntotal == 0) {
        // Fall back to sub_index search
        sub_index->search(n, x, k, distances, labels, params);
        return;
    }

    // Compute query norms
    std::vector<float> q_norms(n);
    for (idx_t i = 0; i < n; i++) {
        q_norms[i] = fvec_norm_L2sqr(x + i * d, d);
    }

    // Get or compute database norms
    std::vector<float> db_norms;
    if (norms_computed) {
        db_norms = xb_norms;
    } else {
        db_norms.resize(ntotal);
        for (idx_t i = 0; i < ntotal; i++) {
            db_norms[i] = fvec_norm_L2sqr(xb + i * d, d);
        }
    }

    // Process queries in batches
    for (idx_t batch_start = 0; batch_start < n; batch_start += bs) {
        idx_t batch_n = std::min(static_cast<idx_t>(bs), n - batch_start);
        const float* batch_q = x + batch_start * d;

        // Compute inner products: batch_n × ntotal matrix
        std::vector<float> ip_matrix(batch_n * ntotal);

        // Use batch inner product
        for (idx_t qi = 0; qi < batch_n; qi++) {
            fvec_inner_products_ny(
                    ip_matrix.data() + qi * ntotal,
                    batch_q + qi * d,
                    xb,
                    d,
                    ntotal);
        }

        // Convert to L2 distances: ||q||^2 + ||x||^2 - 2*<q,x>
#pragma omp parallel for
        for (idx_t qi = 0; qi < batch_n; qi++) {
            float q_norm = q_norms[batch_start + qi];

            // Compute distances and find top-k
            std::vector<std::pair<float, idx_t>> scored(ntotal);
            for (idx_t i = 0; i < ntotal; i++) {
                float ip = ip_matrix[qi * ntotal + i];
                float dist = q_norm + db_norms[i] - 2.0f * ip;
                // Numerical safety
                dist = std::max(0.0f, dist);
                scored[i] = {dist, i};
            }

            // Partial sort for top-k
            size_t actual_k = std::min(static_cast<size_t>(k), static_cast<size_t>(ntotal));
            std::partial_sort(
                    scored.begin(),
                    scored.begin() + actual_k,
                    scored.end());

            // Output results
            for (size_t i = 0; i < actual_k; i++) {
                distances[(batch_start + qi) * k + i] = scored[i].first;
                labels[(batch_start + qi) * k + i] = scored[i].second;
            }
            for (size_t i = actual_k; i < static_cast<size_t>(k); i++) {
                distances[(batch_start + qi) * k + i] = std::numeric_limits<float>::max();
                labels[(batch_start + qi) * k + i] = -1;
            }
        }
    }
}

} // namespace faiss
