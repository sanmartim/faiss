/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroWeighted.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

namespace faiss {

IndexNeuroWeighted::IndexNeuroWeighted(Index* inner, bool own_inner)
        : IndexNeuro(inner, own_inner) {
    weights.assign(d, 1.0f);
}

void IndexNeuroWeighted::train(idx_t /*n*/, const float* /*x*/) {
    weights.assign(d, 1.0f);
    feedback_count = 0;
    is_trained = true;
}

void IndexNeuroWeighted::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is null");

    auto* flat = dynamic_cast<IndexFlat*>(inner_index);
    FAISS_THROW_IF_NOT_MSG(flat, "inner_index must be IndexFlat");

    const float* xb = flat->get_xb();
    idx_t nb = inner_index->ntotal;

    // Use per-query weights override or index weights
    const float* w = weights.data();
    const NeuroWeightedParams* wp = nullptr;
    if (params_in) {
        wp = dynamic_cast<const NeuroWeightedParams*>(params_in);
    }
    if (wp && !wp->weights.empty()) {
        FAISS_THROW_IF_NOT_MSG(
                (int)wp->weights.size() == d,
                "override weights size must match d");
        w = wp->weights.data();
    }

    bool collect = false;
    if (params_in) {
        auto* nsp = dynamic_cast<const NeuroSearchParameters*>(params_in);
        if (nsp) {
            collect = nsp->collect_stats;
        }
    }

    int64_t total_calcs = 0;

#pragma omp parallel for reduction(+ : total_calcs)
    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * d;

        // Compute weighted L2 to all database vectors
        std::vector<std::pair<float, idx_t>> dists(nb);
        for (idx_t i = 0; i < nb; i++) {
            const float* vec = xb + i * d;
            float dist = 0.0f;
            for (int j = 0; j < d; j++) {
                float diff = query[j] - vec[j];
                dist += w[j] * diff * diff;
            }
            dists[i] = {dist, i};
        }
        total_calcs += nb * d;

        // Partial sort for top-k
        std::partial_sort(
                dists.begin(),
                dists.begin() + std::min((idx_t)dists.size(), k),
                dists.end());

        for (idx_t i = 0; i < k && i < nb; i++) {
            distances[q * k + i] = dists[i].first;
            labels[q * k + i] = dists[i].second;
        }
        for (idx_t i = nb; i < k; i++) {
            distances[q * k + i] = std::numeric_limits<float>::max();
            labels[q * k + i] = -1;
        }
    }

    if (collect) {
        last_stats.calculations_performed = total_calcs;
        last_stats.columns_used = d;
    }
}

void IndexNeuroWeighted::feedback(
        idx_t nq,
        const float* queries,
        const float* positives,
        const float* negatives) {
    FAISS_THROW_IF_NOT_MSG(
            (int)weights.size() == d, "weights not initialized, call train()");

    // Accumulate gradient across all query triplets
    std::vector<float> gradient(d, 0.0f);

    for (idx_t q = 0; q < nq; q++) {
        const float* qvec = queries + q * d;
        const float* pvec = positives + q * d;
        const float* nvec = negatives + q * d;

        for (int j = 0; j < d; j++) {
            float dp = (qvec[j] - pvec[j]) * (qvec[j] - pvec[j]);
            float dn = (qvec[j] - nvec[j]) * (qvec[j] - nvec[j]);

            // If dimension j is closer for positive than negative,
            // it's a "good" dimension -> increase weight.
            // If closer for negative, it's misleading -> decrease weight.
            if (dp < dn) {
                gradient[j] += 1.0f; // good: pos closer
            } else if (dp > dn) {
                gradient[j] -= 1.0f; // bad: neg closer
            }
        }
    }

    // Apply decay
    for (int j = 0; j < d; j++) {
        weights[j] *= weight_decay;
    }

    // Apply gradient with learning rate
    float scale = learning_rate / std::max(nq, (idx_t)1);
    for (int j = 0; j < d; j++) {
        weights[j] += scale * gradient[j];
        weights[j] = std::max(weights[j], min_weight);
    }

    feedback_count++;
}

void IndexNeuroWeighted::feedback_contrastive(
        idx_t nq,
        const float* queries,
        const float* positives,
        const float* negatives,
        int n_negatives,
        float margin_scale) {
    FAISS_THROW_IF_NOT_MSG(
            (int)weights.size() == d, "weights not initialized, call train()");
    FAISS_THROW_IF_NOT_MSG(n_negatives >= 1, "need at least 1 negative");

    // MR-03v2: Momentum velocity accumulator
    std::vector<float> velocity(d, 0.0f);

    // MR-03v2: Number of hard negatives to consider per query
    int n_hard = std::max(1, static_cast<int>(n_negatives * hard_negative_ratio));

    // MR-03v2: Run multiple iterations
    for (int iter = 0; iter < contrastive_iterations; iter++) {
        std::vector<float> gradient(d, 0.0f);

        for (idx_t q = 0; q < nq; q++) {
            const float* qvec = queries + q * d;
            const float* pvec = positives + q * d;

            // MR-03v2: Hard negative mining - find top n_hard closest negatives
            std::vector<std::pair<float, int>> neg_dists(n_negatives);
            for (int n = 0; n < n_negatives; n++) {
                const float* nvec = negatives + (q * n_negatives + n) * d;
                float dist = 0.0f;
                for (int j = 0; j < d; j++) {
                    float diff = qvec[j] - nvec[j];
                    dist += weights[j] * diff * diff;
                }
                neg_dists[n] = {dist, n};
            }

            // Sort by distance (ascending) to get hardest negatives
            std::partial_sort(
                    neg_dists.begin(),
                    neg_dists.begin() + n_hard,
                    neg_dists.end());

            // MR-03v2: Accumulate gradient from top n_hard negatives
            for (int h = 0; h < n_hard; h++) {
                int neg_idx = neg_dists[h].second;
                const float* nvec = negatives + (q * n_negatives + neg_idx) * d;

                // Margin-based contrastive update
                for (int j = 0; j < d; j++) {
                    float dp = (qvec[j] - pvec[j]) * (qvec[j] - pvec[j]);
                    float dn = (qvec[j] - nvec[j]) * (qvec[j] - nvec[j]);

                    // Per-dimension margin scales the gradient magnitude
                    float margin = std::abs(dp - dn);
                    float scaled_margin = 1.0f + margin_scale * margin;

                    if (dp < dn) {
                        gradient[j] += scaled_margin;
                    } else if (dp > dn) {
                        gradient[j] -= scaled_margin;
                    }
                }
            }
        }

        // Normalize gradient by number of samples
        float sample_count = static_cast<float>(nq * n_hard);
        if (sample_count > 0) {
            for (int j = 0; j < d; j++) {
                gradient[j] /= sample_count;
            }
        }

        // MR-03v2: Apply momentum update
        for (int j = 0; j < d; j++) {
            velocity[j] = contrastive_momentum * velocity[j] + gradient[j];
        }

        // Apply decay
        for (int j = 0; j < d; j++) {
            weights[j] *= weight_decay;
        }

        // Apply velocity (momentum-accumulated gradient) with learning rate
        for (int j = 0; j < d; j++) {
            weights[j] += learning_rate * velocity[j];
            weights[j] = std::max(weights[j], min_weight);
        }
    }

    feedback_count++;
}

void IndexNeuroWeighted::save_weights(const char* fname) const {
    FILE* f = fopen(fname, "wb");
    FAISS_THROW_IF_NOT_FMT(f, "could not open %s for writing", fname);

    int32_t dim = d;
    int32_t fc = feedback_count;
    fwrite(&dim, sizeof(int32_t), 1, f);
    fwrite(&fc, sizeof(int32_t), 1, f);
    fwrite(weights.data(), sizeof(float), d, f);
    fclose(f);
}

void IndexNeuroWeighted::load_weights(const char* fname) {
    FILE* f = fopen(fname, "rb");
    FAISS_THROW_IF_NOT_FMT(f, "could not open %s for reading", fname);

    int32_t dim, fc;
    size_t nr;
    nr = fread(&dim, sizeof(int32_t), 1, f);
    FAISS_THROW_IF_NOT_MSG(nr == 1, "failed to read dimension");
    nr = fread(&fc, sizeof(int32_t), 1, f);
    FAISS_THROW_IF_NOT_MSG(nr == 1, "failed to read feedback_count");
    FAISS_THROW_IF_NOT_FMT(
            dim == d,
            "dimension mismatch: file has %d, index has %d",
            dim,
            d);

    weights.resize(dim);
    nr = fread(weights.data(), sizeof(float), dim, f);
    FAISS_THROW_IF_NOT_MSG(nr == (size_t)dim, "failed to read weights");
    feedback_count = fc;
    fclose(f);
}

} // namespace faiss
