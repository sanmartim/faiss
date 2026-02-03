/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroPQAware.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>

namespace faiss {

IndexNeuroPQAware::IndexNeuroPQAware(int d, int M, int nbits)
        : IndexNeuro(nullptr, false) {
    this->d = d;
    pq = new ProductQuantizer(d, M, nbits);
    own_pq = true;
    is_trained = false;
}

IndexNeuroPQAware::IndexNeuroPQAware(ProductQuantizer* pq_in, bool own_pq)
        : IndexNeuro(nullptr, false), pq(pq_in), own_pq(own_pq) {
    if (pq) {
        d = pq->d;
    }
    is_trained = pq && !pq->centroids.empty();
}

IndexNeuroPQAware::~IndexNeuroPQAware() {
    if (own_pq && pq) {
        delete pq;
    }
}

void IndexNeuroPQAware::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(pq, "pq is not set");
    FAISS_THROW_IF_NOT_MSG(d > 0, "dimension must be positive");

    if (pq->centroids.empty()) {
        pq->train(n, x);
    }

    // Initialize subquantizer weights
    if (filter_mode == PQ_FILTER_WEIGHTED && subq_weights.empty()) {
        subq_weights.resize(pq->M, 1.0f / pq->M);
    }

    is_trained = true;
}

void IndexNeuroPQAware::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained before adding");

    size_t old_size = codes.size();
    codes.resize(old_size + n * pq->code_size);

    // Encode vectors
    pq->compute_codes(x, codes.data() + old_size, n);

    // Optionally store original vectors
    if (store_original) {
        size_t old_orig = orig_vectors.size();
        orig_vectors.resize(old_orig + n * d);
        std::copy(x, x + n * d, orig_vectors.data() + old_orig);
    }

    ntotal += n;
}

void IndexNeuroPQAware::reset() {
    codes.clear();
    orig_vectors.clear();
    ntotal = 0;
}

void IndexNeuroPQAware::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT_MSG(pq, "pq is not set");
    FAISS_THROW_IF_NOT(key >= 0 && key < ntotal);

    if (store_original && !orig_vectors.empty()) {
        std::copy(
                orig_vectors.data() + key * d,
                orig_vectors.data() + (key + 1) * d,
                recons);
    } else {
        pq->decode(codes.data() + key * pq->code_size, recons, 1);
    }
}

const uint8_t* IndexNeuroPQAware::get_code(idx_t i) const {
    FAISS_THROW_IF_NOT(i >= 0 && i < ntotal);
    return codes.data() + i * pq->code_size;
}

void IndexNeuroPQAware::compute_distance_table(
        const float* x,
        float* dis_table) const {
    pq->compute_distance_table(x, dis_table);
}

float IndexNeuroPQAware::compute_adc_distance(
        const float* query,
        const uint8_t* code,
        const float* dis_table) const {
    float dist = 0.0f;
    int M = pq->M;
    int ksub = pq->ksub;

    for (int m = 0; m < M; m++) {
        float sub_dist = dis_table[m * ksub + code[m]];

        // Apply subquantizer weight if in weighted mode
        if (filter_mode == PQ_FILTER_WEIGHTED && !subq_weights.empty()) {
            sub_dist *= subq_weights[m];
        }

        dist += sub_dist;
    }

    return dist;
}

void IndexNeuroPQAware::filter_candidates(
        const float* dis_table,
        const std::vector<std::pair<float, idx_t>>& candidates,
        std::vector<std::pair<float, idx_t>>& filtered) const {
    filtered.clear();

    if (filter_mode == PQ_FILTER_NONE) {
        filtered = candidates;
        return;
    }

    int M = pq->M;
    int ksub = pq->ksub;

    for (const auto& cand : candidates) {
        const uint8_t* code = get_code(cand.second);

        bool keep = true;

        if (filter_mode == PQ_FILTER_DISPERSION) {
            // Check dispersion: variance of subquantizer distances
            float mean = cand.first / M;
            float variance = 0.0f;

            for (int m = 0; m < M; m++) {
                float sub_dist = dis_table[m * ksub + code[m]];
                float diff = sub_dist - mean;
                variance += diff * diff;
            }
            variance /= M;

            // Skip if too uniform (low dispersion)
            float dispersion = std::sqrt(variance) / (mean + 1e-10f);
            if (dispersion < dispersion_threshold) {
                keep = false;
            }
        }

        if (keep) {
            filtered.push_back(cand);
        }
    }
}

float IndexNeuroPQAware::compute_subq_entropy(int subq_idx) const {
    FAISS_THROW_IF_NOT(subq_idx >= 0 && subq_idx < pq->M);

    int ksub = pq->ksub;
    std::vector<int> counts(ksub, 0);

    // Count code occurrences
    for (idx_t i = 0; i < ntotal; i++) {
        const uint8_t* code = get_code(i);
        counts[code[subq_idx]]++;
    }

    // Compute entropy
    float entropy = 0.0f;
    for (int k = 0; k < ksub; k++) {
        if (counts[k] > 0) {
            float p = static_cast<float>(counts[k]) / ntotal;
            entropy -= p * std::log2(p);
        }
    }

    return entropy;
}

void IndexNeuroPQAware::learn_weights(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained");
    FAISS_THROW_IF_NOT_MSG(n > 0 && x != nullptr, "need training data");

    int M = pq->M;
    subq_weights.resize(M);

    if (filter_mode == PQ_FILTER_ENTROPY) {
        // Weight by entropy (higher entropy = more discriminative)
        float total_entropy = 0.0f;
        for (int m = 0; m < M; m++) {
            subq_weights[m] = compute_subq_entropy(m);
            total_entropy += subq_weights[m];
        }

        // Normalize
        if (total_entropy > 0) {
            for (int m = 0; m < M; m++) {
                subq_weights[m] /= total_entropy;
            }
        } else {
            std::fill(subq_weights.begin(), subq_weights.end(), 1.0f / M);
        }
    } else {
        // Learn from reconstruction error per subquantizer
        std::vector<float> errors(M, 0.0f);
        std::vector<float> recons(d);
        std::vector<uint8_t> code(pq->code_size);

        idx_t n_sample = std::min(n, static_cast<idx_t>(1000));

        for (idx_t i = 0; i < n_sample; i++) {
            const float* xi = x + i * d;
            pq->compute_code(xi, code.data());
            pq->decode(code.data(), recons.data(), 1);

            // Compute per-subquantizer error
            int dsub = d / M;
            for (int m = 0; m < M; m++) {
                float err = 0.0f;
                for (int j = 0; j < dsub; j++) {
                    float diff = xi[m * dsub + j] - recons[m * dsub + j];
                    err += diff * diff;
                }
                errors[m] += err;
            }
        }

        // Inverse error weighting (lower error = higher weight)
        float total_inv = 0.0f;
        for (int m = 0; m < M; m++) {
            errors[m] /= n_sample;
            subq_weights[m] = 1.0f / (errors[m] + 1e-10f);
            total_inv += subq_weights[m];
        }

        // Normalize
        for (int m = 0; m < M; m++) {
            subq_weights[m] /= total_inv;
        }
    }
}

void IndexNeuroPQAware::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(pq, "pq is not set");
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained");

    // Resolve parameters
    int n_cand = n_candidates;
    bool do_rerank = rerank;

    auto pqp = dynamic_cast<const NeuroPQAwareParams*>(params);
    if (pqp) {
        if (pqp->candidates > 0) {
            n_cand = pqp->candidates;
        }
        do_rerank = pqp->rerank;
    }

    bool collect = false;
    if (params) {
        auto nsp = dynamic_cast<const NeuroSearchParameters*>(params);
        if (nsp) {
            collect = nsp->collect_stats;
        }
    }

    int64_t total_calcs = 0;
    int M = pq->M;
    int ksub = pq->ksub;

#pragma omp parallel for reduction(+ : total_calcs)
    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * d;

        // Compute distance table
        std::vector<float> dis_table(M * ksub);
        compute_distance_table(query, dis_table.data());

        // Compute ADC distances to all codes
        std::vector<std::pair<float, idx_t>> scored(ntotal);
        for (idx_t i = 0; i < ntotal; i++) {
            float dist = compute_adc_distance(
                    query, codes.data() + i * pq->code_size, dis_table.data());
            scored[i] = {dist, i};
        }

        // Get top candidates
        size_t actual_cand =
                std::min(static_cast<size_t>(n_cand), static_cast<size_t>(ntotal));
        std::partial_sort(
                scored.begin(), scored.begin() + actual_cand, scored.end());

        // Apply neuro filtering
        std::vector<std::pair<float, idx_t>> filtered;
        std::vector<std::pair<float, idx_t>> top_cand(
                scored.begin(), scored.begin() + actual_cand);
        filter_candidates(dis_table.data(), top_cand, filtered);

        // Ensure we have enough candidates after filtering
        if (filtered.size() < static_cast<size_t>(k)) {
            filtered = top_cand;
        }

        // Rerank with true distance if requested
        if (do_rerank && (store_original || inner_index)) {
            std::vector<float> recons(d);
            for (auto& cand : filtered) {
                reconstruct(cand.second, recons.data());

                float dist;
                if (metric) {
                    dist = metric->distance(query, recons.data(), d);
                } else {
                    dist = fvec_L2sqr(query, recons.data(), d);
                }
                cand.first = dist;
                total_calcs += d;
            }

            std::partial_sort(
                    filtered.begin(),
                    filtered.begin() + std::min(filtered.size(), static_cast<size_t>(k)),
                    filtered.end());
        }

        // Output results
        size_t actual_k = std::min(static_cast<size_t>(k), filtered.size());
        for (size_t i = 0; i < actual_k; i++) {
            distances[q * k + i] = filtered[i].first;
            labels[q * k + i] = filtered[i].second;
        }
        for (size_t i = actual_k; i < static_cast<size_t>(k); i++) {
            distances[q * k + i] = std::numeric_limits<float>::max();
            labels[q * k + i] = -1;
        }
    }

    if (collect) {
        last_stats.calculations_performed = total_calcs;
    }
}

} // namespace faiss
