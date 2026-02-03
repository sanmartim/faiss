/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroScalarQuantization.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace faiss {

IndexNeuroScalarQuantization::IndexNeuroScalarQuantization(int d, int nbits)
        : IndexNeuro(nullptr, false) {
    this->d = d;
    ScalarQuantizer::QuantizerType qtype =
            (nbits == 4) ? ScalarQuantizer::QT_4bit : ScalarQuantizer::QT_8bit;
    sq = new ScalarQuantizer(d, qtype);
    is_trained = false;
}

IndexNeuroScalarQuantization::IndexNeuroScalarQuantization(
        Index* inner_index,
        int nbits,
        int rerank_k)
        : IndexNeuro(inner_index, false), rerank_k(rerank_k) {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index cannot be null");
    ScalarQuantizer::QuantizerType qtype =
            (nbits == 4) ? ScalarQuantizer::QT_4bit : ScalarQuantizer::QT_8bit;
    sq = new ScalarQuantizer(inner_index->d, qtype);
    is_trained = false;
}

IndexNeuroScalarQuantization::~IndexNeuroScalarQuantization() {
    delete sq;
}

void IndexNeuroScalarQuantization::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(sq, "sq is not set");
    FAISS_THROW_IF_NOT_MSG(d > 0, "dimension must be positive");
    FAISS_THROW_IF_NOT_MSG(n > 0, "need training data");

    // Configure calibration method
    switch (calibration) {
        case NEURO_SQ_CALIB_MINMAX:
            sq->rangestat = ScalarQuantizer::RS_minmax;
            sq->rangestat_arg = 0.0f;
            break;
        case NEURO_SQ_CALIB_PERCENTILE:
            sq->rangestat = ScalarQuantizer::RS_quantiles;
            sq->rangestat_arg = percentile_low;
            break;
        case NEURO_SQ_CALIB_OPTIM:
            sq->rangestat = ScalarQuantizer::RS_optim;
            sq->rangestat_arg = 0.0f;
            break;
    }

    sq->train(n, x);
    is_trained = true;
}

void IndexNeuroScalarQuantization::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained before adding");
    FAISS_THROW_IF_NOT_MSG(sq, "sq is not set");

    // Encode vectors
    size_t old_codes = codes.size();
    codes.resize(old_codes + n * sq->code_size);
    sq->compute_codes(x, codes.data() + old_codes, n);

    // Store original vectors for reranking
    size_t old_orig = orig_vectors.size();
    orig_vectors.resize(old_orig + n * d);
    std::copy(x, x + n * d, orig_vectors.data() + old_orig);

    // Update inner index if present
    if (inner_index) {
        inner_index->add(n, x);
    }

    ntotal += n;
}

void IndexNeuroScalarQuantization::reset() {
    codes.clear();
    orig_vectors.clear();
    if (inner_index) {
        inner_index->reset();
    }
    ntotal = 0;
}

void IndexNeuroScalarQuantization::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT(key >= 0 && key < ntotal);

    // Use original vectors
    std::copy(
            orig_vectors.data() + key * d,
            orig_vectors.data() + (key + 1) * d,
            recons);
}

const uint8_t* IndexNeuroScalarQuantization::get_code(idx_t i) const {
    FAISS_THROW_IF_NOT(i >= 0 && i < ntotal);
    return codes.data() + i * sq->code_size;
}

float IndexNeuroScalarQuantization::get_compression_ratio() const {
    if (!sq || sq->code_size == 0) {
        return 1.0f;
    }
    return static_cast<float>(d * sizeof(float)) / sq->code_size;
}

float IndexNeuroScalarQuantization::compute_quantized_distance(
        const float* query,
        const uint8_t* code) const {
    // Decode and compute L2 distance
    std::vector<float> decoded(d);
    sq->decode(code, decoded.data(), 1);
    return fvec_L2sqr(query, decoded.data(), d);
}

void IndexNeuroScalarQuantization::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(sq, "sq is not set");
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained");

    // Resolve parameters
    int n_rerank = rerank_k;
    bool should_rerank = do_rerank;

    auto sqp = dynamic_cast<const NeuroSQParams*>(params);
    if (sqp) {
        if (sqp->rerank_k > 0) {
            n_rerank = sqp->rerank_k;
        }
        should_rerank = sqp->do_rerank;
    }

    bool collect = false;
    if (params) {
        auto nsp = dynamic_cast<const NeuroSearchParameters*>(params);
        if (nsp) {
            collect = nsp->collect_stats;
        }
    }

    int64_t total_calcs = 0;

#pragma omp parallel reduction(+ : total_calcs)
    {
        // Each thread gets its own distance computer
        std::unique_ptr<ScalarQuantizer::SQDistanceComputer> dc(
                sq->get_distance_computer(METRIC_L2));

#pragma omp for
        for (idx_t q = 0; q < n; q++) {
            const float* query = x + q * d;

            // Set query for distance computer
            dc->set_query(query);

            // Compute quantized distances to all codes
            std::vector<std::pair<float, idx_t>> scored(ntotal);
            for (idx_t i = 0; i < ntotal; i++) {
                float dist = dc->query_to_code(codes.data() + i * sq->code_size);
                scored[i] = {dist, i};
            }
            total_calcs += ntotal;

            // Get top rerank candidates
            size_t actual_rerank = std::min(
                    static_cast<size_t>(n_rerank), static_cast<size_t>(ntotal));
            std::partial_sort(
                    scored.begin(), scored.begin() + actual_rerank, scored.end());

            // Rerank with full precision if requested
            if (should_rerank && !orig_vectors.empty()) {
                for (size_t i = 0; i < actual_rerank; i++) {
                    idx_t idx = scored[i].second;
                    const float* orig = orig_vectors.data() + idx * d;

                    float dist;
                    if (metric) {
                        dist = metric->distance(query, orig, d);
                    } else {
                        dist = fvec_L2sqr(query, orig, d);
                    }
                    scored[i].first = dist;
                    total_calcs += d;
                }

                // Re-sort after reranking
                std::partial_sort(
                        scored.begin(),
                        scored.begin() + std::min(actual_rerank, static_cast<size_t>(k)),
                        scored.begin() + actual_rerank);
            }

            // Output results
            size_t actual_k = std::min(static_cast<size_t>(k), actual_rerank);
            for (size_t i = 0; i < actual_k; i++) {
                distances[q * k + i] = scored[i].first;
                labels[q * k + i] = scored[i].second;
            }
            for (size_t i = actual_k; i < static_cast<size_t>(k); i++) {
                distances[q * k + i] = std::numeric_limits<float>::max();
                labels[q * k + i] = -1;
            }
        }
    }

    if (collect) {
        last_stats.calculations_performed = total_calcs;
    }
}

} // namespace faiss
