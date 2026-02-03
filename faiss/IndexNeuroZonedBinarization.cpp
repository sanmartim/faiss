/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroZonedBinarization.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace faiss {

IndexNeuroZonedBinarization::IndexNeuroZonedBinarization(int d, int rerank_k)
        : IndexNeuro(nullptr, false), rerank_k(rerank_k) {
    this->d = d;
    is_trained = false;
}

IndexNeuroZonedBinarization::IndexNeuroZonedBinarization(
        int d,
        float float_pct,
        float int8_pct,
        int rerank_k)
        : IndexNeuro(nullptr, false), rerank_k(rerank_k) {
    this->d = d;
    zone_config.float_ratio = float_pct;
    zone_config.int8_ratio = int8_pct;
    zone_config.binary_ratio = 1.0f - float_pct - int8_pct;
    is_trained = false;
}

IndexNeuroZonedBinarization::IndexNeuroZonedBinarization(
        Index* inner_index,
        int rerank_k)
        : IndexNeuro(inner_index, false), rerank_k(rerank_k) {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index cannot be null");
    is_trained = false;
}

void IndexNeuroZonedBinarization::analyze_importance(idx_t n, const float* x) {
    dim_importance.resize(d);

    // Compute variance per dimension
    std::vector<double> mean(d, 0.0);
    std::vector<double> var(d, 0.0);

    // Mean pass
    for (idx_t i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            mean[j] += x[i * d + j];
        }
    }
    for (int j = 0; j < d; j++) {
        mean[j] /= n;
    }

    // Variance pass
    for (idx_t i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            double diff = x[i * d + j] - mean[j];
            var[j] += diff * diff;
        }
    }
    for (int j = 0; j < d; j++) {
        var[j] /= n;
    }

    // Also compute neighbor consistency for importance
    // Sample a subset for neighbor analysis
    idx_t n_sample = std::min(n, static_cast<idx_t>(1000));
    std::vector<double> consistency(d, 0.0);

    for (idx_t i = 0; i < n_sample; i++) {
        // Find nearest neighbor (brute force on sample)
        idx_t nn = -1;
        float min_dist = std::numeric_limits<float>::max();
        for (idx_t j = 0; j < n_sample; j++) {
            if (i == j) continue;
            float dist = fvec_L2sqr(x + i * d, x + j * d, d);
            if (dist < min_dist) {
                min_dist = dist;
                nn = j;
            }
        }

        if (nn >= 0) {
            // Measure per-dimension contribution to distance
            for (int j = 0; j < d; j++) {
                float diff = x[i * d + j] - x[nn * d + j];
                consistency[j] += diff * diff;
            }
        }
    }

    // Combine variance and consistency
    for (int j = 0; j < d; j++) {
        // Importance = variance * (1 + normalized_consistency)
        float norm_consistency = static_cast<float>(consistency[j]) / n_sample;
        dim_importance[j] = static_cast<float>(var[j]) * (1.0f + norm_consistency);
    }
}

void IndexNeuroZonedBinarization::assign_zones() {
    // Sort dimensions by importance (descending)
    std::vector<std::pair<float, int>> sorted_dims(d);
    for (int i = 0; i < d; i++) {
        sorted_dims[i] = {dim_importance[i], i};
    }
    std::sort(sorted_dims.begin(), sorted_dims.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Calculate zone sizes
    int n_float = static_cast<int>(d * zone_config.float_ratio);
    int n_int8 = static_cast<int>(d * zone_config.int8_ratio);
    int n_binary = d - n_float - n_int8;

    // Ensure at least 1 dimension in each zone if d is large enough
    if (d >= 10) {
        n_float = std::max(1, n_float);
        n_int8 = std::max(1, n_int8);
        n_binary = d - n_float - n_int8;
    }

    // Assign zones
    zone_assignments.resize(d);
    float_dims.clear();
    int8_dims.clear();
    binary_dims.clear();

    for (int i = 0; i < d; i++) {
        int dim = sorted_dims[i].second;
        if (i < n_float) {
            zone_assignments[dim] = 0;  // float zone
            float_dims.push_back(dim);
        } else if (i < n_float + n_int8) {
            zone_assignments[dim] = 1;  // int8 zone
            int8_dims.push_back(dim);
        } else {
            zone_assignments[dim] = 2;  // binary zone
            binary_dims.push_back(dim);
        }
    }
}

void IndexNeuroZonedBinarization::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(d > 0, "dimension must be positive");
    FAISS_THROW_IF_NOT_MSG(n > 0, "need training data");

    // Analyze dimension importance
    analyze_importance(n, x);

    // Assign dimensions to zones
    assign_zones();

    // Compute thresholds for binary zone (median per dimension)
    thresholds.resize(binary_dims.size());
    for (size_t i = 0; i < binary_dims.size(); i++) {
        int dim = binary_dims[i];
        std::vector<float> values(n);
        for (idx_t j = 0; j < n; j++) {
            values[j] = x[j * d + dim];
        }
        std::nth_element(
                values.begin(), values.begin() + n / 2, values.end());
        thresholds[i] = values[n / 2];
    }

    // Compute scaling for int8 zone (min/max per dimension)
    int8_scales.resize(int8_dims.size());
    int8_mins.resize(int8_dims.size());
    for (size_t i = 0; i < int8_dims.size(); i++) {
        int dim = int8_dims[i];
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();
        for (idx_t j = 0; j < n; j++) {
            float val = x[j * d + dim];
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
        int8_mins[i] = min_val;
        float range = max_val - min_val;
        int8_scales[i] = (range > 1e-10f) ? 255.0f / range : 1.0f;
    }

    is_trained = true;
}

void IndexNeuroZonedBinarization::encode_vector(
        const float* x,
        float* out_float,
        int8_t* out_int8,
        uint8_t* out_binary) const {
    // Encode float zone
    for (size_t i = 0; i < float_dims.size(); i++) {
        out_float[i] = x[float_dims[i]];
    }

    // Encode int8 zone
    for (size_t i = 0; i < int8_dims.size(); i++) {
        float val = x[int8_dims[i]];
        float scaled = (val - int8_mins[i]) * int8_scales[i];
        scaled = std::max(0.0f, std::min(255.0f, scaled));
        out_int8[i] = static_cast<int8_t>(scaled - 128);
    }

    // Encode binary zone (bit packing)
    size_t n_bytes = (binary_dims.size() + 7) / 8;
    std::fill(out_binary, out_binary + n_bytes, 0);
    for (size_t i = 0; i < binary_dims.size(); i++) {
        if (x[binary_dims[i]] >= thresholds[i]) {
            out_binary[i / 8] |= (1 << (i % 8));
        }
    }
}

void IndexNeuroZonedBinarization::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained before adding");

    size_t n_float = float_dims.size();
    size_t n_int8 = int8_dims.size();
    size_t n_binary_bytes = (binary_dims.size() + 7) / 8;

    // Resize storage
    size_t old_n = ntotal;
    data_float.resize((old_n + n) * n_float);
    data_int8.resize((old_n + n) * n_int8);
    data_binary.resize((old_n + n) * n_binary_bytes);

    // Encode each vector
    for (idx_t i = 0; i < n; i++) {
        encode_vector(
                x + i * d,
                data_float.data() + (old_n + i) * n_float,
                data_int8.data() + (old_n + i) * n_int8,
                data_binary.data() + (old_n + i) * n_binary_bytes);
    }

    // Store original vectors for full-precision reranking
    size_t old_orig = orig_vectors.size();
    orig_vectors.resize(old_orig + n * d);
    std::copy(x, x + n * d, orig_vectors.data() + old_orig);

    // Add to inner index if present
    if (inner_index) {
        inner_index->add(n, x);
    }

    ntotal += n;
}

void IndexNeuroZonedBinarization::reset() {
    data_float.clear();
    data_int8.clear();
    data_binary.clear();
    orig_vectors.clear();
    if (inner_index) {
        inner_index->reset();
    }
    ntotal = 0;
}

int IndexNeuroZonedBinarization::hamming_distance(
        const uint8_t* a,
        const uint8_t* b,
        size_t nbytes) const {
    int dist = 0;
    for (size_t i = 0; i < nbytes; i++) {
        dist += __builtin_popcount(a[i] ^ b[i]);
    }
    return dist;
}

float IndexNeuroZonedBinarization::l1_distance_int8(
        const int8_t* a,
        const int8_t* b,
        size_t n) const {
    float dist = 0.0f;
    for (size_t i = 0; i < n; i++) {
        dist += std::abs(static_cast<float>(a[i]) - static_cast<float>(b[i]));
    }
    return dist;
}

float IndexNeuroZonedBinarization::l2_distance_float(
        const float* a,
        const float* b,
        size_t n) const {
    float dist = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

void IndexNeuroZonedBinarization::get_zone_sizes(
        int& n_float,
        int& n_int8,
        int& n_binary) const {
    n_float = static_cast<int>(float_dims.size());
    n_int8 = static_cast<int>(int8_dims.size());
    n_binary = static_cast<int>(binary_dims.size());
}

float IndexNeuroZonedBinarization::get_compression_ratio() const {
    if (d == 0) return 1.0f;

    size_t orig_bytes = d * sizeof(float);
    size_t compressed_bytes = float_dims.size() * sizeof(float) +
                              int8_dims.size() * sizeof(int8_t) +
                              (binary_dims.size() + 7) / 8;

    return static_cast<float>(orig_bytes) / compressed_bytes;
}

void IndexNeuroZonedBinarization::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained");

    // Resolve parameters
    int n_rerank = rerank_k;

    auto zp = dynamic_cast<const NeuroZonedParams*>(params);
    if (zp && zp->rerank_k > 0) {
        n_rerank = zp->rerank_k;
    }

    bool collect = false;
    if (params) {
        auto nsp = dynamic_cast<const NeuroSearchParameters*>(params);
        if (nsp) {
            collect = nsp->collect_stats;
        }
    }

    int64_t total_calcs = 0;

    size_t n_float_dims = float_dims.size();
    size_t n_int8_dims = int8_dims.size();
    size_t n_binary_bytes = (binary_dims.size() + 7) / 8;

#pragma omp parallel for reduction(+ : total_calcs)
    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * d;

        // Encode query to zoned format
        std::vector<float> q_float(n_float_dims);
        std::vector<int8_t> q_int8(n_int8_dims);
        std::vector<uint8_t> q_binary(n_binary_bytes);
        encode_vector(query, q_float.data(), q_int8.data(), q_binary.data());

        // Phase 1: Fast Hamming distance on binary zone
        std::vector<std::pair<float, idx_t>> scored(ntotal);
        for (idx_t i = 0; i < ntotal; i++) {
            int ham_dist = hamming_distance(
                    q_binary.data(),
                    data_binary.data() + i * n_binary_bytes,
                    n_binary_bytes);
            scored[i] = {static_cast<float>(ham_dist), i};
        }

        // Get top candidates based on Hamming distance
        size_t actual_rerank = std::min(
                static_cast<size_t>(n_rerank * 2),
                static_cast<size_t>(ntotal));
        std::partial_sort(
                scored.begin(), scored.begin() + actual_rerank, scored.end());

        // Phase 2: Refine with int8 distance
        for (size_t i = 0; i < actual_rerank; i++) {
            idx_t idx = scored[i].second;

            float int8_dist = l1_distance_int8(
                    q_int8.data(),
                    data_int8.data() + idx * n_int8_dims,
                    n_int8_dims);

            // Combined score
            scored[i].first = weight_binary * scored[i].first +
                              weight_int8 * int8_dist;
        }

        // Re-sort and get final candidates
        std::partial_sort(
                scored.begin(),
                scored.begin() + std::min(actual_rerank, static_cast<size_t>(n_rerank)),
                scored.begin() + actual_rerank);
        actual_rerank = std::min(actual_rerank, static_cast<size_t>(n_rerank));

        // Phase 3: Final rerank with float zone or full-precision
        if (do_full_rerank && !orig_vectors.empty()) {
            // Full precision reranking with original vectors
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
        } else {
            // Partial rerank with float zone only
            for (size_t i = 0; i < actual_rerank; i++) {
                idx_t idx = scored[i].second;

                float float_dist = l2_distance_float(
                        q_float.data(),
                        data_float.data() + idx * n_float_dims,
                        n_float_dims);

                // Final combined score
                scored[i].first = scored[i].first + weight_float * float_dist;
                total_calcs += n_float_dims + n_int8_dims;
            }
        }

        // Final sort
        std::partial_sort(
                scored.begin(),
                scored.begin() + std::min(actual_rerank, static_cast<size_t>(k)),
                scored.begin() + actual_rerank);

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

    if (collect) {
        last_stats.calculations_performed = total_calcs;
    }
}

} // namespace faiss
