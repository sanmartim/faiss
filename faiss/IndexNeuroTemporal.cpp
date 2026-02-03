/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroTemporal.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>

namespace faiss {

IndexNeuroTemporal::IndexNeuroTemporal(Index* inner, int n_bases, bool own_inner)
        : IndexNeuro(inner, own_inner), n_bases(n_bases) {}

void IndexNeuroTemporal::train(idx_t /*n*/, const float* /*x*/) {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");
    FAISS_THROW_IF_NOT_MSG(d > 0, "dimension must be positive");

    // Generate temporal basis functions
    // b_k(t) = exp(-t / tau_k) * sin(omega_k * t)
    basis_functions.resize(n_bases * max_sequence_length);

    for (int k = 0; k < n_bases; k++) {
        // Logarithmically spaced tau values
        float t_frac = static_cast<float>(k) / std::max(1, n_bases - 1);
        float tau = tau_min * std::pow(tau_max / tau_min, t_frac);

        // Frequency increases with shorter tau
        float omega = M_PI / (tau * 2.0f);

        for (int t = 0; t < max_sequence_length; t++) {
            float tf = static_cast<float>(t) / max_sequence_length;
            float decay = std::exp(-tf / tau);
            float oscillation = std::sin(omega * tf);
            basis_functions[k * max_sequence_length + t] = decay * oscillation;
        }

        // Normalize basis function
        float norm = 0.0f;
        for (int t = 0; t < max_sequence_length; t++) {
            float v = basis_functions[k * max_sequence_length + t];
            norm += v * v;
        }
        norm = std::sqrt(norm);
        if (norm > 1e-10f) {
            for (int t = 0; t < max_sequence_length; t++) {
                basis_functions[k * max_sequence_length + t] /= norm;
            }
        }
    }

    encoded_data.clear();
    sequence_lengths.clear();
    is_trained = true;
}

float IndexNeuroTemporal::basis_value(int basis_idx, int t) const {
    if (t >= max_sequence_length) {
        t = max_sequence_length - 1;
    }
    return basis_functions[basis_idx * max_sequence_length + t];
}

void IndexNeuroTemporal::encode_sequence(
        int seq_length,
        const float* seq,
        float* encoded) const {
    // Encoded dimension: n_bases * d
    std::fill(encoded, encoded + n_bases * d, 0.0f);

    // Scale time indices to fit basis function range
    float time_scale = static_cast<float>(max_sequence_length - 1) /
                       std::max(1, seq_length - 1);

    for (int k = 0; k < n_bases; k++) {
        for (int t = 0; t < seq_length; t++) {
            int t_scaled = static_cast<int>(t * time_scale);
            t_scaled = std::min(t_scaled, max_sequence_length - 1);
            float b = basis_functions[k * max_sequence_length + t_scaled];

            for (int j = 0; j < d; j++) {
                encoded[k * d + j] += seq[t * d + j] * b;
            }
        }
    }
}

void IndexNeuroTemporal::add_sequence(int seq_length, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained before adding");
    FAISS_THROW_IF_NOT_MSG(seq_length > 0, "sequence must have positive length");

    // Encode the sequence
    std::vector<float> encoded(n_bases * d);
    encode_sequence(seq_length, x, encoded.data());

    // Store encoded representation
    size_t old_size = encoded_data.size();
    encoded_data.resize(old_size + n_bases * d);
    std::copy(encoded.begin(), encoded.end(), encoded_data.begin() + old_size);

    sequence_lengths.push_back(seq_length);

    // Also add flattened sequence to inner index for reconstruction
    // Note: inner index stores original data, we store encoded separately
    ntotal++;
}

void IndexNeuroTemporal::reset() {
    IndexNeuro::reset();
    encoded_data.clear();
    sequence_lengths.clear();
}

void IndexNeuroTemporal::search_sequence(
        int seq_length,
        const float* query,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained");

    idx_t nb = static_cast<idx_t>(sequence_lengths.size());
    if (nb == 0) {
        for (idx_t i = 0; i < k; i++) {
            distances[i] = std::numeric_limits<float>::max();
            labels[i] = -1;
        }
        return;
    }

    // Encode query sequence
    std::vector<float> query_encoded(n_bases * d);
    encode_sequence(seq_length, query, query_encoded.data());

    // Compute distances to all encoded sequences
    std::vector<std::pair<float, idx_t>> scored(nb);
    int encoded_dim = n_bases * d;

    for (idx_t i = 0; i < nb; i++) {
        const float* data_encoded = encoded_data.data() + i * encoded_dim;

        // L2 distance in encoded space
        float dist = 0.0f;
        for (int j = 0; j < encoded_dim; j++) {
            float diff = query_encoded[j] - data_encoded[j];
            dist += diff * diff;
        }
        scored[i] = {dist, i};
    }

    // Sort and output
    size_t actual_k = std::min(static_cast<size_t>(k), static_cast<size_t>(nb));
    std::partial_sort(
            scored.begin(),
            scored.begin() + actual_k,
            scored.end());

    for (size_t i = 0; i < actual_k; i++) {
        distances[i] = scored[i].first;
        labels[i] = scored[i].second;
    }
    for (size_t i = actual_k; i < static_cast<size_t>(k); i++) {
        distances[i] = std::numeric_limits<float>::max();
        labels[i] = -1;
    }
}

void IndexNeuroTemporal::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    // Default behavior: treat input as single-timestep sequences
    // Use params to specify actual sequence length

    int seq_len = 1;
    auto tp = dynamic_cast<const NeuroTemporalParams*>(params);
    if (tp && tp->sequence_length > 0) {
        seq_len = tp->sequence_length;
    }

    // Each query is a sequence of seq_len timesteps
    idx_t n_queries = n / seq_len;
    if (n_queries == 0) {
        n_queries = 1;
        seq_len = static_cast<int>(n);
    }

#pragma omp parallel for
    for (idx_t q = 0; q < n_queries; q++) {
        search_sequence(
                seq_len,
                x + q * seq_len * d,
                k,
                distances + q * k,
                labels + q * k);
    }
}

} // namespace faiss
