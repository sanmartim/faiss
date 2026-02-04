/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroMultiZoneSign.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace faiss {

IndexNeuroMultiZoneSign::IndexNeuroMultiZoneSign(
        int d,
        int n_scales,
        TrigFunction trig_func,
        int bits_per_zone)
        : IndexNeuro(nullptr, false),
          n_scales(n_scales),
          bits_per_zone(bits_per_zone),
          trig_func(trig_func) {
    this->d = d;

    n_zones = 1 << bits_per_zone;  // 4 for 2 bits, 8 for 3 bits
    bytes_per_vec = (d * bits_per_zone + 7) / 8;

    // Generate logarithmically spaced scales
    scales.resize(n_scales);
    for (int i = 0; i < n_scales; i++) {
        float t = static_cast<float>(i) / std::max(1, n_scales - 1);
        scales[i] = scale_min * std::pow(scale_max / scale_min, t);
    }

    zone_codes.resize(n_scales);
    is_trained = false;
}

float IndexNeuroMultiZoneSign::apply_trig(float x, float scale) const {
    float v = x * scale;

    switch (trig_func) {
        case TrigFunction::TANH:
            return std::tanh(v);

        case TrigFunction::SIGMOID:
            return 2.0f / (1.0f + std::exp(-v)) - 1.0f;  // Map to [-1, 1]

        case TrigFunction::SIN:
            return std::sin(v);

        case TrigFunction::ATAN:
            return std::atan(v) / (M_PI / 2.0f);  // Map to [-1, 1]

        case TrigFunction::ERF:
            return std::erf(v);

        case TrigFunction::SOFTSIGN:
            return v / (1.0f + std::abs(v));

        default:
            return std::tanh(v);
    }
}

void IndexNeuroMultiZoneSign::compute_zone_code(
        const float* vec,
        int scale_idx,
        uint8_t* code) const {
    std::fill(code, code + bytes_per_vec, 0);

    float scale = scales[scale_idx];
    int boundary_offset = scale_idx * d * (n_zones + 1);

    for (int j = 0; j < d; j++) {
        float val = apply_trig(vec[j], scale);

        // Find zone using learned boundaries
        int zone = 0;
        int bound_start = boundary_offset + j * (n_zones + 1);
        for (int z = 0; z < n_zones; z++) {
            if (val >= zone_boundaries[bound_start + z + 1]) {
                zone = z + 1;
            } else {
                break;
            }
        }
        zone = std::min(zone, n_zones - 1);

        // Pack zone into code
        int bit_pos = j * bits_per_zone;
        int byte_idx = bit_pos / 8;
        int bit_offset = bit_pos % 8;

        if (bit_offset + bits_per_zone <= 8) {
            code[byte_idx] |= (zone << bit_offset);
        } else {
            code[byte_idx] |= (zone << bit_offset);
            code[byte_idx + 1] |= (zone >> (8 - bit_offset));
        }
    }
}

int IndexNeuroMultiZoneSign::zone_distance(
        const uint8_t* code1,
        const uint8_t* code2) const {
    int total_dist = 0;
    int mask = (1 << bits_per_zone) - 1;

    for (int j = 0; j < d; j++) {
        int bit_pos = j * bits_per_zone;
        int byte_idx = bit_pos / 8;
        int bit_offset = bit_pos % 8;

        int zone1, zone2;

        if (bit_offset + bits_per_zone <= 8) {
            zone1 = (code1[byte_idx] >> bit_offset) & mask;
            zone2 = (code2[byte_idx] >> bit_offset) & mask;
        } else {
            zone1 = ((code1[byte_idx] >> bit_offset) |
                     (code1[byte_idx + 1] << (8 - bit_offset))) &
                    mask;
            zone2 = ((code2[byte_idx] >> bit_offset) |
                     (code2[byte_idx + 1] << (8 - bit_offset))) &
                    mask;
        }

        total_dist += std::abs(zone1 - zone2);
    }

    return total_dist;
}

void IndexNeuroMultiZoneSign::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(n > 0);

    // Learn zone boundaries for each scale and dimension
    // Use quantiles of transformed values
    zone_boundaries.resize(n_scales * d * (n_zones + 1));

    std::vector<float> transformed(n);

    for (int s = 0; s < n_scales; s++) {
        float scale = scales[s];

        for (int j = 0; j < d; j++) {
            // Collect transformed values for this dimension
            for (idx_t i = 0; i < n; i++) {
                transformed[i] = apply_trig(x[i * d + j], scale);
            }

            // Sort to find quantiles
            std::sort(transformed.begin(), transformed.end());

            // Set zone boundaries
            int bound_start = s * d * (n_zones + 1) + j * (n_zones + 1);
            zone_boundaries[bound_start] = transformed.front() - 0.001f;
            zone_boundaries[bound_start + n_zones] = transformed.back() + 0.001f;

            for (int z = 1; z < n_zones; z++) {
                size_t idx = (n * z) / n_zones;
                zone_boundaries[bound_start + z] = transformed[idx];
            }
        }
    }

    // Clear codes - train only learns boundaries
    for (auto& codes : zone_codes) {
        codes.clear();
    }
    vectors.clear();
    ntotal = 0;

    is_trained = true;
}

void IndexNeuroMultiZoneSign::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "Index must be trained first");

    // Compute zone codes for all scales
    std::vector<uint8_t> code(bytes_per_vec);
    for (idx_t i = 0; i < n; i++) {
        const float* vec = x + i * d;
        for (int s = 0; s < n_scales; s++) {
            compute_zone_code(vec, s, code.data());
            zone_codes[s].insert(zone_codes[s].end(), code.begin(), code.end());
        }
    }

    // Store original vectors
    size_t old_size = vectors.size();
    vectors.resize(old_size + n * d);
    std::copy(x, x + n * d, vectors.data() + old_size);

    ntotal += n;
}

void IndexNeuroMultiZoneSign::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(ntotal > 0);

    float kr = keep_ratio;
    auto* mzsp = dynamic_cast<const NeuroMultiZoneSignParams*>(params);
    if (mzsp) {
        kr = mzsp->keep_ratio;
    }

    // Use subset of scales for efficiency (every 10th scale if many)
    int scale_step = std::max(1, n_scales / 10);
    std::vector<int> active_scales;
    for (int s = 0; s < n_scales; s += scale_step) {
        active_scales.push_back(s);
    }

#pragma omp parallel for
    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * d;

        // Compute query zone codes for active scales
        std::vector<std::vector<uint8_t>> query_codes(active_scales.size());
        for (size_t si = 0; si < active_scales.size(); si++) {
            query_codes[si].resize(bytes_per_vec);
            compute_zone_code(query, active_scales[si], query_codes[si].data());
        }

        // Compute total zone distance across active scales
        std::vector<std::pair<int, idx_t>> candidates;
        candidates.reserve(ntotal);

        for (idx_t i = 0; i < ntotal; i++) {
            int total_zdist = 0;

            for (size_t si = 0; si < active_scales.size(); si++) {
                int s = active_scales[si];
                const uint8_t* vec_code =
                        zone_codes[s].data() + i * bytes_per_vec;
                total_zdist += zone_distance(query_codes[si].data(), vec_code);
            }

            candidates.push_back({total_zdist, i});
        }

        // Keep top candidates by zone distance
        idx_t n_keep = std::max(
                static_cast<idx_t>(ntotal * kr), std::min(k * 4, ntotal));

        std::partial_sort(
                candidates.begin(),
                candidates.begin() + n_keep,
                candidates.end());
        candidates.resize(n_keep);

        // Rerank with precise L2
        std::vector<std::pair<float, idx_t>> l2_dists;
        l2_dists.reserve(candidates.size());

        for (const auto& cand : candidates) {
            idx_t idx = cand.second;
            float dist = fvec_L2sqr(query, vectors.data() + idx * d, d);
            l2_dists.push_back({dist, idx});
        }

        // Sort and output
        size_t actual_k = std::min(static_cast<size_t>(k), l2_dists.size());
        std::partial_sort(
                l2_dists.begin(), l2_dists.begin() + actual_k, l2_dists.end());

        for (size_t i = 0; i < actual_k; i++) {
            distances[q * k + i] = l2_dists[i].first;
            labels[q * k + i] = l2_dists[i].second;
        }
        for (size_t i = actual_k; i < static_cast<size_t>(k); i++) {
            distances[q * k + i] = std::numeric_limits<float>::max();
            labels[q * k + i] = -1;
        }
    }
}

void IndexNeuroMultiZoneSign::reset() {
    for (auto& codes : zone_codes) {
        codes.clear();
    }
    vectors.clear();
    ntotal = 0;
}

} // namespace faiss
