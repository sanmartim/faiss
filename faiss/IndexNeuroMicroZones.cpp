/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroMicroZones.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace faiss {

IndexNeuroMicroZones::IndexNeuroMicroZones(int d, int n_zones)
        : IndexNeuro(nullptr, false), n_zones(n_zones) {
    this->d = d;

    // Compute bits per zone
    bits_per_zone = 0;
    int temp = n_zones - 1;
    while (temp > 0) {
        bits_per_zone++;
        temp >>= 1;
    }
    if (bits_per_zone == 0) {
        bits_per_zone = 1;
    }

    // Bytes per vector
    bytes_per_vec = (d * bits_per_zone + 7) / 8;

    // Initialize zone boundaries for uniform zones in [-1, 1]
    zone_boundaries.resize(n_zones + 1);
    for (int i = 0; i <= n_zones; i++) {
        zone_boundaries[i] = -1.0f + 2.0f * i / n_zones;
    }

    zone_codes.resize(scales.size());
    is_trained = false;
}

int IndexNeuroMicroZones::get_zone(float val) const {
    // Binary search for zone
    for (int z = 0; z < n_zones; z++) {
        if (val < zone_boundaries[z + 1]) {
            return z;
        }
    }
    return n_zones - 1;
}

void IndexNeuroMicroZones::compute_zone_code(
        const float* vec,
        float scale,
        uint8_t* code) const {
    // Initialize code to zeros
    std::fill(code, code + bytes_per_vec, 0);

    for (int j = 0; j < d; j++) {
        // Use scaled value directly (not tanh) for better discrimination
        // The zone_boundaries are learned from data distribution
        float val = vec[j] * scale;

        // Clamp to boundary range
        val = std::max(zone_boundaries[0], std::min(zone_boundaries[n_zones], val));

        int zone = get_zone(val);

        // Pack zone into code (2 bits per dimension for 4 zones)
        int bit_pos = j * bits_per_zone;
        int byte_idx = bit_pos / 8;
        int bit_offset = bit_pos % 8;

        // Handle zone value that may span bytes
        if (bit_offset + bits_per_zone <= 8) {
            code[byte_idx] |= (zone << bit_offset);
        } else {
            // Spans two bytes
            code[byte_idx] |= (zone << bit_offset);
            code[byte_idx + 1] |= (zone >> (8 - bit_offset));
        }
    }
}

int IndexNeuroMicroZones::zone_distance(
        const uint8_t* code1,
        const uint8_t* code2) const {
    int total_dist = 0;

    for (int j = 0; j < d; j++) {
        int bit_pos = j * bits_per_zone;
        int byte_idx = bit_pos / 8;
        int bit_offset = bit_pos % 8;

        int zone1, zone2;
        int mask = (1 << bits_per_zone) - 1;

        if (bit_offset + bits_per_zone <= 8) {
            zone1 = (code1[byte_idx] >> bit_offset) & mask;
            zone2 = (code2[byte_idx] >> bit_offset) & mask;
        } else {
            // Spans two bytes
            zone1 = ((code1[byte_idx] >> bit_offset) |
                     (code1[byte_idx + 1] << (8 - bit_offset))) &
                    mask;
            zone2 = ((code2[byte_idx] >> bit_offset) |
                     (code2[byte_idx + 1] << (8 - bit_offset))) &
                    mask;
        }

        // Zone distance is absolute difference
        total_dist += std::abs(zone1 - zone2);
    }

    return total_dist;
}

void IndexNeuroMicroZones::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(n > 0);

    // Compute adaptive zone boundaries based on data distribution
    // Use raw scaled values (not tanh) for better discrimination
    std::vector<float> all_values;
    all_values.reserve(n * d);

    // Use middle scale for boundary computation
    float mid_scale = scales[scales.size() / 2];

    for (idx_t i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            float val = x[i * d + j] * mid_scale;
            all_values.push_back(val);
        }
    }

    // Sort and find quantile boundaries
    std::sort(all_values.begin(), all_values.end());

    zone_boundaries.resize(n_zones + 1);
    // Use actual data range for boundaries
    zone_boundaries[0] = all_values.front() - 0.001f;
    zone_boundaries[n_zones] = all_values.back() + 0.001f;

    for (int z = 1; z < n_zones; z++) {
        size_t idx = (all_values.size() * z) / n_zones;
        zone_boundaries[z] = all_values[idx];
    }

    // Clear existing data - train only learns boundaries
    zone_codes.resize(scales.size());
    for (auto& codes : zone_codes) {
        codes.clear();
    }
    vectors.clear();
    ntotal = 0;

    is_trained = true;
}

void IndexNeuroMicroZones::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "Index must be trained first");

    // Compute zone codes for new vectors
    std::vector<uint8_t> code(bytes_per_vec);
    for (idx_t i = 0; i < n; i++) {
        const float* vec = x + i * d;
        for (size_t s = 0; s < scales.size(); s++) {
            compute_zone_code(vec, scales[s], code.data());
            zone_codes[s].insert(zone_codes[s].end(), code.begin(), code.end());
        }
    }

    // Store original vectors
    size_t old_size = vectors.size();
    vectors.resize(old_size + n * d);
    std::copy(x, x + n * d, vectors.data() + old_size);

    ntotal += n;
}

void IndexNeuroMicroZones::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(ntotal > 0);

    // Parse parameters
    int max_zdist = max_zone_distance;
    float kr = keep_ratio;

    auto* mzp = dynamic_cast<const NeuroMicroZonesParams*>(params);
    if (mzp) {
        if (mzp->max_zone_distance > 0) {
            max_zdist = mzp->max_zone_distance;
        }
        kr = mzp->keep_ratio;
    }

    // Auto-compute max zone distance if not set
    // Maximum zone dist per dim is (n_zones - 1), total is d * (n_zones - 1)
    // Use more permissive threshold for better recall:
    // Allow up to 75% of max possible distance
    if (max_zdist <= 0) {
        max_zdist = d * (n_zones - 1) * 3 / 4;
    }

    bool collect = false;
    if (params) {
        auto* nsp = dynamic_cast<const NeuroSearchParameters*>(params);
        if (nsp) {
            collect = nsp->collect_stats;
        }
    }

    int64_t total_calcs = 0;

#pragma omp parallel for reduction(+ : total_calcs)
    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * d;

        // Compute query zone codes for all scales
        std::vector<std::vector<uint8_t>> query_codes(scales.size());
        for (size_t s = 0; s < scales.size(); s++) {
            query_codes[s].resize(bytes_per_vec);
            compute_zone_code(query, scales[s], query_codes[s].data());
        }

        // Stage 1: Filter by zone distance (sum across all scales)
        std::vector<std::pair<int, idx_t>> candidates;
        candidates.reserve(ntotal);

        for (idx_t i = 0; i < ntotal; i++) {
            int total_zdist = 0;
            bool passes = true;

            for (size_t s = 0; s < scales.size(); s++) {
                const uint8_t* vec_code =
                        zone_codes[s].data() + i * bytes_per_vec;
                int zdist = zone_distance(query_codes[s].data(), vec_code);
                total_zdist += zdist;

                // Early exit if already over threshold
                if (total_zdist > max_zdist * static_cast<int>(scales.size())) {
                    passes = false;
                    break;
                }
            }

            if (passes) {
                candidates.push_back({total_zdist, i});
            }
        }

        // If no candidates pass threshold, take best ones by zone distance
        if (candidates.empty()) {
            for (idx_t i = 0; i < ntotal; i++) {
                int total_zdist = 0;
                for (size_t s = 0; s < scales.size(); s++) {
                    const uint8_t* vec_code =
                            zone_codes[s].data() + i * bytes_per_vec;
                    total_zdist += zone_distance(query_codes[s].data(), vec_code);
                }
                candidates.push_back({total_zdist, i});
            }
        }

        // Keep top candidates by zone distance
        idx_t n_keep = std::max(
                static_cast<idx_t>(ntotal * kr),
                std::min(k * 4, ntotal));

        if (candidates.size() > static_cast<size_t>(n_keep)) {
            std::partial_sort(
                    candidates.begin(),
                    candidates.begin() + n_keep,
                    candidates.end());
            candidates.resize(n_keep);
        }

        // Stage 2: Precise L2 on candidates
        std::vector<std::pair<float, idx_t>> l2_dists;
        l2_dists.reserve(candidates.size());

        for (const auto& cand : candidates) {
            idx_t idx = cand.second;
            float dist = fvec_L2sqr(query, vectors.data() + idx * d, d);
            l2_dists.push_back({dist, idx});
        }
        total_calcs += candidates.size() * d;

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

    if (collect) {
        last_stats.calculations_performed = total_calcs;
        last_stats.columns_used = d;
    }
}

void IndexNeuroMicroZones::reset() {
    for (auto& codes : zone_codes) {
        codes.clear();
    }
    vectors.clear();
    ntotal = 0;
}

} // namespace faiss
