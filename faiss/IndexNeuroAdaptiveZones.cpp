/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroAdaptiveZones.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace faiss {

IndexNeuroAdaptiveZones::IndexNeuroAdaptiveZones(int d, int nregions)
        : IndexNeuro(nullptr, false), nregions(nregions) {
    this->d = d;
    region_bits.resize(nregions, 8);  // Default 8 bits
    region_codes.resize(nregions);
    region_vectors.resize(nregions);
    is_trained = false;
}

void IndexNeuroAdaptiveZones::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(n >= nregions, "need at least nregions vectors to train");

    // Cluster vectors into regions
    Clustering clus(d, nregions);
    clus.verbose = false;
    clus.niter = 20;

    IndexFlatL2 quantizer(d);
    clus.train(n, x, quantizer);

    // Store region centroids
    region_centroids.resize(nregions * d);
    std::copy(
            quantizer.get_xb(),
            quantizer.get_xb() + nregions * d,
            region_centroids.data());

    // Compute region densities (count per region)
    std::vector<int> region_counts(nregions, 0);
    for (idx_t i = 0; i < n; i++) {
        int region = find_region(x + i * d);
        region_counts[region]++;
    }

    // Assign bits based on density
    // Dense regions get more bits (higher precision)
    float avg_count = static_cast<float>(n) / nregions;
    for (int r = 0; r < nregions; r++) {
        float density_ratio = region_counts[r] / avg_count;
        if (density_ratio > 1.5f) {
            region_bits[r] = 8;  // High density: 8-bit
        } else if (density_ratio > 0.8f) {
            region_bits[r] = 4;  // Medium density: 4-bit
        } else {
            region_bits[r] = 2;  // Low density: 2-bit
        }
    }

    is_trained = true;
}

int IndexNeuroAdaptiveZones::find_region(const float* vec) const {
    int best_region = 0;
    float best_dist = std::numeric_limits<float>::max();

    for (int r = 0; r < nregions; r++) {
        float dist = fvec_L2sqr(vec, region_centroids.data() + r * d, d);
        if (dist < best_dist) {
            best_dist = dist;
            best_region = r;
        }
    }
    return best_region;
}

size_t IndexNeuroAdaptiveZones::code_size_for_region(int region) const {
    int bits = region_bits[region];
    return (d * bits + 7) / 8;
}

void IndexNeuroAdaptiveZones::encode_for_region(
        const float* vec,
        int region,
        uint8_t* code) const {
    int bits = region_bits[region];
    size_t code_sz = code_size_for_region(region);
    std::fill(code, code + code_sz, 0);

    if (bits == 2) {
        for (int i = 0; i < d; i++) {
            int q = static_cast<int>((vec[i] + 3.0f) / 1.5f);
            q = std::max(0, std::min(3, q));
            int byte_idx = (i * 2) / 8;
            int bit_offset = (i * 2) % 8;
            code[byte_idx] |= (q << bit_offset);
        }
    } else if (bits == 4) {
        for (int i = 0; i < d; i++) {
            int q = static_cast<int>((vec[i] + 3.0f) / 0.375f);
            q = std::max(0, std::min(15, q));
            int byte_idx = (i * 4) / 8;
            int bit_offset = (i * 4) % 8;
            code[byte_idx] |= (q << bit_offset);
        }
    } else {  // 8 bits
        for (int i = 0; i < d; i++) {
            int q = static_cast<int>((vec[i] + 4.0f) / 8.0f * 255);
            q = std::max(0, std::min(255, q));
            code[i] = static_cast<uint8_t>(q);
        }
    }
}

void IndexNeuroAdaptiveZones::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained");

    for (idx_t i = 0; i < n; i++) {
        const float* vec = x + i * d;
        int region = find_region(vec);

        // Store vector ID in region
        idx_t vec_id = ntotal + i;
        region_vectors[region].push_back(vec_id);
        region_assignments.push_back(region);

        // Encode for region
        size_t code_sz = code_size_for_region(region);
        size_t old_size = region_codes[region].size();
        region_codes[region].resize(old_size + code_sz);
        encode_for_region(vec, region, region_codes[region].data() + old_size);
    }

    // Store original vectors
    size_t old_orig = orig_vectors.size();
    orig_vectors.resize(old_orig + n * d);
    std::copy(x, x + n * d, orig_vectors.data() + old_orig);

    ntotal += n;
}

void IndexNeuroAdaptiveZones::reset() {
    for (auto& codes : region_codes) {
        codes.clear();
    }
    for (auto& vecs : region_vectors) {
        vecs.clear();
    }
    region_assignments.clear();
    orig_vectors.clear();
    ntotal = 0;
}

void IndexNeuroAdaptiveZones::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT(key >= 0 && key < ntotal);
    std::copy(
            orig_vectors.data() + key * d,
            orig_vectors.data() + (key + 1) * d,
            recons);
}

float IndexNeuroAdaptiveZones::approx_distance(const float* query, idx_t idx) const {
    // Use full precision for now
    const float* vec = orig_vectors.data() + idx * d;
    return fvec_L2sqr(query, vec, d);
}

void IndexNeuroAdaptiveZones::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(ntotal > 0);

    // Resolve parameters
    int np = nprobe;
    int n_rerank = rerank_k;
    auto ap = dynamic_cast<const NeuroAdaptiveZonesParams*>(params);
    if (ap) {
        if (ap->nprobe > 0) np = ap->nprobe;
        if (ap->rerank_k > 0) n_rerank = ap->rerank_k;
    }
    np = std::min(np, nregions);

#pragma omp parallel for
    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * d;

        // Find nearest regions
        std::vector<std::pair<float, int>> region_dists(nregions);
        for (int r = 0; r < nregions; r++) {
            float dist = fvec_L2sqr(query, region_centroids.data() + r * d, d);
            region_dists[r] = {dist, r};
        }
        std::partial_sort(
                region_dists.begin(),
                region_dists.begin() + np,
                region_dists.end());

        // Collect candidates from probed regions
        std::vector<std::pair<float, idx_t>> candidates;
        for (int i = 0; i < np; i++) {
            int region = region_dists[i].second;
            for (idx_t vec_id : region_vectors[region]) {
                float dist = approx_distance(query, vec_id);
                candidates.push_back({dist, vec_id});
            }
        }

        // Sort and output top-k
        size_t actual_k = std::min(static_cast<size_t>(k), candidates.size());
        std::partial_sort(
                candidates.begin(),
                candidates.begin() + actual_k,
                candidates.end());

        for (size_t i = 0; i < actual_k; i++) {
            distances[q * k + i] = candidates[i].first;
            labels[q * k + i] = candidates[i].second;
        }
        for (size_t i = actual_k; i < static_cast<size_t>(k); i++) {
            distances[q * k + i] = std::numeric_limits<float>::max();
            labels[q * k + i] = -1;
        }
    }
}

} // namespace faiss
