/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroAdaptiveQuantization.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace faiss {

IndexNeuroAdaptiveQuantization::IndexNeuroAdaptiveQuantization(
        int d,
        int nregions,
        float hot_ratio)
        : IndexNeuro(nullptr, false),
          nregions(nregions),
          hot_ratio(hot_ratio) {
    this->d = d;
    is_hot.resize(nregions, false);
    query_counts.resize(nregions, 0);
    hot_vectors.resize(nregions);
    hot_ids.resize(nregions);
    cold_codes.resize(nregions);
    cold_ids.resize(nregions);
    is_trained = false;
}

int IndexNeuroAdaptiveQuantization::find_region(const float* vec) const {
    int best = 0;
    float best_dist = std::numeric_limits<float>::max();
    for (int r = 0; r < nregions; r++) {
        float dist = fvec_L2sqr(vec, centroids.data() + r * d, d);
        if (dist < best_dist) {
            best_dist = dist;
            best = r;
        }
    }
    return best;
}

void IndexNeuroAdaptiveQuantization::encode_4bit(const float* vec, uint8_t* code) const {
    size_t cs = code_size_4bit();
    std::fill(code, code + cs, 0);

    for (int i = 0; i < d; i++) {
        int q = static_cast<int>((vec[i] + 3.0f) / 0.375f);
        q = std::max(0, std::min(15, q));
        if (i % 2 == 0) {
            code[i / 2] |= q;
        } else {
            code[i / 2] |= (q << 4);
        }
    }
}

void IndexNeuroAdaptiveQuantization::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(n >= nregions, "need at least nregions vectors");

    // Cluster into regions
    Clustering clus(d, nregions);
    clus.verbose = false;
    clus.niter = 20;

    IndexFlatL2 quantizer(d);
    clus.train(n, x, quantizer);

    centroids.resize(nregions * d);
    std::copy(
            quantizer.get_xb(),
            quantizer.get_xb() + nregions * d,
            centroids.data());

    // Initially mark regions as hot based on density
    std::vector<int> region_counts(nregions, 0);
    for (idx_t i = 0; i < n; i++) {
        int region = find_region(x + i * d);
        region_counts[region]++;
    }

    // Sort regions by count and mark top hot_ratio as hot
    std::vector<std::pair<int, int>> sorted_regions(nregions);
    for (int r = 0; r < nregions; r++) {
        sorted_regions[r] = {region_counts[r], r};
    }
    std::sort(sorted_regions.rbegin(), sorted_regions.rend());

    int n_hot = static_cast<int>(nregions * hot_ratio);
    for (int i = 0; i < n_hot && i < nregions; i++) {
        is_hot[sorted_regions[i].second] = true;
    }

    is_trained = true;
}

void IndexNeuroAdaptiveQuantization::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained");

    for (idx_t i = 0; i < n; i++) {
        const float* vec = x + i * d;
        int region = find_region(vec);
        idx_t vec_id = ntotal + i;

        if (is_hot[region]) {
            // Store full precision
            size_t old_size = hot_vectors[region].size();
            hot_vectors[region].resize(old_size + d);
            std::copy(vec, vec + d, hot_vectors[region].data() + old_size);
            hot_ids[region].push_back(vec_id);
        } else {
            // Store 4-bit quantized
            size_t cs = code_size_4bit();
            size_t old_size = cold_codes[region].size();
            cold_codes[region].resize(old_size + cs);
            encode_4bit(vec, cold_codes[region].data() + old_size);
            cold_ids[region].push_back(vec_id);
        }
    }

    // Store original vectors for cold reranking
    size_t old_orig = orig_vectors.size();
    orig_vectors.resize(old_orig + n * d);
    std::copy(x, x + n * d, orig_vectors.data() + old_orig);

    ntotal += n;
}

void IndexNeuroAdaptiveQuantization::reset() {
    for (auto& v : hot_vectors) v.clear();
    for (auto& v : hot_ids) v.clear();
    for (auto& v : cold_codes) v.clear();
    for (auto& v : cold_ids) v.clear();
    orig_vectors.clear();
    std::fill(query_counts.begin(), query_counts.end(), 0);
    ntotal = 0;
}

void IndexNeuroAdaptiveQuantization::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT(key >= 0 && key < ntotal);
    std::copy(
            orig_vectors.data() + key * d,
            orig_vectors.data() + (key + 1) * d,
            recons);
}

void IndexNeuroAdaptiveQuantization::update_hot_regions() {
    // Re-assign hot/cold based on query counts
    std::vector<std::pair<int64_t, int>> sorted_regions(nregions);
    for (int r = 0; r < nregions; r++) {
        sorted_regions[r] = {query_counts[r], r};
    }
    std::sort(sorted_regions.rbegin(), sorted_regions.rend());

    int n_hot = static_cast<int>(nregions * hot_ratio);
    std::fill(is_hot.begin(), is_hot.end(), false);
    for (int i = 0; i < n_hot && i < nregions; i++) {
        is_hot[sorted_regions[i].second] = true;
    }

    // Note: In a real implementation, we'd need to re-encode vectors
    // when regions change between hot and cold
}

void IndexNeuroAdaptiveQuantization::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(ntotal > 0);

    int n_rerank = rerank_k;
    auto ap = dynamic_cast<const NeuroAdaptiveQuantParams*>(params);
    if (ap && ap->rerank_k > 0) {
        n_rerank = ap->rerank_k;
    }

#pragma omp parallel for
    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * d;

        // Find query's region and update counts
        int query_region = find_region(query);
        query_counts[query_region]++;

        std::vector<std::pair<float, idx_t>> candidates;

        // Search all regions
        for (int r = 0; r < nregions; r++) {
            if (is_hot[r]) {
                // Hot region: exact search
                size_t nvecs = hot_ids[r].size();
                for (size_t i = 0; i < nvecs; i++) {
                    const float* vec = hot_vectors[r].data() + i * d;
                    float dist = fvec_L2sqr(query, vec, d);
                    candidates.push_back({dist, hot_ids[r][i]});
                }
            } else {
                // Cold region: approximate then rerank
                size_t cs = code_size_4bit();
                std::vector<uint8_t> query_code(cs);
                encode_4bit(query, query_code.data());

                size_t nvecs = cold_ids[r].size();
                std::vector<std::pair<int, idx_t>> approx_scores;
                approx_scores.reserve(nvecs);

                for (size_t i = 0; i < nvecs; i++) {
                    const uint8_t* code = cold_codes[r].data() + i * cs;
                    int approx_dist = 0;
                    for (size_t j = 0; j < cs; j++) {
                        int dq = query_code[j] & 0x0F;
                        int dc = code[j] & 0x0F;
                        approx_dist += (dq - dc) * (dq - dc);
                        dq = query_code[j] >> 4;
                        dc = code[j] >> 4;
                        approx_dist += (dq - dc) * (dq - dc);
                    }
                    approx_scores.push_back({approx_dist, cold_ids[r][i]});
                }

                // Keep top candidates for reranking
                size_t keep = std::min(static_cast<size_t>(n_rerank), approx_scores.size());
                std::partial_sort(
                        approx_scores.begin(),
                        approx_scores.begin() + keep,
                        approx_scores.end());

                for (size_t i = 0; i < keep; i++) {
                    idx_t idx = approx_scores[i].second;
                    const float* vec = orig_vectors.data() + idx * d;
                    float dist = fvec_L2sqr(query, vec, d);
                    candidates.push_back({dist, idx});
                }
            }
        }

        // Sort all candidates and output top-k
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
