/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroLearnedBinarization.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace faiss {

IndexNeuroLearnedBinarization::IndexNeuroLearnedBinarization(int d, int rerank_k)
        : IndexNeuro(nullptr, false), rerank_k(rerank_k) {
    this->d = d;
    thresholds.resize(d, 0.0f);  // Default: threshold at 0
    is_trained = false;
}

void IndexNeuroLearnedBinarization::encode(const float* vec, uint8_t* code) const {
    size_t cs = code_size();
    std::fill(code, code + cs, 0);

    for (int i = 0; i < d; i++) {
        if (vec[i] > thresholds[i]) {
            code[i / 8] |= (1 << (i % 8));
        }
    }
}

int IndexNeuroLearnedBinarization::hamming_distance(
        const uint8_t* a,
        const uint8_t* b) const {
    size_t cs = code_size();
    int dist = 0;
    for (size_t i = 0; i < cs; i++) {
        dist += __builtin_popcount(a[i] ^ b[i]);
    }
    return dist;
}

void IndexNeuroLearnedBinarization::learn_thresholds(idx_t n, const float* x) {
    // Initialize thresholds with median values
    for (int dim = 0; dim < d; dim++) {
        std::vector<float> values(n);
        for (idx_t i = 0; i < n; i++) {
            values[i] = x[i * d + dim];
        }
        std::sort(values.begin(), values.end());
        thresholds[dim] = values[n / 2];
    }

    // Sample pairs for optimization
    int n_pairs = std::min(static_cast<idx_t>(10000), n * (n - 1) / 2);
    RandomGenerator rng(1234);

    // Simple gradient-based optimization
    for (int iter = 0; iter < n_iter; iter++) {
        std::vector<float> gradient(d, 0.0f);

        for (int p = 0; p < n_pairs; p++) {
            idx_t i = rng.rand_int(n);
            idx_t j = rng.rand_int(n);
            if (i == j) continue;

            const float* vi = x + i * d;
            const float* vj = x + j * d;

            // Compute true distance
            float true_dist = fvec_L2sqr(vi, vj, d);

            // Compute Hamming distance with current thresholds
            int hamming = 0;
            for (int dim = 0; dim < d; dim++) {
                bool bi = vi[dim] > thresholds[dim];
                bool bj = vj[dim] > thresholds[dim];
                if (bi != bj) hamming++;
            }

            // Target: small true_dist should have small Hamming
            // Gradient: adjust thresholds to reduce correlation error
            float target_hamming = true_dist / (true_dist + 100.0f) * d;
            float error = hamming - target_hamming;

            // Update gradient
            for (int dim = 0; dim < d; dim++) {
                bool bi = vi[dim] > thresholds[dim];
                bool bj = vj[dim] > thresholds[dim];
                if (bi != bj) {
                    // This dimension contributes to Hamming
                    // Move threshold to potentially flip one bit
                    float mid = (vi[dim] + vj[dim]) / 2;
                    gradient[dim] += error * (thresholds[dim] - mid) * 0.001f;
                }
            }
        }

        // Apply gradient update
        for (int dim = 0; dim < d; dim++) {
            thresholds[dim] -= learning_rate * gradient[dim] / n_pairs;
        }
    }
}

void IndexNeuroLearnedBinarization::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(n > 0, "need training data");

    learn_thresholds(n, x);
    is_trained = true;
}

void IndexNeuroLearnedBinarization::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained");

    size_t cs = code_size();
    size_t old_size = codes.size();
    codes.resize(old_size + n * cs);

    for (idx_t i = 0; i < n; i++) {
        encode(x + i * d, codes.data() + old_size + i * cs);
    }

    // Store original vectors
    size_t old_orig = orig_vectors.size();
    orig_vectors.resize(old_orig + n * d);
    std::copy(x, x + n * d, orig_vectors.data() + old_orig);

    ntotal += n;
}

void IndexNeuroLearnedBinarization::reset() {
    codes.clear();
    orig_vectors.clear();
    ntotal = 0;
}

void IndexNeuroLearnedBinarization::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT(key >= 0 && key < ntotal);
    std::copy(
            orig_vectors.data() + key * d,
            orig_vectors.data() + (key + 1) * d,
            recons);
}

void IndexNeuroLearnedBinarization::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(ntotal > 0);

    int n_rerank = rerank_k;
    auto bp = dynamic_cast<const NeuroLearnedBinaryParams*>(params);
    if (bp && bp->rerank_k > 0) {
        n_rerank = bp->rerank_k;
    }

    size_t cs = code_size();

#pragma omp parallel for
    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * d;

        // Encode query
        std::vector<uint8_t> query_code(cs);
        encode(query, query_code.data());

        // Compute Hamming distances
        std::vector<std::pair<int, idx_t>> scored(ntotal);
        for (idx_t i = 0; i < ntotal; i++) {
            int dist = hamming_distance(query_code.data(), codes.data() + i * cs);
            scored[i] = {dist, i};
        }

        // Get top candidates
        size_t n_cand = std::min(static_cast<size_t>(n_rerank), static_cast<size_t>(ntotal));
        std::partial_sort(scored.begin(), scored.begin() + n_cand, scored.end());

        // Rerank with full precision
        std::vector<std::pair<float, idx_t>> reranked;
        reranked.reserve(n_cand);

        if (do_rerank && !orig_vectors.empty()) {
            for (size_t i = 0; i < n_cand; i++) {
                idx_t idx = scored[i].second;
                const float* vec = orig_vectors.data() + idx * d;
                float dist = fvec_L2sqr(query, vec, d);
                reranked.push_back({dist, idx});
            }
            std::sort(reranked.begin(), reranked.end());
        } else {
            for (size_t i = 0; i < n_cand; i++) {
                reranked.push_back({static_cast<float>(scored[i].first), scored[i].second});
            }
        }

        // Output
        size_t actual_k = std::min(static_cast<size_t>(k), reranked.size());
        for (size_t i = 0; i < actual_k; i++) {
            distances[q * k + i] = reranked[i].first;
            labels[q * k + i] = reranked[i].second;
        }
        for (size_t i = actual_k; i < static_cast<size_t>(k); i++) {
            distances[q * k + i] = std::numeric_limits<float>::max();
            labels[q * k + i] = -1;
        }
    }
}

} // namespace faiss
