/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroHash.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_set>

namespace faiss {

namespace {

/// Count set bits in a uint64
inline int popcount64(uint64_t x) {
#ifdef __GNUC__
    return __builtin_popcountll(x);
#else
    int count = 0;
    while (x) {
        count += x & 1;
        x >>= 1;
    }
    return count;
#endif
}

/// Hamming distance between two code arrays
inline int hamming_distance(
        const uint64_t* a,
        const uint64_t* b,
        int n_words) {
    int dist = 0;
    for (int i = 0; i < n_words; i++) {
        dist += popcount64(a[i] ^ b[i]);
    }
    return dist;
}

/// Get raw data pointer from IndexFlat
const float* get_flat_data_hash(const Index* index) {
    auto flat = dynamic_cast<const IndexFlat*>(index);
    FAISS_THROW_IF_NOT_MSG(
            flat, "IndexNeuroHash requires inner_index to be IndexFlat");
    return flat->get_xb();
}

} // anonymous namespace

/*************************************************************
 * IndexNeuroHash (HS-01: SimHash)
 *************************************************************/

IndexNeuroHash::IndexNeuroHash(
        Index* inner,
        int n_bits,
        int n_tables,
        bool own_inner)
        : IndexNeuro(inner, own_inner),
          n_bits(n_bits),
          n_tables(n_tables) {}

void IndexNeuroHash::train(idx_t /*n*/, const float* /*x*/) {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");
    FAISS_THROW_IF_NOT_MSG(d > 0, "dimension must be positive");

    // Generate random hyperplanes for each table
    // Each hyperplane is a d-dimensional unit vector
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::normal_distribution<float> normal(0.0f, 1.0f);

    hyperplanes.resize(n_tables * n_bits * d);

    for (int t = 0; t < n_tables; t++) {
        for (int b = 0; b < n_bits; b++) {
            float* hp = hyperplanes.data() + (t * n_bits + b) * d;
            float norm = 0.0f;
            for (int i = 0; i < d; i++) {
                hp[i] = normal(rng);
                norm += hp[i] * hp[i];
            }
            // Normalize to unit vector
            norm = std::sqrt(norm);
            if (norm > 1e-10f) {
                for (int i = 0; i < d; i++) {
                    hp[i] /= norm;
                }
            }
        }
    }

    // Initialize hash tables
    tables.resize(n_tables);
    for (auto& t : tables) {
        t.clear();
    }

    is_trained = true;
}

void IndexNeuroHash::compute_code(const float* x, uint64_t* code) const {
    int words = code_words();
    std::fill(code, code + n_tables * words, 0ULL);

    for (int t = 0; t < n_tables; t++) {
        for (int b = 0; b < n_bits; b++) {
            const float* hp = hyperplanes.data() + (t * n_bits + b) * d;
            float dot = 0.0f;
            for (int i = 0; i < d; i++) {
                dot += hp[i] * x[i];
            }
            // Set bit if dot product is positive
            if (dot > 0) {
                int word_idx = b / 64;
                int bit_idx = b % 64;
                code[t * words + word_idx] |= (1ULL << bit_idx);
            }
        }
    }
}

void IndexNeuroHash::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained before adding");

    // Add to inner index
    IndexNeuro::add(n, x);

    // Compute and store hash codes
    int words = code_words();
    idx_t old_ntotal = codes.size() / (n_tables * words);

    codes.resize((old_ntotal + n) * n_tables * words);

    for (idx_t i = 0; i < n; i++) {
        uint64_t* code = codes.data() + (old_ntotal + i) * n_tables * words;
        compute_code(x + i * d, code);

        // Add to hash tables (use first word as bucket key for simplicity)
        for (int t = 0; t < n_tables; t++) {
            uint64_t bucket = code[t * words];
            tables[t][bucket].push_back(old_ntotal + i);
        }
    }
}

void IndexNeuroHash::reset() {
    IndexNeuro::reset();
    codes.clear();
    for (auto& t : tables) {
        t.clear();
    }
}

void IndexNeuroHash::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained");

    const float* data = get_flat_data_hash(inner_index);
    idx_t nb = inner_index->ntotal;
    int words = code_words();

    // Resolve parameters
    int ham_thresh = hamming_threshold;
    bool do_rerank = rerank;

    auto hp = dynamic_cast<const NeuroHashParams*>(params);
    if (hp) {
        if (hp->hamming_threshold >= 0) {
            ham_thresh = hp->hamming_threshold;
        }
        do_rerank = hp->rerank;
    }

    bool collect = false;
    if (params) {
        auto nsp = dynamic_cast<const NeuroSearchParameters*>(params);
        if (nsp) {
            collect = nsp->collect_stats;
        }
    }

    int64_t total_calcs = 0;

#pragma omp parallel for reduction(+ : total_calcs)
    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * d;

        // Compute query hash code
        std::vector<uint64_t> query_code(n_tables * words);
        compute_code(query, query_code.data());

        // Collect candidates from all tables
        std::unordered_set<idx_t> candidate_set;

        for (int t = 0; t < n_tables; t++) {
            const uint64_t* qc = query_code.data() + t * words;

            // Check all codes in this table
            for (idx_t i = 0; i < nb; i++) {
                const uint64_t* dc =
                        codes.data() + i * n_tables * words + t * words;
                int ham = hamming_distance(qc, dc, words);
                if (ham <= ham_thresh) {
                    candidate_set.insert(i);
                }
            }
        }

        std::vector<idx_t> candidates(
                candidate_set.begin(), candidate_set.end());
        size_t n_cand = candidates.size();

        // Compute distances for candidates
        std::vector<std::pair<float, idx_t>> scored;
        scored.reserve(n_cand);

        for (idx_t cand : candidates) {
            float dist;
            if (do_rerank) {
                if (metric) {
                    dist = metric->distance(query, data + cand * d, d);
                } else {
                    dist = fvec_L2sqr(query, data + cand * d, d);
                }
                total_calcs += d;
            } else {
                // Use Hamming distance from first table as proxy
                const uint64_t* qc = query_code.data();
                const uint64_t* dc = codes.data() + cand * n_tables * words;
                dist = static_cast<float>(hamming_distance(qc, dc, words));
            }
            scored.emplace_back(dist, cand);
        }

        // Sort and return top-k
        size_t actual_k = std::min(static_cast<size_t>(k), scored.size());
        std::partial_sort(
                scored.begin(),
                scored.begin() + actual_k,
                scored.end());

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

/*************************************************************
 * IndexNeuroHierarchicalHash (HS-04)
 *************************************************************/

IndexNeuroHierarchicalHash::IndexNeuroHierarchicalHash(
        Index* inner,
        bool own_inner)
        : IndexNeuro(inner, own_inner),
          bits_per_level({8, 32, 128}),
          thresholds({2, 4, 8}) {}

IndexNeuroHierarchicalHash::IndexNeuroHierarchicalHash(
        Index* inner,
        const std::vector<int>& bits,
        const std::vector<int>& thresh,
        bool own_inner)
        : IndexNeuro(inner, own_inner),
          bits_per_level(bits),
          thresholds(thresh) {
    FAISS_THROW_IF_NOT_MSG(
            bits.size() == thresh.size(),
            "bits_per_level and thresholds must have same length");
}

void IndexNeuroHierarchicalHash::train(idx_t /*n*/, const float* /*x*/) {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");
    FAISS_THROW_IF_NOT_MSG(d > 0, "dimension must be positive");

    int n_levels = static_cast<int>(bits_per_level.size());
    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    level_hyperplanes.resize(n_levels);
    level_codes.resize(n_levels);

    for (int l = 0; l < n_levels; l++) {
        int n_bits = bits_per_level[l];
        level_hyperplanes[l].resize(n_bits * d);

        for (int b = 0; b < n_bits; b++) {
            float* hp = level_hyperplanes[l].data() + b * d;
            float norm = 0.0f;
            for (int i = 0; i < d; i++) {
                hp[i] = normal(rng);
                norm += hp[i] * hp[i];
            }
            norm = std::sqrt(norm);
            if (norm > 1e-10f) {
                for (int i = 0; i < d; i++) {
                    hp[i] /= norm;
                }
            }
        }
    }

    is_trained = true;
}

void IndexNeuroHierarchicalHash::compute_level_code(
        int level,
        const float* x,
        uint64_t* code) const {
    int n_bits = bits_per_level[level];
    int words = (n_bits + 63) / 64;
    std::fill(code, code + words, 0ULL);

    const float* hps = level_hyperplanes[level].data();
    for (int b = 0; b < n_bits; b++) {
        const float* hp = hps + b * d;
        float dot = 0.0f;
        for (int i = 0; i < d; i++) {
            dot += hp[i] * x[i];
        }
        if (dot > 0) {
            int word_idx = b / 64;
            int bit_idx = b % 64;
            code[word_idx] |= (1ULL << bit_idx);
        }
    }
}

void IndexNeuroHierarchicalHash::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained before adding");

    IndexNeuro::add(n, x);

    int n_levels = static_cast<int>(bits_per_level.size());
    idx_t old_ntotal = 0;
    if (!level_codes.empty() && !level_codes[0].empty()) {
        int words0 = (bits_per_level[0] + 63) / 64;
        old_ntotal = level_codes[0].size() / words0;
    }

    for (int l = 0; l < n_levels; l++) {
        int words = (bits_per_level[l] + 63) / 64;
        level_codes[l].resize((old_ntotal + n) * words);

        for (idx_t i = 0; i < n; i++) {
            uint64_t* code = level_codes[l].data() + (old_ntotal + i) * words;
            compute_level_code(l, x + i * d, code);
        }
    }
}

void IndexNeuroHierarchicalHash::reset() {
    IndexNeuro::reset();
    for (auto& codes : level_codes) {
        codes.clear();
    }
}

void IndexNeuroHierarchicalHash::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained");

    const float* data = get_flat_data_hash(inner_index);
    idx_t nb = inner_index->ntotal;
    int n_levels = static_cast<int>(bits_per_level.size());

    bool collect = false;
    if (params) {
        auto nsp = dynamic_cast<const NeuroSearchParameters*>(params);
        if (nsp) {
            collect = nsp->collect_stats;
        }
    }

    int64_t total_calcs = 0;

#pragma omp parallel for reduction(+ : total_calcs)
    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * d;

        // Start with all candidates
        std::vector<idx_t> candidates(nb);
        std::iota(candidates.begin(), candidates.end(), 0);

        // Cascade through levels
        for (int l = 0; l < n_levels && candidates.size() > min_candidates; l++) {
            int n_bits = bits_per_level[l];
            int words = (n_bits + 63) / 64;
            int thresh = thresholds[l];

            // Compute query code for this level
            std::vector<uint64_t> query_code(words);
            compute_level_code(l, query, query_code.data());

            // Filter candidates
            std::vector<idx_t> survivors;
            survivors.reserve(candidates.size());

            for (idx_t cand : candidates) {
                const uint64_t* dc =
                        level_codes[l].data() + cand * words;
                int ham = hamming_distance(query_code.data(), dc, words);
                if (ham <= thresh) {
                    survivors.push_back(cand);
                }
            }

            candidates = std::move(survivors);
        }

        // Rerank survivors with true distance
        std::vector<std::pair<float, idx_t>> scored;
        scored.reserve(candidates.size());

        for (idx_t cand : candidates) {
            float dist;
            if (metric) {
                dist = metric->distance(query, data + cand * d, d);
            } else {
                dist = fvec_L2sqr(query, data + cand * d, d);
            }
            scored.emplace_back(dist, cand);
            total_calcs += d;
        }

        // Sort and return top-k
        size_t actual_k = std::min(static_cast<size_t>(k), scored.size());
        if (actual_k > 0) {
            std::partial_sort(
                    scored.begin(),
                    scored.begin() + actual_k,
                    scored.end());
        }

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
