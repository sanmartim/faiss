/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroPlaceCell.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_set>

namespace faiss {

namespace {

const float* get_flat_data_pc(const Index* index) {
    auto flat = dynamic_cast<const IndexFlat*>(index);
    FAISS_THROW_IF_NOT_MSG(
            flat, "IndexNeuroPlaceCell requires inner_index to be IndexFlat");
    return flat->get_xb();
}

} // anonymous namespace

IndexNeuroPlaceCell::IndexNeuroPlaceCell(
        Index* inner,
        int n_cells,
        float field_size,
        bool own_inner)
        : IndexNeuro(inner, own_inner),
          n_cells(n_cells),
          field_size(field_size) {
    inv_2_sigma_sq = 1.0f / (2.0f * field_size * field_size);
}

void IndexNeuroPlaceCell::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");
    FAISS_THROW_IF_NOT_MSG(d > 0, "dimension must be positive");

    inv_2_sigma_sq = 1.0f / (2.0f * field_size * field_size);

    // Initialize cell centers
    // If training data provided, sample from it; otherwise random
    cell_centers.resize(n_cells * d);
    cell_lists.resize(n_cells);

    std::mt19937 rng(42);

    if (n > 0 && x != nullptr) {
        // Sample from training data
        idx_t actual_n_cells = std::min(static_cast<idx_t>(n_cells), n);
        std::vector<idx_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        for (idx_t c = 0; c < actual_n_cells; c++) {
            std::copy(
                    x + indices[c] * d,
                    x + (indices[c] + 1) * d,
                    cell_centers.data() + c * d);
        }

        // If more cells than samples, generate random centers
        std::normal_distribution<float> normal(0.0f, 1.0f);
        for (int c = actual_n_cells; c < n_cells; c++) {
            for (int j = 0; j < d; j++) {
                cell_centers[c * d + j] = normal(rng);
            }
        }
    } else {
        // Random initialization
        std::normal_distribution<float> normal(0.0f, 1.0f);
        for (int c = 0; c < n_cells; c++) {
            for (int j = 0; j < d; j++) {
                cell_centers[c * d + j] = normal(rng);
            }
        }
    }

    // Clear inverted lists
    for (auto& list : cell_lists) {
        list.clear();
    }

    is_trained = true;
}

float IndexNeuroPlaceCell::cell_activation(int cell, const float* x) const {
    const float* center = cell_centers.data() + cell * d;
    float dist_sq = fvec_L2sqr(x, center, d);
    return std::exp(-dist_sq * inv_2_sigma_sq);
}

void IndexNeuroPlaceCell::get_active_cells(
        const float* x,
        float threshold,
        std::vector<int>& cells) const {
    cells.clear();

    // Compute activations for all cells
    std::vector<std::pair<float, int>> activations(n_cells);
    for (int c = 0; c < n_cells; c++) {
        activations[c] = {cell_activation(c, x), c};
    }

    // Sort by activation descending
    std::partial_sort(
            activations.begin(),
            activations.begin() + std::min(max_active_cells, n_cells),
            activations.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

    // Keep cells above threshold
    for (int i = 0; i < std::min(max_active_cells, n_cells); i++) {
        if (activations[i].first >= threshold) {
            cells.push_back(activations[i].second);
        } else {
            break;
        }
    }
}

void IndexNeuroPlaceCell::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained before adding");

    idx_t old_ntotal = inner_index->ntotal;
    IndexNeuro::add(n, x);

    // Assign each new vector to active cells
    for (idx_t i = 0; i < n; i++) {
        std::vector<int> active;
        get_active_cells(x + i * d, activation_threshold, active);

        for (int cell : active) {
            cell_lists[cell].push_back(old_ntotal + i);
        }
    }
}

void IndexNeuroPlaceCell::reset() {
    IndexNeuro::reset();
    for (auto& list : cell_lists) {
        list.clear();
    }
}

void IndexNeuroPlaceCell::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained");

    const float* data = get_flat_data_pc(inner_index);

    // Resolve parameters
    bool do_rerank = rerank;
    float thresh = activation_threshold;

    auto pp = dynamic_cast<const NeuroPlaceCellParams*>(params);
    if (pp) {
        do_rerank = pp->rerank;
        if (pp->activation_threshold >= 0) {
            thresh = pp->activation_threshold;
        }
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

        // Get active cells for query
        std::vector<int> active_cells;
        get_active_cells(query, thresh, active_cells);

        // Union candidates from active cell lists
        std::unordered_set<idx_t> candidate_set;
        for (int cell : active_cells) {
            for (idx_t idx : cell_lists[cell]) {
                candidate_set.insert(idx);
            }
        }

        std::vector<idx_t> candidates(candidate_set.begin(), candidate_set.end());
        size_t n_cand = candidates.size();

        // Score candidates
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
                // Use activation-based score (not distance)
                dist = 0.0f;
            }
            scored.emplace_back(dist, cand);
        }

        // Sort and output
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
