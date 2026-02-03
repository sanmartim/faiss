/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroDiskANN.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace faiss {

IndexNeuroDiskANN::IndexNeuroDiskANN(int d, int max_degree, int search_L)
        : IndexNeuro(nullptr, false),
          max_degree(max_degree),
          search_L(search_L),
          build_L(std::max(search_L, max_degree * 2)) {
    this->d = d;
    is_trained = true;  // No training required
}

IndexNeuroDiskANN::~IndexNeuroDiskANN() {
    if (disk_file.is_open()) {
        disk_file.close();
    }
}

void IndexNeuroDiskANN::train(idx_t /*n*/, const float* /*x*/) {
    // No training required
    is_trained = true;
}

void IndexNeuroDiskANN::set_disk_path(const std::string& path) {
    disk_path = path;
    use_disk = true;
}

void IndexNeuroDiskANN::read_vector(idx_t idx, float* vec) const {
    if (use_disk && !disk_path.empty()) {
        if (!disk_file.is_open()) {
            disk_file.open(disk_path, std::ios::in | std::ios::binary);
        }
        disk_file.seekg(idx * d * sizeof(float));
        disk_file.read(reinterpret_cast<char*>(vec), d * sizeof(float));
    } else {
        std::copy(
                mem_vectors.data() + idx * d,
                mem_vectors.data() + (idx + 1) * d,
                vec);
    }
}

void IndexNeuroDiskANN::flush_to_disk() {
    FAISS_THROW_IF_NOT_MSG(!disk_path.empty(), "disk path not set");
    FAISS_THROW_IF_NOT_MSG(!mem_vectors.empty(), "no vectors to flush");

    std::ofstream out(disk_path, std::ios::binary);
    out.write(
            reinterpret_cast<const char*>(mem_vectors.data()),
            mem_vectors.size() * sizeof(float));
    out.close();

    use_disk = true;
    mem_vectors.clear();
    mem_vectors.shrink_to_fit();
}

void IndexNeuroDiskANN::load_to_memory() {
    FAISS_THROW_IF_NOT_MSG(!disk_path.empty(), "disk path not set");
    FAISS_THROW_IF_NOT(ntotal > 0);

    mem_vectors.resize(ntotal * d);
    std::ifstream in(disk_path, std::ios::binary);
    in.read(
            reinterpret_cast<char*>(mem_vectors.data()),
            ntotal * d * sizeof(float));
    in.close();

    use_disk = false;
}

void IndexNeuroDiskANN::greedy_search(
        const float* query,
        idx_t start,
        int beam_width,
        std::vector<std::pair<float, idx_t>>& result,
        int max_results) const {
    // For now, use brute force search
    // TODO: Implement proper graph-based greedy search
    result.clear();
    result.reserve(ntotal);

    std::vector<float> vec(d);
    for (idx_t i = 0; i < ntotal; i++) {
        read_vector(i, vec.data());
        float dist = fvec_L2sqr(query, vec.data(), d);
        result.push_back({dist, i});
    }

    size_t actual_k = std::min(static_cast<size_t>(max_results), result.size());
    std::partial_sort(result.begin(), result.begin() + actual_k, result.end());
    result.resize(actual_k);
}

void IndexNeuroDiskANN::build_graph(idx_t n, const float* x) {
    // Graph building is not needed for brute force implementation
    // This is a placeholder for future graph-based implementation
    graph.resize(ntotal);
}

void IndexNeuroDiskANN::add(idx_t n, const float* x) {
    // Store vectors in memory
    size_t old_size = mem_vectors.size();
    mem_vectors.resize(old_size + n * d);
    std::copy(x, x + n * d, mem_vectors.data() + old_size);

    ntotal += n;

    // Update graph (currently no-op)
    build_graph(n, x);
}

void IndexNeuroDiskANN::reset() {
    graph.clear();
    mem_vectors.clear();
    ntotal = 0;
    if (disk_file.is_open()) {
        disk_file.close();
    }
}

void IndexNeuroDiskANN::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT(key >= 0 && key < ntotal);
    read_vector(key, recons);
}

void IndexNeuroDiskANN::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(ntotal > 0);

    // Resolve parameters
    int L = search_L;
    auto dp = dynamic_cast<const NeuroDiskANNParams*>(params);
    if (dp && dp->search_L > 0) {
        L = dp->search_L;
    }

#pragma omp parallel for
    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * d;

        std::vector<std::pair<float, idx_t>> results;
        greedy_search(query, 0, L, results, k);

        size_t actual_k = std::min(static_cast<size_t>(k), results.size());
        for (size_t i = 0; i < actual_k; i++) {
            distances[q * k + i] = results[i].first;
            labels[q * k + i] = results[i].second;
        }
        for (size_t i = actual_k; i < static_cast<size_t>(k); i++) {
            distances[q * k + i] = std::numeric_limits<float>::max();
            labels[q * k + i] = -1;
        }
    }
}

} // namespace faiss
