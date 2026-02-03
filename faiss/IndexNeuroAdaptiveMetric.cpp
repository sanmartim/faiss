/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroAdaptiveMetric.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <random>

namespace faiss {

IndexNeuroAdaptiveMetric::IndexNeuroAdaptiveMetric(
        Index* sub_index,
        bool own_fields)
        : Index(sub_index->d, sub_index->metric_type),
          sub_index(sub_index),
          own_fields(own_fields) {
    ntotal = sub_index->ntotal;
    is_trained = sub_index->is_trained;
}

IndexNeuroAdaptiveMetric::~IndexNeuroAdaptiveMetric() {
    if (own_fields && sub_index) {
        delete sub_index;
    }
}

void IndexNeuroAdaptiveMetric::add_metric(
        std::unique_ptr<NeuroMetric> metric,
        const std::string& name) {
    candidate_metrics.push_back(std::move(metric));
    metric_names.push_back(name);
}

void IndexNeuroAdaptiveMetric::add_default_metrics() {
    add_metric(std::make_unique<NeuroMetricL2>(), "L2");
    add_metric(std::make_unique<NeuroMetricCosine>(), "Cosine");
    add_metric(std::make_unique<NeuroMetricDot>(), "Dot");
}

void IndexNeuroAdaptiveMetric::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");
    sub_index->train(n, x);
    is_trained = sub_index->is_trained;

    // Add default metrics if none added
    if (candidate_metrics.empty()) {
        add_default_metrics();
    }

    // Adapt based on training data
    if (n > 0 && x != nullptr) {
        adapt(std::min(n, static_cast<idx_t>(1000)), x, false);
    }
}

void IndexNeuroAdaptiveMetric::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");
    sub_index->add(n, x);
    ntotal = sub_index->ntotal;
}

void IndexNeuroAdaptiveMetric::reset() {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");
    sub_index->reset();
    ntotal = 0;
}

void IndexNeuroAdaptiveMetric::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");
    sub_index->reconstruct(key, recons);
}

void IndexNeuroAdaptiveMetric::analyze_distribution(
        idx_t n,
        const float* x,
        float& sparsity,
        float& normality) const {
    // Compute sparsity (fraction of near-zero values)
    int zero_count = 0;
    float threshold = 1e-6f;
    for (idx_t i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            if (std::abs(x[i * d + j]) < threshold) {
                zero_count++;
            }
        }
    }
    sparsity = static_cast<float>(zero_count) / (n * d);

    // Compute normality (check if vectors are normalized)
    float norm_variance = 0.0f;
    for (idx_t i = 0; i < n; i++) {
        float norm = 0.0f;
        for (int j = 0; j < d; j++) {
            norm += x[i * d + j] * x[i * d + j];
        }
        norm = std::sqrt(norm);
        float diff = norm - 1.0f;
        norm_variance += diff * diff;
    }
    norm_variance /= n;
    normality = 1.0f / (1.0f + norm_variance);  // Higher = more normalized
}

float IndexNeuroAdaptiveMetric::evaluate_metric(
        int metric_idx,
        idx_t n,
        const float* x) const {
    // Evaluate metric based on k-NN consistency
    // A good metric should have high correlation between
    // distance and neighbor rank

    if (n < 10) return 0.0f;

    const NeuroMetric* m = candidate_metrics[metric_idx].get();
    std::mt19937 rng(42);
    std::uniform_int_distribution<idx_t> dist(0, n - 1);

    // Sample queries and compute rank consistency
    int n_queries = std::min(static_cast<idx_t>(50), n);
    float total_score = 0.0f;

    for (int q = 0; q < n_queries; q++) {
        idx_t query_idx = dist(rng);
        const float* query = x + query_idx * d;

        // Compute distances to all others
        std::vector<std::pair<float, idx_t>> dists;
        dists.reserve(n - 1);
        for (idx_t i = 0; i < n; i++) {
            if (i == query_idx) continue;
            float dist_val = m->distance(query, x + i * d, d);
            dists.emplace_back(dist_val, i);
        }

        // Sort by distance
        std::sort(dists.begin(), dists.end());

        // Compute variance of distances (lower variance in top-k = more discriminative)
        int k = std::min(10, static_cast<int>(dists.size()));
        float mean = 0.0f;
        for (int i = 0; i < k; i++) {
            mean += dists[i].first;
        }
        mean /= k;

        float var = 0.0f;
        for (int i = 0; i < k; i++) {
            float diff = dists[i].first - mean;
            var += diff * diff;
        }
        var /= k;

        // Higher discrimination (wider spread) is better
        float spread = dists[k].first - dists[0].first;
        if (spread > 0) {
            total_score += std::sqrt(var) / spread;
        }
    }

    return total_score / n_queries;
}

void IndexNeuroAdaptiveMetric::adapt(idx_t n, const float* x, bool combine) {
    FAISS_THROW_IF_NOT_MSG(!candidate_metrics.empty(), "no candidate metrics");

    // Analyze data distribution
    float sparsity, normality;
    analyze_distribution(n, x, sparsity, normality);

    // Evaluate each metric
    int n_metrics = static_cast<int>(candidate_metrics.size());
    std::vector<float> scores(n_metrics);

    for (int m = 0; m < n_metrics; m++) {
        scores[m] = evaluate_metric(m, n, x);

        // Adjust score based on data characteristics
        if (metric_names[m] == "Cosine" && normality > 0.9f) {
            scores[m] *= 1.2f;  // Boost cosine for normalized data
        }
        if (metric_names[m] == "L2" && sparsity < 0.1f) {
            scores[m] *= 1.1f;  // Boost L2 for dense data
        }
    }

    if (combine) {
        // Combination mode: normalize scores to weights
        float sum = 0.0f;
        for (float s : scores) {
            sum += s;
        }
        metric_weights.resize(n_metrics);
        for (int m = 0; m < n_metrics; m++) {
            metric_weights[m] = (sum > 0) ? scores[m] / sum : 1.0f / n_metrics;
        }
        selected_metric = -1;
    } else {
        // Selection mode: pick best metric
        int best = 0;
        for (int m = 1; m < n_metrics; m++) {
            if (scores[m] > scores[best]) {
                best = m;
            }
        }
        selected_metric = best;
        metric_weights.clear();
    }

    adapted = true;
}

std::string IndexNeuroAdaptiveMetric::get_selected_metric_name() const {
    if (selected_metric >= 0 &&
        selected_metric < static_cast<int>(metric_names.size())) {
        return metric_names[selected_metric];
    }
    return "Combined";
}

float IndexNeuroAdaptiveMetric::combined_distance(
        const float* x1,
        const float* x2) const {
    if (selected_metric >= 0) {
        return candidate_metrics[selected_metric]->distance(x1, x2, d);
    }

    // Weighted combination
    float total = 0.0f;
    for (size_t m = 0; m < candidate_metrics.size(); m++) {
        float dist = candidate_metrics[m]->distance(x1, x2, d);
        total += metric_weights[m] * dist;
    }
    return total;
}

void IndexNeuroAdaptiveMetric::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");

    // If not adapted, use default sub_index search
    if (!adapted || candidate_metrics.empty()) {
        sub_index->search(n, x, k, distances, labels, params);
        return;
    }

    // Custom search using selected/combined metric
    idx_t nb = sub_index->ntotal;
    if (nb == 0) {
        for (idx_t q = 0; q < n; q++) {
            for (idx_t i = 0; i < k; i++) {
                distances[q * k + i] = std::numeric_limits<float>::max();
                labels[q * k + i] = -1;
            }
        }
        return;
    }

    // Reconstruct database vectors (expensive but necessary for custom metric)
    // In practice, you'd want to cache this or use a different approach
    std::vector<float> db_vectors(nb * d);
    for (idx_t i = 0; i < nb; i++) {
        sub_index->reconstruct(i, db_vectors.data() + i * d);
    }

#pragma omp parallel for
    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * d;

        std::vector<std::pair<float, idx_t>> scored(nb);
        for (idx_t i = 0; i < nb; i++) {
            float dist = combined_distance(query, db_vectors.data() + i * d);
            scored[i] = {dist, i};
        }

        size_t actual_k = std::min(static_cast<size_t>(k), static_cast<size_t>(nb));
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
}

} // namespace faiss
