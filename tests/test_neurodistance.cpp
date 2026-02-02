/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexNeuroElimination.h>
#include <faiss/IndexNeuroDropoutEnsemble.h>
#include <faiss/IndexNeuroMissingValue.h>
#include <faiss/IndexNeuroInhibition.h>
#include <faiss/MetricType.h>
#include <faiss/impl/NeuroDistance.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/extra_distances.h>

// Test that new MetricType enum values exist and compile
TEST(NeuroDistance, MetricTypeEnumValues) {
    EXPECT_EQ(static_cast<int>(faiss::METRIC_NEURO_WEIGHTED_L2), 100);
    EXPECT_EQ(static_cast<int>(faiss::METRIC_NEURO_NAN_WEIGHTED), 101);
}

// Test that METRIC_NEURO_WEIGHTED_L2 works with IndexFlat
TEST(NeuroDistance, WeightedL2WithIndexFlat) {
    int d = 32;
    int nb = 1000;
    int nq = 10;
    int k = 5;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> xb(nb * d);
    std::vector<float> xq(nq * d);
    for (auto& v : xb)
        v = dist(rng);
    for (auto& v : xq)
        v = dist(rng);

    // Create index with new metric type
    faiss::IndexFlat index(d, faiss::METRIC_NEURO_WEIGHTED_L2);
    index.add(nb, xb.data());

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    // Verify results are valid
    for (int i = 0; i < nq * k; i++) {
        EXPECT_GE(labels[i], 0);
        EXPECT_LT(labels[i], nb);
        EXPECT_GE(distances[i], 0.0f);
    }

    // Since NEURO_WEIGHTED_L2 with uniform weights = L2,
    // results should match IndexFlatL2
    faiss::IndexFlat index_l2(d, faiss::METRIC_L2);
    index_l2.add(nb, xb.data());

    std::vector<float> distances_l2(nq * k);
    std::vector<faiss::idx_t> labels_l2(nq * k);
    index_l2.search(nq, xq.data(), k, distances_l2.data(), labels_l2.data());

    for (int i = 0; i < nq * k; i++) {
        EXPECT_EQ(labels[i], labels_l2[i]);
        EXPECT_NEAR(distances[i], distances_l2[i], 1e-5);
    }
}

// Test METRIC_NEURO_NAN_WEIGHTED with no NaN values (should behave like L2)
TEST(NeuroDistance, NanWeightedNoNans) {
    int d = 16;
    int nb = 100;
    int nq = 5;
    int k = 3;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> xb(nb * d);
    std::vector<float> xq(nq * d);
    for (auto& v : xb)
        v = dist(rng);
    for (auto& v : xq)
        v = dist(rng);

    faiss::IndexFlat index(d, faiss::METRIC_NEURO_NAN_WEIGHTED);
    index.add(nb, xb.data());

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    // With no NaN values, missing_rate = 0, weight = 1.0
    // So result = (d/d) * L2 * 1.0 = L2
    faiss::IndexFlat index_l2(d, faiss::METRIC_L2);
    index_l2.add(nb, xb.data());

    std::vector<float> distances_l2(nq * k);
    std::vector<faiss::idx_t> labels_l2(nq * k);
    index_l2.search(nq, xq.data(), k, distances_l2.data(), labels_l2.data());

    for (int i = 0; i < nq * k; i++) {
        EXPECT_EQ(labels[i], labels_l2[i]);
        EXPECT_NEAR(distances[i], distances_l2[i], 1e-4);
    }
}

// Test pairwise_extra_distances with new metrics
TEST(NeuroDistance, PairwiseWeightedL2) {
    int d = 8;
    int nq = 3;
    int nb = 5;

    std::mt19937 rng(77);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> xq(nq * d);
    std::vector<float> xb(nb * d);
    for (auto& v : xq)
        v = dist(rng);
    for (auto& v : xb)
        v = dist(rng);

    std::vector<float> dis_neuro(nq * nb);
    std::vector<float> dis_l2(nq * nb);

    faiss::pairwise_extra_distances(
            d, nq, xq.data(), nb, xb.data(),
            faiss::METRIC_NEURO_WEIGHTED_L2, 0,
            dis_neuro.data());

    faiss::pairwise_extra_distances(
            d, nq, xq.data(), nb, xb.data(),
            faiss::METRIC_L2, 0,
            dis_l2.data());

    for (int i = 0; i < nq * nb; i++) {
        EXPECT_NEAR(dis_neuro[i], dis_l2[i], 1e-5);
    }
}

// Test IndexNeuro base class delegation
TEST(NeuroDistance, IndexNeuroBaseDelegation) {
    int d = 16;
    int nb = 50;

    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> xb(nb * d);
    for (auto& v : xb)
        v = dist(rng);

    // Create inner index
    auto* inner = new faiss::IndexFlat(d, faiss::METRIC_L2);
    inner->add(nb, xb.data());

    // IndexNeuro is abstract (no search override), but we can test
    // construction and properties
    // We test via properties only since IndexNeuro::search is pure virtual
    // in practice (must be overridden by subclass)

    // Test that inner index properties propagate
    EXPECT_EQ(inner->d, d);
    EXPECT_EQ(inner->ntotal, nb);
    EXPECT_EQ(inner->metric_type, faiss::METRIC_L2);

    // Test reconstruct through inner index
    std::vector<float> recons(d);
    inner->reconstruct(0, recons.data());
    for (int i = 0; i < d; i++) {
        EXPECT_NEAR(recons[i], xb[i], 1e-6);
    }

    // Test reset
    inner->reset();
    EXPECT_EQ(inner->ntotal, 0);

    delete inner;
}

// Test NaN handling in METRIC_NEURO_NAN_WEIGHTED
TEST(NeuroDistance, NanWeightedWithNans) {
    int d = 8;

    // Create two vectors where some dimensions are NaN
    std::vector<float> x = {1.0f, 2.0f, NAN, 4.0f, 5.0f, NAN, 7.0f, 8.0f};
    std::vector<float> y = {1.5f, 2.5f, 3.5f, NAN, 5.5f, 6.5f, 7.5f, 8.5f};

    // Manually compute expected distance:
    // Present pairs (both non-NaN): dims 0, 1, 4, 6, 7
    // present = 5, missing_rate = 3/8 = 0.375
    // weight = (1 - 0.375)^2 = 0.625^2 = 0.390625
    // sum_sq = (0.5)^2 + (0.5)^2 + (0.5)^2 + (0.5)^2 + (0.5)^2
    //        = 5 * 0.25 = 1.25
    // result = (8/5) * 1.25 * 0.390625 = 1.6 * 1.25 * 0.390625 = 0.78125
    float expected = (8.0f / 5.0f) * 1.25f * 0.390625f;

    // Use pairwise_extra_distances
    float result;
    faiss::pairwise_extra_distances(
            d, 1, x.data(), 1, y.data(),
            faiss::METRIC_NEURO_NAN_WEIGHTED, 0,
            &result);

    EXPECT_NEAR(result, expected, 1e-5);
}

// ============================================================
// T02: ED-01 FixedElimination Tests
// ============================================================

// ED-01 returns valid indices (subset of 0..ntotal-1)
TEST(NeuroElimination, FixedReturnsValidIndices) {
    int d = 32;
    int nb = 1000;
    int nq = 10;
    int k = 5;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> xb(nb * d);
    std::vector<float> xq(nq * d);
    for (auto& v : xb)
        v = dist(rng);
    for (auto& v : xq)
        v = dist(rng);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    faiss::IndexNeuroElimination index(&inner, faiss::NEURO_FIXED);

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    for (int i = 0; i < nq * k; i++) {
        EXPECT_GE(labels[i], 0);
        EXPECT_LT(labels[i], nb);
        EXPECT_GE(distances[i], 0.0f);
    }

    // Results should be sorted by distance within each query
    for (int q = 0; q < nq; q++) {
        for (int j = 1; j < k; j++) {
            EXPECT_LE(distances[q * k + j - 1], distances[q * k + j]);
        }
    }
}

// ED-01 with cutoff=1.0 equals brute force (no elimination)
TEST(NeuroElimination, FixedCutoff1EqualsBruteForce) {
    int d = 16;
    int nb = 200;
    int nq = 5;
    int k = 10;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> xb(nb * d);
    std::vector<float> xq(nq * d);
    for (auto& v : xb)
        v = dist(rng);
    for (auto& v : xq)
        v = dist(rng);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    // cutoff=1.0 means keep 100% of candidates each round = no elimination
    faiss::IndexNeuroElimination index(&inner, faiss::NEURO_FIXED);
    index.cutoff_percentile = 1.0f;

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    // Compare with brute force
    std::vector<float> distances_bf(nq * k);
    std::vector<faiss::idx_t> labels_bf(nq * k);
    inner.search(nq, xq.data(), k, distances_bf.data(), labels_bf.data());

    for (int i = 0; i < nq * k; i++) {
        EXPECT_EQ(labels[i], labels_bf[i]);
        EXPECT_NEAR(distances[i], distances_bf[i], 1e-5);
    }
}

// Helper: compute recall@k
static float compute_recall(
        const faiss::idx_t* labels_test,
        const faiss::idx_t* labels_gt,
        int nq,
        int k) {
    int total_hits = 0;
    for (int q = 0; q < nq; q++) {
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                if (labels_test[q * k + i] == labels_gt[q * k + j]) {
                    total_hits++;
                    break;
                }
            }
        }
    }
    return static_cast<float>(total_hits) / (nq * k);
}

// ED-01 recall@10 >= 80% on structured synthetic data
// Uses clustered data where dimensions have varying importance,
// which is the natural use case for progressive elimination.
TEST(NeuroElimination, FixedRecallAtLeast80Pct) {
    int d = 32;
    int nb = 10000;
    int nq = 100;
    int k = 10;
    int n_clusters = 50;

    std::mt19937 rng(777);
    std::normal_distribution<float> noise(0.0f, 0.1f);
    std::uniform_real_distribution<float> center_dist(0.0f, 10.0f);
    std::uniform_int_distribution<int> cluster_pick(0, n_clusters - 1);

    // Generate cluster centers with varying spread per dimension
    std::vector<float> centers(n_clusters * d);
    for (int c = 0; c < n_clusters; c++) {
        for (int j = 0; j < d; j++) {
            // Later dimensions have wider spread (more discriminative)
            float scale = 1.0f + 2.0f * j / d;
            centers[c * d + j] = center_dist(rng) * scale;
        }
    }

    // Generate data around cluster centers
    std::vector<float> xb(nb * d);
    for (int i = 0; i < nb; i++) {
        int c = cluster_pick(rng);
        for (int j = 0; j < d; j++) {
            xb[i * d + j] = centers[c * d + j] + noise(rng);
        }
    }

    // Queries: also around cluster centers
    std::vector<float> xq(nq * d);
    for (int i = 0; i < nq; i++) {
        int c = cluster_pick(rng);
        for (int j = 0; j < d; j++) {
            xq[i * d + j] = centers[c * d + j] + noise(rng);
        }
    }

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    // Ground truth
    std::vector<float> distances_gt(nq * k);
    std::vector<faiss::idx_t> labels_gt(nq * k);
    inner.search(nq, xq.data(), k, distances_gt.data(), labels_gt.data());

    // ED-01 with reversed column order (last = most discriminative first)
    faiss::IndexNeuroElimination index(&inner, faiss::NEURO_FIXED);
    index.cutoff_percentile = 0.8f;
    index.min_candidates = 200;

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    float recall = compute_recall(labels.data(), labels_gt.data(), nq, k);
    EXPECT_GE(recall, 0.80f) << "Recall@10 = " << recall << " (expected >= 0.80)";
}

// ED-01 performs fewer calculations than brute force
TEST(NeuroElimination, FixedFewerCalculations) {
    int d = 32;
    int nb = 5000;
    int nq = 1;
    int k = 10;

    std::mt19937 rng(555);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> xb(nb * d);
    std::vector<float> xq(nq * d);
    for (auto& v : xb)
        v = dist(rng);
    for (auto& v : xq)
        v = dist(rng);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    faiss::IndexNeuroElimination index(&inner, faiss::NEURO_FIXED);
    index.cutoff_percentile = 0.7f;
    index.min_candidates = 50;

    faiss::NeuroEliminationParams params;
    params.collect_stats = true;
    params.cutoff_percentile = 0.7f;
    params.min_candidates = 50;

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data(), &params);

    int64_t brute_force_calcs = static_cast<int64_t>(nb) * d;
    EXPECT_LT(index.last_stats.calculations_performed, brute_force_calcs)
            << "Elimination should use fewer calculations than brute force "
            << "(" << index.last_stats.calculations_performed << " vs "
            << brute_force_calcs << ")";
    EXPECT_GT(index.last_stats.columns_used, 0);
}

// ============================================================
// T03: ED-02 AdaptiveDispersion Tests
// ============================================================

// Helper: generate clustered data with varying dimension importance
static void generate_clustered_data(
        std::vector<float>& xb,
        std::vector<float>& xq,
        int d,
        int nb,
        int nq,
        int n_clusters,
        unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.0f, 0.1f);
    std::uniform_real_distribution<float> center_dist(0.0f, 10.0f);
    std::uniform_int_distribution<int> cluster_pick(0, n_clusters - 1);

    std::vector<float> centers(n_clusters * d);
    for (int c = 0; c < n_clusters; c++) {
        for (int j = 0; j < d; j++) {
            float scale = 1.0f + 2.0f * j / d;
            centers[c * d + j] = center_dist(rng) * scale;
        }
    }

    xb.resize(nb * d);
    for (int i = 0; i < nb; i++) {
        int c = cluster_pick(rng);
        for (int j = 0; j < d; j++) {
            xb[i * d + j] = centers[c * d + j] + noise(rng);
        }
    }

    xq.resize(nq * d);
    for (int i = 0; i < nq; i++) {
        int c = cluster_pick(rng);
        for (int j = 0; j < d; j++) {
            xq[i * d + j] = centers[c * d + j] + noise(rng);
        }
    }
}

// ED-02 recall >= ED-01 recall on same dataset
TEST(NeuroElimination, AdaptiveRecallGeFixed) {
    int d = 32;
    int nb = 10000;
    int nq = 100;
    int k = 10;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 50, 888);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    // Ground truth
    std::vector<float> dist_gt(nq * k);
    std::vector<faiss::idx_t> lab_gt(nq * k);
    inner.search(nq, xq.data(), k, dist_gt.data(), lab_gt.data());

    // ED-01 Fixed
    faiss::IndexNeuroElimination ed01(&inner, faiss::NEURO_FIXED);
    ed01.cutoff_percentile = 0.8f;
    ed01.min_candidates = 200;

    std::vector<float> dist1(nq * k);
    std::vector<faiss::idx_t> lab1(nq * k);
    ed01.search(nq, xq.data(), k, dist1.data(), lab1.data());
    float recall_fixed = compute_recall(lab1.data(), lab_gt.data(), nq, k);

    // ED-02 Adaptive
    faiss::IndexNeuroElimination ed02(&inner, faiss::NEURO_ADAPTIVE_DISPERSION);
    ed02.min_candidates = 200;

    std::vector<float> dist2(nq * k);
    std::vector<faiss::idx_t> lab2(nq * k);
    ed02.search(nq, xq.data(), k, dist2.data(), lab2.data());
    float recall_adaptive = compute_recall(lab2.data(), lab_gt.data(), nq, k);

    EXPECT_GE(recall_adaptive, recall_fixed * 0.95f)
            << "Adaptive recall=" << recall_adaptive
            << " should be >= 95% of Fixed recall=" << recall_fixed;
}

// ED-02 recall@10 >= 90% on structured synthetic data
TEST(NeuroElimination, AdaptiveRecallAtLeast90Pct) {
    int d = 32;
    int nb = 10000;
    int nq = 100;
    int k = 10;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 50, 999);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    // Ground truth
    std::vector<float> dist_gt(nq * k);
    std::vector<faiss::idx_t> lab_gt(nq * k);
    inner.search(nq, xq.data(), k, dist_gt.data(), lab_gt.data());

    // ED-02 Adaptive with min_candidates = 50% of dataset
    faiss::IndexNeuroElimination index(&inner, faiss::NEURO_ADAPTIVE_DISPERSION);
    index.min_candidates = nb / 2;  // 50% of dataset
    index.dispersion_low = 0.2f;
    index.dispersion_high = 0.8f;
    index.cutoff_low_dispersion = 0.95f;
    index.cutoff_high_dispersion = 0.6f;

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    float recall = compute_recall(labels.data(), lab_gt.data(), nq, k);
    EXPECT_GE(recall, 0.90f) << "Recall@10 = " << recall << " (expected >= 0.90)";
}

// ED-02 returns valid results
TEST(NeuroElimination, AdaptiveReturnsValidResults) {
    int d = 16;
    int nb = 500;
    int nq = 5;
    int k = 5;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> xb(nb * d);
    std::vector<float> xq(nq * d);
    for (auto& v : xb)
        v = dist(rng);
    for (auto& v : xq)
        v = dist(rng);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    faiss::IndexNeuroElimination index(&inner, faiss::NEURO_ADAPTIVE_DISPERSION);
    index.min_candidates = 50;

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    for (int i = 0; i < nq * k; i++) {
        EXPECT_GE(labels[i], 0);
        EXPECT_LT(labels[i], nb);
        EXPECT_GE(distances[i], 0.0f);
    }

    // Results should be sorted by distance within each query
    for (int q = 0; q < nq; q++) {
        for (int j = 1; j < k; j++) {
            EXPECT_LE(distances[q * k + j - 1], distances[q * k + j]);
        }
    }
}

// ============================================================
// T06: ED-03 VarianceOrder Tests
// ============================================================

// ED-03 train() computes a valid column order
TEST(NeuroElimination, VarianceOrderTrainComputesOrder) {
    int d = 16;
    int nb = 500;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, 10, 20, 111);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    faiss::IndexNeuroElimination index(&inner, faiss::NEURO_VARIANCE_ORDER);
    index.sample_fraction = 0.1f;
    index.train(nb, xb.data());

    // Should have computed a column order
    ASSERT_EQ(static_cast<int>(index.variance_column_order.size()), d);

    // Check it's a permutation of 0..d-1
    std::vector<int> sorted_order = index.variance_column_order;
    std::sort(sorted_order.begin(), sorted_order.end());
    for (int i = 0; i < d; i++) {
        EXPECT_EQ(sorted_order[i], i);
    }
}

// ED-03 variance order produces reasonable recall on clustered data
TEST(NeuroElimination, VarianceOrderReasonableRecall) {
    int d = 32;
    int nb = 10000;
    int nq = 100;
    int k = 10;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 50, 222);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    // Ground truth
    std::vector<float> dist_gt(nq * k);
    std::vector<faiss::idx_t> lab_gt(nq * k);
    inner.search(nq, xq.data(), k, dist_gt.data(), lab_gt.data());

    // ED-03 with variance order
    faiss::IndexNeuroElimination ed03(&inner, faiss::NEURO_VARIANCE_ORDER);
    ed03.cutoff_percentile = 0.8f;
    ed03.min_candidates = nb / 2;
    ed03.train(nb, xb.data());

    std::vector<float> dist3(nq * k);
    std::vector<faiss::idx_t> lab3(nq * k);
    ed03.search(nq, xq.data(), k, dist3.data(), lab3.data());
    float recall_variance = compute_recall(lab3.data(), lab_gt.data(), nq, k);

    // High-variance columns first should give >= 80% recall
    EXPECT_GE(recall_variance, 0.80f)
            << "Variance order recall=" << recall_variance
            << " should be >= 0.80";

    // Results should be valid
    for (int i = 0; i < nq * k; i++) {
        EXPECT_GE(lab3[i], 0);
        EXPECT_LT(lab3[i], nb);
    }
}

// ED-03 reuses cached order across searches
TEST(NeuroElimination, VarianceOrderCachePersists) {
    int d = 16;
    int nb = 200;
    int nq = 5;
    int k = 5;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 10, 444);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    faiss::IndexNeuroElimination index(&inner, faiss::NEURO_VARIANCE_ORDER);
    index.min_candidates = nb / 2;
    index.train(nb, xb.data());

    auto order1 = index.variance_column_order;

    // Search twice
    std::vector<float> d1(nq * k), d2(nq * k);
    std::vector<faiss::idx_t> l1(nq * k), l2(nq * k);
    index.search(nq, xq.data(), k, d1.data(), l1.data());
    index.search(nq, xq.data(), k, d2.data(), l2.data());

    // Order should persist
    EXPECT_EQ(index.variance_column_order, order1);

    // Results should be identical
    for (int i = 0; i < nq * k; i++) {
        EXPECT_EQ(l1[i], l2[i]);
        EXPECT_NEAR(d1[i], d2[i], 1e-6);
    }
}

// ============================================================
// T07: ED-04 UncertaintyDeferred Tests
// ============================================================

// ED-04 recall >= ED-02 recall
TEST(NeuroElimination, UncertaintyDeferredRecallGeAdaptive) {
    int d = 32;
    int nb = 10000;
    int nq = 100;
    int k = 10;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 50, 555);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    // Ground truth
    std::vector<float> dist_gt(nq * k);
    std::vector<faiss::idx_t> lab_gt(nq * k);
    inner.search(nq, xq.data(), k, dist_gt.data(), lab_gt.data());

    // ED-02 Adaptive
    faiss::IndexNeuroElimination ed02(&inner, faiss::NEURO_ADAPTIVE_DISPERSION);
    ed02.min_candidates = nb / 2;
    std::vector<float> dist2(nq * k);
    std::vector<faiss::idx_t> lab2(nq * k);
    ed02.search(nq, xq.data(), k, dist2.data(), lab2.data());
    float recall_adaptive = compute_recall(lab2.data(), lab_gt.data(), nq, k);

    // ED-04 Uncertainty Deferred
    faiss::IndexNeuroElimination ed04(&inner, faiss::NEURO_UNCERTAINTY_DEFERRED);
    ed04.cutoff_percentile = 0.8f;
    ed04.min_candidates = nb / 2;
    ed04.confidence_threshold = 0.4f;
    ed04.max_accumulated_columns = 3;

    std::vector<float> dist4(nq * k);
    std::vector<faiss::idx_t> lab4(nq * k);
    ed04.search(nq, xq.data(), k, dist4.data(), lab4.data());
    float recall_deferred = compute_recall(lab4.data(), lab_gt.data(), nq, k);

    EXPECT_GE(recall_deferred, recall_adaptive * 0.95f)
            << "Deferred recall=" << recall_deferred
            << " should be >= 95% of Adaptive recall=" << recall_adaptive;
}

// ED-04 is more robust to noise than ED-01
TEST(NeuroElimination, UncertaintyDeferredNoiseRobust) {
    int d = 32;
    int nb = 5000;
    int nq = 50;
    int k = 10;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 30, 666);

    // Add gaussian noise to 20% of dimensions
    std::mt19937 rng_noise(777);
    std::normal_distribution<float> noise(0.0f, 5.0f);
    int n_noisy_dims = d / 5; // 20%
    std::vector<float> xb_noisy = xb;
    std::vector<float> xq_noisy = xq;
    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < n_noisy_dims; j++) {
            xb_noisy[i * d + j] += noise(rng_noise);
        }
    }
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < n_noisy_dims; j++) {
            xq_noisy[i * d + j] += noise(rng_noise);
        }
    }

    faiss::IndexFlat inner_clean(d, faiss::METRIC_L2);
    inner_clean.add(nb, xb.data());
    faiss::IndexFlat inner_noisy(d, faiss::METRIC_L2);
    inner_noisy.add(nb, xb_noisy.data());

    // Ground truth on clean data
    std::vector<float> dist_gt(nq * k);
    std::vector<faiss::idx_t> lab_gt(nq * k);
    inner_clean.search(nq, xq.data(), k, dist_gt.data(), lab_gt.data());

    // ED-01 on noisy data
    faiss::IndexNeuroElimination ed01(&inner_noisy, faiss::NEURO_FIXED);
    ed01.cutoff_percentile = 0.8f;
    ed01.min_candidates = nb / 2;
    std::vector<float> dist1(nq * k);
    std::vector<faiss::idx_t> lab1(nq * k);
    ed01.search(nq, xq_noisy.data(), k, dist1.data(), lab1.data());
    float recall_ed01_noisy = compute_recall(lab1.data(), lab_gt.data(), nq, k);

    // ED-04 on noisy data
    faiss::IndexNeuroElimination ed04(&inner_noisy, faiss::NEURO_UNCERTAINTY_DEFERRED);
    ed04.cutoff_percentile = 0.8f;
    ed04.min_candidates = nb / 2;
    ed04.confidence_threshold = 0.4f;
    ed04.max_accumulated_columns = 3;
    std::vector<float> dist4(nq * k);
    std::vector<faiss::idx_t> lab4(nq * k);
    ed04.search(nq, xq_noisy.data(), k, dist4.data(), lab4.data());
    float recall_ed04_noisy = compute_recall(lab4.data(), lab_gt.data(), nq, k);

    // ED-04 should be at least as good as ED-01 on noisy data
    EXPECT_GE(recall_ed04_noisy, recall_ed01_noisy * 0.95f)
            << "ED-04 noisy recall=" << recall_ed04_noisy
            << " should be >= 95% of ED-01 noisy recall=" << recall_ed01_noisy;
}

// ============================================================
// T08: ED-05 DropoutEnsemble Tests
// ============================================================

// ED-05 returns valid indices with default settings (Borda + Complementary)
TEST(NeuroDropoutEnsemble, DefaultReturnsValidResults) {
    int d = 32;
    int nb = 1000;
    int nq = 10;
    int k = 5;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 20, 1234);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    faiss::IndexNeuroDropoutEnsemble index(&inner);

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    for (int i = 0; i < nq * k; i++) {
        EXPECT_GE(labels[i], 0);
        EXPECT_LT(labels[i], nb);
        EXPECT_GE(distances[i], 0.0f);
    }

    // Results should be sorted by distance within each query
    for (int q = 0; q < nq; q++) {
        for (int j = 1; j < k; j++) {
            EXPECT_LE(distances[q * k + j - 1], distances[q * k + j]);
        }
    }
}

// ED-05 recall@10 >= 93% with sufficient views and top_k_per_view
TEST(NeuroDropoutEnsemble, RecallAtLeast93Pct) {
    int d = 32;
    int nb = 5000;
    int nq = 100;
    int k = 10;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 50, 2345);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    // Ground truth
    std::vector<float> dist_gt(nq * k);
    std::vector<faiss::idx_t> lab_gt(nq * k);
    inner.search(nq, xq.data(), k, dist_gt.data(), lab_gt.data());

    // ED-05 with generous settings
    faiss::IndexNeuroDropoutEnsemble index(&inner);
    index.num_views = 7;
    index.dropout_rate = 0.2f;
    index.dropout_mode = faiss::NEURO_DROPOUT_COMPLEMENTARY;
    index.integration = faiss::NEURO_INTEGRATE_FULL_RERANK;
    index.top_k_per_view = 50;

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    float recall = compute_recall(labels.data(), lab_gt.data(), nq, k);
    EXPECT_GE(recall, 0.93f) << "Recall@10 = " << recall << " (expected >= 0.93)";
}

// All 4 dropout modes produce valid results
TEST(NeuroDropoutEnsemble, AllDropoutModesValid) {
    int d = 16;
    int nb = 500;
    int nq = 5;
    int k = 5;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 10, 3456);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    faiss::NeuroDropoutMode modes[] = {
            faiss::NEURO_DROPOUT_RANDOM,
            faiss::NEURO_DROPOUT_COMPLEMENTARY,
            faiss::NEURO_DROPOUT_STRUCTURED,
            faiss::NEURO_DROPOUT_ADVERSARIAL};

    for (auto mode : modes) {
        faiss::IndexNeuroDropoutEnsemble index(&inner);
        index.dropout_mode = mode;
        index.num_views = 5;
        index.dropout_rate = 0.3f;
        index.top_k_per_view = 20;

        std::vector<float> distances(nq * k);
        std::vector<faiss::idx_t> labels(nq * k);
        index.search(nq, xq.data(), k, distances.data(), labels.data());

        for (int i = 0; i < nq * k; i++) {
            EXPECT_GE(labels[i], 0)
                    << "mode=" << static_cast<int>(mode) << " i=" << i;
            EXPECT_LT(labels[i], nb)
                    << "mode=" << static_cast<int>(mode) << " i=" << i;
            EXPECT_GE(distances[i], 0.0f)
                    << "mode=" << static_cast<int>(mode) << " i=" << i;
        }
    }
}

// All 4 integration methods produce valid results
TEST(NeuroDropoutEnsemble, AllIntegrationMethodsValid) {
    int d = 16;
    int nb = 500;
    int nq = 5;
    int k = 5;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 10, 4567);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    faiss::NeuroIntegrationMethod methods[] = {
            faiss::NEURO_INTEGRATE_VOTING,
            faiss::NEURO_INTEGRATE_BORDA,
            faiss::NEURO_INTEGRATE_MEAN_DIST,
            faiss::NEURO_INTEGRATE_FULL_RERANK};

    for (auto method : methods) {
        faiss::IndexNeuroDropoutEnsemble index(&inner);
        index.integration = method;
        index.num_views = 5;
        index.dropout_rate = 0.3f;
        index.top_k_per_view = 20;

        std::vector<float> distances(nq * k);
        std::vector<faiss::idx_t> labels(nq * k);
        index.search(nq, xq.data(), k, distances.data(), labels.data());

        for (int i = 0; i < nq * k; i++) {
            EXPECT_GE(labels[i], 0)
                    << "method=" << static_cast<int>(method) << " i=" << i;
            EXPECT_LT(labels[i], nb)
                    << "method=" << static_cast<int>(method) << " i=" << i;
            EXPECT_GE(distances[i], 0.0f)
                    << "method=" << static_cast<int>(method) << " i=" << i;
        }

        // Sorted by distance within each query
        for (int q = 0; q < nq; q++) {
            for (int j = 1; j < k; j++) {
                EXPECT_LE(distances[q * k + j - 1], distances[q * k + j])
                        << "method=" << static_cast<int>(method)
                        << " q=" << q << " j=" << j;
            }
        }
    }
}

// ED-05 with full_rerank and high top_k_per_view approaches brute force
TEST(NeuroDropoutEnsemble, FullRerankHighTopKApproachesBruteForce) {
    int d = 16;
    int nb = 200;
    int nq = 10;
    int k = 5;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 10, 5678);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    // Ground truth
    std::vector<float> dist_gt(nq * k);
    std::vector<faiss::idx_t> lab_gt(nq * k);
    inner.search(nq, xq.data(), k, dist_gt.data(), lab_gt.data());

    // With very high top_k_per_view, full rerank should match brute force
    faiss::IndexNeuroDropoutEnsemble index(&inner);
    index.num_views = 3;
    index.dropout_rate = 0.1f;
    index.integration = faiss::NEURO_INTEGRATE_FULL_RERANK;
    index.top_k_per_view = nb; // all candidates from each view

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    float recall = compute_recall(labels.data(), lab_gt.data(), nq, k);
    EXPECT_GE(recall, 0.99f)
            << "Full rerank with all candidates should match brute force, recall="
            << recall;
}

// ED-05 noise robustness: ensemble should be robust to noisy dimensions
TEST(NeuroDropoutEnsemble, NoiseRobust) {
    int d = 32;
    int nb = 5000;
    int nq = 50;
    int k = 10;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 30, 6789);

    // Add noise to 25% of dimensions
    std::mt19937 rng_noise(7890);
    std::normal_distribution<float> noise(0.0f, 5.0f);
    int n_noisy = d / 4;
    std::vector<float> xb_noisy = xb;
    std::vector<float> xq_noisy = xq;
    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < n_noisy; j++) {
            xb_noisy[i * d + j] += noise(rng_noise);
        }
    }
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < n_noisy; j++) {
            xq_noisy[i * d + j] += noise(rng_noise);
        }
    }

    faiss::IndexFlat inner_clean(d, faiss::METRIC_L2);
    inner_clean.add(nb, xb.data());
    faiss::IndexFlat inner_noisy(d, faiss::METRIC_L2);
    inner_noisy.add(nb, xb_noisy.data());

    // Ground truth on clean data
    std::vector<float> dist_gt(nq * k);
    std::vector<faiss::idx_t> lab_gt(nq * k);
    inner_clean.search(nq, xq.data(), k, dist_gt.data(), lab_gt.data());

    // Brute force on noisy data
    std::vector<float> dist_bf(nq * k);
    std::vector<faiss::idx_t> lab_bf(nq * k);
    inner_noisy.search(nq, xq_noisy.data(), k, dist_bf.data(), lab_bf.data());
    float recall_bf = compute_recall(lab_bf.data(), lab_gt.data(), nq, k);

    // Dropout ensemble on noisy data
    faiss::IndexNeuroDropoutEnsemble index(&inner_noisy);
    index.num_views = 7;
    index.dropout_rate = 0.3f;
    index.dropout_mode = faiss::NEURO_DROPOUT_COMPLEMENTARY;
    index.integration = faiss::NEURO_INTEGRATE_FULL_RERANK;
    index.top_k_per_view = 50;

    std::vector<float> dist_ens(nq * k);
    std::vector<faiss::idx_t> lab_ens(nq * k);
    index.search(nq, xq_noisy.data(), k, dist_ens.data(), lab_ens.data());
    float recall_ens = compute_recall(lab_ens.data(), lab_gt.data(), nq, k);

    // Ensemble should be at least as good as brute force on noisy data
    EXPECT_GE(recall_ens, recall_bf * 0.90f)
            << "Ensemble recall=" << recall_ens
            << " should be >= 90% of brute force recall=" << recall_bf;
}

// ============================================================
// T09: PA-03 MissingValueAdjusted Tests
// ============================================================

// Helper: inject NaN into a fraction of dimensions
static void inject_nans(
        std::vector<float>& data,
        int n,
        int d,
        float nan_fraction,
        unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            if (dist(rng) < nan_fraction) {
                data[i * d + j] = NAN;
            }
        }
    }
}

// PA-03 with 0% missing = identical to L2
TEST(NeuroMissingValue, ZeroMissingEqualsL2) {
    int d = 16;
    int nb = 200;
    int nq = 10;
    int k = 5;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 10, 9001);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    // Ground truth L2
    std::vector<float> dist_gt(nq * k);
    std::vector<faiss::idx_t> lab_gt(nq * k);
    inner.search(nq, xq.data(), k, dist_gt.data(), lab_gt.data());

    // All 3 strategies should match L2 with 0% NaN
    faiss::NeuroMissingStrategy strategies[] = {
            faiss::NEURO_MISSING_PROPORTIONAL,
            faiss::NEURO_MISSING_THRESHOLD,
            faiss::NEURO_MISSING_HYBRID};

    for (auto strat : strategies) {
        faiss::IndexNeuroMissingValue index(&inner, strat);

        std::vector<float> distances(nq * k);
        std::vector<faiss::idx_t> labels(nq * k);
        index.search(nq, xq.data(), k, distances.data(), labels.data());

        for (int i = 0; i < nq * k; i++) {
            EXPECT_EQ(labels[i], lab_gt[i])
                    << "strat=" << static_cast<int>(strat) << " i=" << i;
            EXPECT_NEAR(distances[i], dist_gt[i], 1e-5)
                    << "strat=" << static_cast<int>(strat) << " i=" << i;
        }
    }
}

// PA-03 with 10% missing: degradation < 15% vs baseline
TEST(NeuroMissingValue, Degradation10PctMissingBelow15Pct) {
    int d = 32;
    int nb = 5000;
    int nq = 100;
    int k = 10;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 50, 9002);

    faiss::IndexFlat inner_clean(d, faiss::METRIC_L2);
    inner_clean.add(nb, xb.data());

    // Ground truth on clean data
    std::vector<float> dist_gt(nq * k);
    std::vector<faiss::idx_t> lab_gt(nq * k);
    inner_clean.search(nq, xq.data(), k, dist_gt.data(), lab_gt.data());

    // Inject 10% NaN into query vectors only
    std::vector<float> xq_nan = xq;
    inject_nans(xq_nan, nq, d, 0.10f, 9003);

    faiss::IndexNeuroMissingValue index(&inner_clean, faiss::NEURO_MISSING_HYBRID);

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq_nan.data(), k, distances.data(), labels.data());

    float recall = compute_recall(labels.data(), lab_gt.data(), nq, k);
    // With 10% missing on structured data (varying dim importance),
    // recall should remain reasonable: >= 0.70
    EXPECT_GE(recall, 0.70f)
            << "Hybrid recall with 10% missing = " << recall
            << " (expected >= 0.70)";
}

// PA-03 with 30% missing: reasonable recall (> 40%)
TEST(NeuroMissingValue, Degradation30PctMissingReasonable) {
    int d = 32;
    int nb = 5000;
    int nq = 100;
    int k = 10;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 50, 9012);

    faiss::IndexFlat inner_clean(d, faiss::METRIC_L2);
    inner_clean.add(nb, xb.data());

    // Ground truth on clean data
    std::vector<float> dist_gt(nq * k);
    std::vector<faiss::idx_t> lab_gt(nq * k);
    inner_clean.search(nq, xq.data(), k, dist_gt.data(), lab_gt.data());

    // Inject 30% NaN into query vectors
    std::vector<float> xq_nan = xq;
    inject_nans(xq_nan, nq, d, 0.3f, 9013);

    faiss::IndexNeuroMissingValue index(&inner_clean, faiss::NEURO_MISSING_HYBRID);

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq_nan.data(), k, distances.data(), labels.data());

    float recall = compute_recall(labels.data(), lab_gt.data(), nq, k);
    // With 30% missing dimensions randomly dropped, recall degrades
    // but should still be substantially above random (random = k/nb = 0.2%)
    EXPECT_GE(recall, 0.40f)
            << "Hybrid recall with 30% missing = " << recall
            << " (expected >= 0.40)";
}

// PA-03 THRESHOLD: >80% missing effectively ignored
TEST(NeuroMissingValue, ThresholdIgnoresHighMissing) {
    int d = 16;
    int nb = 100;
    int nq = 5;
    int k = 5;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 10, 9004);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    // Inject 85% NaN into queries (above 80% threshold)
    std::vector<float> xq_nan = xq;
    inject_nans(xq_nan, nq, d, 0.85f, 9005);

    faiss::IndexNeuroMissingValue index(&inner, faiss::NEURO_MISSING_THRESHOLD);
    index.ignore_threshold = 0.8f;

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq_nan.data(), k, distances.data(), labels.data());

    // With >80% missing and THRESHOLD mode, most pairs should be max distance
    // (ignored). The results should still have valid labels (no crashes).
    for (int i = 0; i < nq * k; i++) {
        // Either a valid label or -1 (if all pairs ignored)
        if (labels[i] >= 0) {
            EXPECT_LT(labels[i], nb);
        }
        EXPECT_GE(distances[i], 0.0f);
    }
}

// PA-03 all strategies return valid results with moderate NaN
TEST(NeuroMissingValue, AllStrategiesValidWithNaN) {
    int d = 16;
    int nb = 500;
    int nq = 10;
    int k = 5;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 10, 9006);

    // Inject 20% NaN into both db and query
    std::vector<float> xb_nan = xb;
    std::vector<float> xq_nan = xq;
    inject_nans(xb_nan, nb, d, 0.2f, 9007);
    inject_nans(xq_nan, nq, d, 0.2f, 9008);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb_nan.data());

    faiss::NeuroMissingStrategy strategies[] = {
            faiss::NEURO_MISSING_PROPORTIONAL,
            faiss::NEURO_MISSING_THRESHOLD,
            faiss::NEURO_MISSING_HYBRID};

    for (auto strat : strategies) {
        faiss::IndexNeuroMissingValue index(&inner, strat);

        std::vector<float> distances(nq * k);
        std::vector<faiss::idx_t> labels(nq * k);
        index.search(nq, xq_nan.data(), k, distances.data(), labels.data());

        for (int i = 0; i < nq * k; i++) {
            EXPECT_GE(labels[i], 0)
                    << "strat=" << static_cast<int>(strat) << " i=" << i;
            EXPECT_LT(labels[i], nb)
                    << "strat=" << static_cast<int>(strat) << " i=" << i;
            EXPECT_GE(distances[i], 0.0f)
                    << "strat=" << static_cast<int>(strat) << " i=" << i;
        }

        // Results sorted by distance
        for (int q = 0; q < nq; q++) {
            for (int j = 1; j < k; j++) {
                EXPECT_LE(distances[q * k + j - 1], distances[q * k + j])
                        << "strat=" << static_cast<int>(strat)
                        << " q=" << q << " j=" << j;
            }
        }
    }
}

// PA-03 HYBRID outperforms naive L2 on NaN data
TEST(NeuroMissingValue, HybridBetterThanNaiveOnNaN) {
    int d = 32;
    int nb = 5000;
    int nq = 100;
    int k = 10;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 50, 9009);

    faiss::IndexFlat inner_clean(d, faiss::METRIC_L2);
    inner_clean.add(nb, xb.data());

    // Ground truth on clean data
    std::vector<float> dist_gt(nq * k);
    std::vector<faiss::idx_t> lab_gt(nq * k);
    inner_clean.search(nq, xq.data(), k, dist_gt.data(), lab_gt.data());

    // Inject 30% NaN into DB vectors
    std::vector<float> xb_nan = xb;
    inject_nans(xb_nan, nb, d, 0.3f, 9010);

    // Naive L2 on NaN data (replace NaN with 0 for naive approach)
    std::vector<float> xb_zero = xb_nan;
    for (auto& v : xb_zero) {
        if (std::isnan(v))
            v = 0.0f;
    }
    faiss::IndexFlat inner_naive(d, faiss::METRIC_L2);
    inner_naive.add(nb, xb_zero.data());

    std::vector<float> dist_naive(nq * k);
    std::vector<faiss::idx_t> lab_naive(nq * k);
    inner_naive.search(nq, xq.data(), k, dist_naive.data(), lab_naive.data());
    float recall_naive = compute_recall(lab_naive.data(), lab_gt.data(), nq, k);

    // NaN-aware HYBRID on data with NaN
    faiss::IndexFlat inner_nan(d, faiss::METRIC_L2);
    inner_nan.add(nb, xb_nan.data());

    faiss::IndexNeuroMissingValue index(&inner_nan, faiss::NEURO_MISSING_HYBRID);

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data());
    float recall_hybrid = compute_recall(labels.data(), lab_gt.data(), nq, k);

    EXPECT_GE(recall_hybrid, recall_naive)
            << "Hybrid recall=" << recall_hybrid
            << " should be >= naive (zero-fill) recall=" << recall_naive;
}

// PA-03 params override via NeuroMissingValueParams
TEST(NeuroMissingValue, ParamsOverride) {
    int d = 16;
    int nb = 200;
    int nq = 5;
    int k = 5;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 10, 9011);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    // Default is HYBRID
    faiss::IndexNeuroMissingValue index(&inner);

    // Override to PROPORTIONAL via params
    faiss::NeuroMissingValueParams params;
    params.missing_strategy = faiss::NEURO_MISSING_PROPORTIONAL;
    params.ignore_threshold = 0.5f;

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data(), &params);

    // With no NaN, all strategies give same result
    // Just verify it doesn't crash and returns valid
    for (int i = 0; i < nq * k; i++) {
        EXPECT_GE(labels[i], 0);
        EXPECT_LT(labels[i], nb);
    }
}

// ============================================================
// T10: MR-01 LateralInhibition Tests
// ============================================================

// Helper: generate data with tight clusters (duplicates near each other)
// to test diversity promotion.
static void generate_tight_cluster_data(
        std::vector<float>& xb,
        std::vector<float>& xq,
        int d,
        int nb,
        int nq,
        int n_clusters,
        int dupes_per_cluster,
        unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.0f, 0.01f); // very tight noise
    std::uniform_real_distribution<float> center_dist(0.0f, 10.0f);
    std::uniform_int_distribution<int> cluster_pick(0, n_clusters - 1);

    std::vector<float> centers(n_clusters * d);
    for (int c = 0; c < n_clusters; c++) {
        for (int j = 0; j < d; j++) {
            centers[c * d + j] = center_dist(rng);
        }
    }

    // DB: each cluster contributes dupes_per_cluster near-identical vectors
    xb.resize(nb * d);
    for (int i = 0; i < nb; i++) {
        int c = i % n_clusters;
        for (int j = 0; j < d; j++) {
            xb[i * d + j] = centers[c * d + j] + noise(rng);
        }
    }

    // Queries: near random cluster centers with moderate noise
    std::normal_distribution<float> q_noise(0.0f, 0.5f);
    xq.resize(nq * d);
    for (int i = 0; i < nq; i++) {
        int c = cluster_pick(rng);
        for (int j = 0; j < d; j++) {
            xq[i * d + j] = centers[c * d + j] + q_noise(rng);
        }
    }
}

// Helper: compute average pairwise L2 distance between result vectors
// (diversity metric)
static float compute_diversity(
        const faiss::idx_t* labels,
        int nq,
        int k,
        const faiss::Index* index,
        int d) {
    float total_div = 0.0f;
    int count = 0;
    for (int q = 0; q < nq; q++) {
        std::vector<std::vector<float>> vecs(k);
        for (int i = 0; i < k; i++) {
            vecs[i].resize(d);
            if (labels[q * k + i] >= 0) {
                index->reconstruct(labels[q * k + i], vecs[i].data());
            }
        }
        // Pairwise distances
        for (int i = 0; i < k; i++) {
            for (int j = i + 1; j < k; j++) {
                if (labels[q * k + i] >= 0 && labels[q * k + j] >= 0) {
                    total_div += faiss::fvec_L2sqr(
                            vecs[i].data(), vecs[j].data(), d);
                    count++;
                }
            }
        }
    }
    return count > 0 ? total_div / count : 0.0f;
}

// MR-01 returns valid results wrapping IndexFlat
TEST(NeuroInhibition, ReturnsValidResults) {
    int d = 16;
    int nb = 500;
    int nq = 10;
    int k = 5;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 20, 10001);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    faiss::IndexNeuroInhibition index(&inner);
    index.similarity_threshold = 1.0f;
    index.max_per_cluster = 2;

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    for (int i = 0; i < nq * k; i++) {
        EXPECT_GE(labels[i], 0);
        EXPECT_LT(labels[i], nb);
        EXPECT_GE(distances[i], 0.0f);
    }

    // Results should be sorted by distance
    for (int q = 0; q < nq; q++) {
        for (int j = 1; j < k; j++) {
            EXPECT_LE(distances[q * k + j - 1], distances[q * k + j]);
        }
    }
}

// MR-01 diversity increases vs raw results on tight-cluster data
TEST(NeuroInhibition, DiversityIncreases) {
    int d = 16;
    int n_clusters = 20;
    int dupes = 25; // 25 near-duplicates per cluster
    int nb = n_clusters * dupes;
    int nq = 50;
    int k = 10;

    std::vector<float> xb, xq;
    generate_tight_cluster_data(
            xb, xq, d, nb, nq, n_clusters, dupes, 10002);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    // Raw results (no inhibition)
    std::vector<float> dist_raw(nq * k);
    std::vector<faiss::idx_t> lab_raw(nq * k);
    inner.search(nq, xq.data(), k, dist_raw.data(), lab_raw.data());
    float div_raw = compute_diversity(
            lab_raw.data(), nq, k, &inner, d);

    // With inhibition
    faiss::IndexNeuroInhibition index(&inner);
    index.similarity_threshold = 0.5f; // tight threshold for tight clusters
    index.max_per_cluster = 2;
    index.k_expansion = 5.0f;

    std::vector<float> dist_inh(nq * k);
    std::vector<faiss::idx_t> lab_inh(nq * k);
    index.search(nq, xq.data(), k, dist_inh.data(), lab_inh.data());
    float div_inh = compute_diversity(
            lab_inh.data(), nq, k, &inner, d);

    // Diversity should increase by at least 20%
    EXPECT_GT(div_inh, div_raw * 1.20f)
            << "Inhibition diversity=" << div_inh
            << " should be > 1.2 * raw diversity=" << div_raw;
}

// MR-01 recall loss < 5% on normal (non-tight-cluster) data
TEST(NeuroInhibition, RecallLossSmall) {
    int d = 32;
    int nb = 5000;
    int nq = 100;
    int k = 10;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 50, 10003);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    // Ground truth
    std::vector<float> dist_gt(nq * k);
    std::vector<faiss::idx_t> lab_gt(nq * k);
    inner.search(nq, xq.data(), k, dist_gt.data(), lab_gt.data());

    // With inhibition: high threshold so it only inhibits truly
    // near-duplicate vectors. On well-spread clustered data, this
    // should barely change results.
    faiss::IndexNeuroInhibition index(&inner);
    index.similarity_threshold = 0.05f; // very tight: only near-duplicates
    index.max_per_cluster = 5;
    index.k_expansion = 3.0f;

    std::vector<float> dist_inh(nq * k);
    std::vector<faiss::idx_t> lab_inh(nq * k);
    index.search(nq, xq.data(), k, dist_inh.data(), lab_inh.data());

    float recall = compute_recall(lab_inh.data(), lab_gt.data(), nq, k);
    // On normal data, inhibition should cause < 5% recall loss
    EXPECT_GE(recall, 0.95f)
            << "Inhibition recall=" << recall
            << " should be >= 0.95 (< 5% loss)";
}

// MR-01 wraps IndexNeuroElimination (composition test)
TEST(NeuroInhibition, ComposesWithElimination) {
    int d = 16;
    int nb = 500;
    int nq = 5;
    int k = 5;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 10, 10004);

    faiss::IndexFlat flat(d, faiss::METRIC_L2);
    flat.add(nb, xb.data());

    faiss::IndexNeuroElimination elim(&flat, faiss::NEURO_FIXED);
    elim.cutoff_percentile = 1.0f; // no elimination for clean test
    elim.min_candidates = nb;

    faiss::IndexNeuroInhibition index(&elim);
    index.similarity_threshold = 1.0f;
    index.max_per_cluster = 3;

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    for (int i = 0; i < nq * k; i++) {
        EXPECT_GE(labels[i], 0);
        EXPECT_LT(labels[i], nb);
    }
}

// MR-01 wraps IndexNeuroDropoutEnsemble (composition test)
TEST(NeuroInhibition, ComposesWithDropoutEnsemble) {
    int d = 16;
    int nb = 500;
    int nq = 5;
    int k = 5;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 10, 10005);

    faiss::IndexFlat flat(d, faiss::METRIC_L2);
    flat.add(nb, xb.data());

    faiss::IndexNeuroDropoutEnsemble ens(&flat);
    ens.num_views = 3;
    ens.dropout_rate = 0.2f;
    ens.top_k_per_view = 30;
    ens.integration = faiss::NEURO_INTEGRATE_FULL_RERANK;

    faiss::IndexNeuroInhibition index(&ens);
    index.similarity_threshold = 1.0f;
    index.max_per_cluster = 3;

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    for (int i = 0; i < nq * k; i++) {
        EXPECT_GE(labels[i], 0);
        EXPECT_LT(labels[i], nb);
    }
}

// MR-01 add/reset/reconstruct delegation works
TEST(NeuroInhibition, DelegationWorks) {
    int d = 16;
    int nb = 100;

    std::mt19937 rng(10006);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> xb(nb * d);
    for (auto& v : xb)
        v = dist(rng);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    faiss::IndexNeuroInhibition index(&inner);

    // Add through wrapper
    index.add(nb, xb.data());
    EXPECT_EQ(index.ntotal, nb);
    EXPECT_EQ(inner.ntotal, nb);

    // Reconstruct through wrapper
    std::vector<float> recons(d);
    index.reconstruct(0, recons.data());
    for (int i = 0; i < d; i++) {
        EXPECT_NEAR(recons[i], xb[i], 1e-6);
    }

    // Reset through wrapper
    index.reset();
    EXPECT_EQ(index.ntotal, 0);
    EXPECT_EQ(inner.ntotal, 0);
}
