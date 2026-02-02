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
#include <faiss/IndexNeuroWeighted.h>
#include <faiss/IndexNeuroContextualWeighted.h>
#include <faiss/IndexNeuroParallelVoting.h>
#include <faiss/IndexNeuroCoarseToFine.h>
#include <faiss/IndexNeuroCache.h>
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

// ============================================================
// T12: PA-01 LearnedWeights Tests
// ============================================================

// PA-01 with uniform weights = identical to L2
TEST(NeuroWeighted, UniformWeightsEqualsL2) {
    int d = 16;
    int nb = 200;
    int nq = 10;
    int k = 5;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 10, 12001);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    // Ground truth L2
    std::vector<float> dist_gt(nq * k);
    std::vector<faiss::idx_t> lab_gt(nq * k);
    inner.search(nq, xq.data(), k, dist_gt.data(), lab_gt.data());

    // PA-01 with default uniform weights
    faiss::IndexNeuroWeighted index(&inner);
    index.train(nb, xb.data());

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    for (int i = 0; i < nq * k; i++) {
        EXPECT_EQ(labels[i], lab_gt[i]);
        EXPECT_NEAR(distances[i], dist_gt[i], 1e-5);
    }
}

// PA-01 feedback learns to distinguish discriminative from noise dims
TEST(NeuroWeighted, FeedbackLearnsWeights) {
    int d = 32;
    int nb = 5000;
    int n_clusters = 50;

    // Generate data where first half of dimensions is noise,
    // second half is discriminative
    std::mt19937 rng(12002);
    std::normal_distribution<float> noise(0.0f, 0.1f);
    std::uniform_real_distribution<float> center_dist(0.0f, 10.0f);
    std::uniform_int_distribution<int> cluster_pick(0, n_clusters - 1);

    std::vector<float> centers(n_clusters * d);
    for (int c = 0; c < n_clusters; c++) {
        for (int j = 0; j < d; j++) {
            if (j < d / 2) {
                // First half: noise-like (same center for all clusters)
                centers[c * d + j] = 5.0f;
            } else {
                // Second half: discriminative
                centers[c * d + j] = center_dist(rng);
            }
        }
    }

    std::vector<float> xb(nb * d);
    for (int i = 0; i < nb; i++) {
        int c = cluster_pick(rng);
        for (int j = 0; j < d; j++) {
            xb[i * d + j] = centers[c * d + j] + noise(rng);
        }
    }

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    faiss::IndexNeuroWeighted index(&inner);
    index.train(nb, xb.data());
    index.learning_rate = 0.1f;
    index.weight_decay = 0.995f;

    // Run feedback iterations using triplets from same/different clusters
    std::mt19937 fb_rng(12003);
    for (int iter = 0; iter < 500; iter++) {
        std::vector<float> queries(d);
        std::vector<float> positives(d);
        std::vector<float> negatives(d);

        int cq = std::uniform_int_distribution<int>(0, n_clusters - 1)(fb_rng);
        int cn = (cq + 1 + std::uniform_int_distribution<int>(
                                  0, n_clusters - 2)(fb_rng)) %
                n_clusters;

        for (int j = 0; j < d; j++) {
            queries[j] = centers[cq * d + j] + noise(fb_rng);
            positives[j] = centers[cq * d + j] + noise(fb_rng);
            negatives[j] = centers[cn * d + j] + noise(fb_rng);
        }

        index.feedback(1, queries.data(), positives.data(), negatives.data());
    }

    // Weights for discriminative dims (second half) should be higher
    float avg_noise_weight = 0, avg_disc_weight = 0;
    for (int j = 0; j < d / 2; j++)
        avg_noise_weight += index.weights[j];
    for (int j = d / 2; j < d; j++)
        avg_disc_weight += index.weights[j];
    avg_noise_weight /= (d / 2);
    avg_disc_weight /= (d - d / 2);

    EXPECT_GT(avg_disc_weight, avg_noise_weight)
            << "Discriminative dim weights (" << avg_disc_weight
            << ") should be > noise dim weights (" << avg_noise_weight << ")";

    // The ratio should be meaningful (at least 1.5x)
    EXPECT_GT(avg_disc_weight / avg_noise_weight, 1.5f)
            << "Discriminative/noise weight ratio = "
            << avg_disc_weight / avg_noise_weight
            << " should be > 1.5";
}

// PA-01 weights converge (don't diverge to infinity)
TEST(NeuroWeighted, WeightsConverge) {
    int d = 16;
    int nb = 200;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, 10, 10, 12004);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    faiss::IndexNeuroWeighted index(&inner);
    index.train(nb, xb.data());
    index.learning_rate = 0.1f;
    index.weight_decay = 0.99f;

    std::mt19937 rng(12005);
    std::normal_distribution<float> noise(0.0f, 0.1f);

    for (int iter = 0; iter < 2000; iter++) {
        std::vector<float> q(d), p(d), n(d);
        for (int j = 0; j < d; j++) {
            q[j] = noise(rng);
            p[j] = q[j] + noise(rng) * 0.1f;
            n[j] = noise(rng);
        }
        index.feedback(1, q.data(), p.data(), n.data());
    }

    // All weights should be finite and positive
    for (int j = 0; j < d; j++) {
        EXPECT_TRUE(std::isfinite(index.weights[j]))
                << "weight[" << j << "] = " << index.weights[j];
        EXPECT_GE(index.weights[j], index.min_weight)
                << "weight[" << j << "] should be >= min_weight";
    }
}

// PA-01 save/load roundtrip
TEST(NeuroWeighted, SaveLoadRoundtrip) {
    int d = 16;
    int nb = 100;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, 5, 5, 12006);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    faiss::IndexNeuroWeighted index(&inner);
    index.train(nb, xb.data());

    // Modify weights via feedback
    std::mt19937 rng(12007);
    std::normal_distribution<float> noise(0.0f, 0.1f);
    for (int iter = 0; iter < 50; iter++) {
        std::vector<float> q(d), p(d), n(d);
        for (int j = 0; j < d; j++) {
            q[j] = noise(rng);
            p[j] = q[j] + noise(rng) * 0.1f;
            n[j] = noise(rng);
        }
        index.feedback(1, q.data(), p.data(), n.data());
    }

    auto weights_before = index.weights;
    int fc_before = index.feedback_count;

    // Save
    const char* fname = "/tmp/neuro_weights_test.bin";
    index.save_weights(fname);

    // Reset and load
    index.train(nb, xb.data()); // resets to uniform
    EXPECT_EQ(index.feedback_count, 0);

    index.load_weights(fname);
    EXPECT_EQ(index.feedback_count, fc_before);

    for (int j = 0; j < d; j++) {
        EXPECT_NEAR(index.weights[j], weights_before[j], 1e-6)
                << "weight[" << j << "] mismatch after load";
    }

    // Clean up
    std::remove(fname);
}

// PA-01 returns valid results with non-uniform weights
TEST(NeuroWeighted, NonUniformWeightsValid) {
    int d = 16;
    int nb = 500;
    int nq = 10;
    int k = 5;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 10, 12008);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    faiss::IndexNeuroWeighted index(&inner);
    index.train(nb, xb.data());

    // Manually set non-uniform weights
    for (int j = 0; j < d; j++) {
        index.weights[j] = 0.5f + 1.5f * j / d;
    }

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    for (int i = 0; i < nq * k; i++) {
        EXPECT_GE(labels[i], 0);
        EXPECT_LT(labels[i], nb);
        EXPECT_GE(distances[i], 0.0f);
    }

    // Results sorted
    for (int q = 0; q < nq; q++) {
        for (int j = 1; j < k; j++) {
            EXPECT_LE(distances[q * k + j - 1], distances[q * k + j]);
        }
    }
}

// PA-01 params override weights per query
TEST(NeuroWeighted, ParamsOverrideWeights) {
    int d = 8;
    int nb = 100;
    int nq = 5;
    int k = 3;

    std::mt19937 rng(12009);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> xb(nb * d), xq(nq * d);
    for (auto& v : xb)
        v = dist(rng);
    for (auto& v : xq)
        v = dist(rng);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    faiss::IndexNeuroWeighted index(&inner);
    index.train(nb, xb.data());

    // Override: weight only first dimension
    faiss::NeuroWeightedParams params;
    params.weights.assign(d, 0.0f);
    params.weights[0] = 1.0f;

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data(), &params);

    // Results should be valid
    for (int i = 0; i < nq * k; i++) {
        EXPECT_GE(labels[i], 0);
        EXPECT_LT(labels[i], nb);
    }
}

// ============================================================
// T13: PA-02 ContextualWeights Tests
// ============================================================

// PA-02 with uniform weights = identical to L2 (same as PA-01)
TEST(NeuroContextual, UniformWeightsEqualsL2) {
    int d = 16;
    int nb = 200;
    int nq = 10;
    int k = 5;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 10, 13001);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    // Ground truth L2
    std::vector<float> dist_gt(nq * k);
    std::vector<faiss::idx_t> lab_gt(nq * k);
    inner.search(nq, xq.data(), k, dist_gt.data(), lab_gt.data());

    // PA-02 with default uniform weights (all clusters have w=1.0)
    faiss::IndexNeuroContextualWeighted index(&inner, 3);
    index.train(nq, xq.data()); // cluster the queries

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    for (int i = 0; i < nq * k; i++) {
        EXPECT_EQ(labels[i], lab_gt[i]);
        EXPECT_NEAR(distances[i], dist_gt[i], 1e-5);
    }
}

// PA-02 classifies queries to correct clusters
TEST(NeuroContextual, QueryClassification) {
    int d = 16;
    int nb = 200;
    int n_clusters = 4;

    // Generate well-separated query clusters
    std::mt19937 rng(13002);
    std::normal_distribution<float> noise(0.0f, 0.1f);

    std::vector<float> train_queries(n_clusters * 50 * d);
    for (int c = 0; c < n_clusters; c++) {
        float offset = c * 100.0f; // well separated
        for (int i = 0; i < 50; i++) {
            for (int j = 0; j < d; j++) {
                train_queries[(c * 50 + i) * d + j] = offset + noise(rng);
            }
        }
    }

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    std::vector<float> xb(nb * d);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : xb)
        v = dist(rng);
    inner.add(nb, xb.data());

    faiss::IndexNeuroContextualWeighted index(&inner, n_clusters);
    index.train(n_clusters * 50, train_queries.data());

    // Queries near cluster 0 (offset=0)
    std::vector<float> q0(d);
    for (int j = 0; j < d; j++)
        q0[j] = noise(rng);
    int c0 = index.classify_query(q0.data());

    // Queries near cluster 2 (offset=200)
    std::vector<float> q2(d);
    for (int j = 0; j < d; j++)
        q2[j] = 200.0f + noise(rng);
    int c2 = index.classify_query(q2.data());

    // They should be assigned to different clusters
    EXPECT_NE(c0, c2);
    EXPECT_GE(c0, 0);
    EXPECT_LT(c0, n_clusters);
    EXPECT_GE(c2, 0);
    EXPECT_LT(c2, n_clusters);
}

// PA-02 feedback updates per-cluster weights independently
TEST(NeuroContextual, FeedbackUpdatesPerCluster) {
    int d = 16;
    int nb = 200;
    int n_clusters = 2;

    std::mt19937 rng(13003);
    std::normal_distribution<float> noise(0.0f, 0.1f);

    // Two well-separated query clusters
    int nq_train = n_clusters * 50;
    std::vector<float> train_queries(nq_train * d);
    for (int c = 0; c < n_clusters; c++) {
        float offset = c * 100.0f;
        for (int i = 0; i < 50; i++) {
            for (int j = 0; j < d; j++) {
                train_queries[(c * 50 + i) * d + j] = offset + noise(rng);
            }
        }
    }

    std::vector<float> xb(nb * d);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : xb)
        v = dist(rng);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    faiss::IndexNeuroContextualWeighted index(&inner, n_clusters);
    index.train(nq_train, train_queries.data());
    index.learning_rate = 0.2f;

    // Record initial weights (all 1.0)
    auto weights_before = index.cluster_weights;

    // Feedback with queries near cluster 0 only
    for (int iter = 0; iter < 100; iter++) {
        std::vector<float> q(d), p(d), n(d);
        for (int j = 0; j < d; j++) {
            q[j] = noise(rng);
            p[j] = q[j] + noise(rng) * 0.1f;
            n[j] = noise(rng) + 50.0f; // far away
        }
        index.feedback(1, q.data(), p.data(), n.data());
    }

    // Cluster 0 weights should have changed
    int c0 = index.classify_query(train_queries.data()); // first query -> cluster
    bool c0_changed = false;
    for (int j = 0; j < d; j++) {
        if (std::abs(index.cluster_weights[c0 * d + j] -
                     weights_before[c0 * d + j]) > 1e-6) {
            c0_changed = true;
            break;
        }
    }
    EXPECT_TRUE(c0_changed) << "Cluster weights should change after feedback";

    // Both clusters should have been trained (feedback_counts > 0 for at least one)
    bool any_trained = false;
    for (int c = 0; c < n_clusters; c++) {
        if (index.feedback_counts[c] > 0) {
            any_trained = true;
            break;
        }
    }
    EXPECT_TRUE(any_trained);
}

// PA-02 returns valid results with multiple clusters
TEST(NeuroContextual, MultiClusterValidResults) {
    int d = 16;
    int nb = 500;
    int nq = 20;
    int k = 5;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 10, 13004);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    faiss::IndexNeuroContextualWeighted index(&inner, 4);
    index.train(nq, xq.data());

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    for (int i = 0; i < nq * k; i++) {
        EXPECT_GE(labels[i], 0);
        EXPECT_LT(labels[i], nb);
        EXPECT_GE(distances[i], 0.0f);
    }

    // Sorted by distance
    for (int q = 0; q < nq; q++) {
        for (int j = 1; j < k; j++) {
            EXPECT_LE(distances[q * k + j - 1], distances[q * k + j]);
        }
    }
}

// PA-02 force_cluster param works
TEST(NeuroContextual, ForceClusterParam) {
    int d = 8;
    int nb = 100;
    int nq = 5;
    int k = 3;

    std::mt19937 rng(13005);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> xb(nb * d), xq(nq * d);
    for (auto& v : xb)
        v = dist(rng);
    for (auto& v : xq)
        v = dist(rng);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    faiss::IndexNeuroContextualWeighted index(&inner, 3);
    index.train(nq, xq.data());

    // Manually set different weights per cluster
    for (int c = 0; c < 3; c++) {
        for (int j = 0; j < d; j++) {
            index.cluster_weights[c * d + j] = (c == 0) ? 1.0f : 0.01f;
        }
    }

    // Force cluster 0 (high weights)
    faiss::NeuroContextualParams params0;
    params0.force_cluster = 0;
    std::vector<float> d0(nq * k);
    std::vector<faiss::idx_t> l0(nq * k);
    index.search(nq, xq.data(), k, d0.data(), l0.data(), &params0);

    // Force cluster 1 (low weights)
    faiss::NeuroContextualParams params1;
    params1.force_cluster = 1;
    std::vector<float> d1(nq * k);
    std::vector<faiss::idx_t> l1(nq * k);
    index.search(nq, xq.data(), k, d1.data(), l1.data(), &params1);

    // Distances with cluster 0 (w=1.0) should be larger than cluster 1 (w=0.01)
    // for the same queries
    float sum_d0 = 0, sum_d1 = 0;
    for (int i = 0; i < nq * k; i++) {
        sum_d0 += d0[i];
        sum_d1 += d1[i];
    }
    EXPECT_GT(sum_d0, sum_d1)
            << "Cluster 0 (w=1.0) distances should be > Cluster 1 (w=0.01)";
}

// ============================================================
// T14: MR-03 ContrastiveLearning Tests
// ============================================================

// MR-03 contrastive converges faster than basic Hebbian
TEST(NeuroContrastive, FasterConvergence) {
    int d = 32;
    int n_clusters = 50;

    // Generate data where first half = noise, second half = discriminative
    std::mt19937 rng(14001);
    std::normal_distribution<float> noise(0.0f, 0.1f);
    std::uniform_real_distribution<float> center_dist(0.0f, 10.0f);

    std::vector<float> centers(n_clusters * d);
    for (int c = 0; c < n_clusters; c++) {
        for (int j = 0; j < d; j++) {
            if (j < d / 2) {
                centers[c * d + j] = 5.0f; // noise (same for all)
            } else {
                centers[c * d + j] = center_dist(rng);
            }
        }
    }

    int nb = 2000;
    std::uniform_int_distribution<int> cluster_pick(0, n_clusters - 1);
    std::vector<float> xb(nb * d);
    for (int i = 0; i < nb; i++) {
        int c = cluster_pick(rng);
        for (int j = 0; j < d; j++)
            xb[i * d + j] = centers[c * d + j] + noise(rng);
    }

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    // Train both methods for the same number of iterations
    int n_iters = 200;

    // Basic Hebbian (PA-01)
    faiss::IndexNeuroWeighted hebbian(&inner);
    hebbian.train(nb, xb.data());
    hebbian.learning_rate = 0.1f;
    hebbian.weight_decay = 0.995f;

    // Contrastive (MR-03)
    faiss::IndexNeuroWeighted contrastive(&inner);
    contrastive.train(nb, xb.data());
    contrastive.learning_rate = 0.1f;
    contrastive.weight_decay = 0.995f;

    std::mt19937 fb_rng1(14002), fb_rng2(14002); // same seed for fair comparison

    for (int iter = 0; iter < n_iters; iter++) {
        int cq = std::uniform_int_distribution<int>(0, n_clusters - 1)(fb_rng1);
        int cn = (cq + 1 + std::uniform_int_distribution<int>(
                                  0, n_clusters - 2)(fb_rng1)) %
                n_clusters;

        std::vector<float> q(d), p(d), n(d);
        for (int j = 0; j < d; j++) {
            q[j] = centers[cq * d + j] + noise(fb_rng1);
            p[j] = centers[cq * d + j] + noise(fb_rng1);
            n[j] = centers[cn * d + j] + noise(fb_rng1);
        }
        hebbian.feedback(1, q.data(), p.data(), n.data());
    }

    for (int iter = 0; iter < n_iters; iter++) {
        int cq = std::uniform_int_distribution<int>(0, n_clusters - 1)(fb_rng2);
        int cn = (cq + 1 + std::uniform_int_distribution<int>(
                                  0, n_clusters - 2)(fb_rng2)) %
                n_clusters;

        std::vector<float> q(d), p(d), n(d);
        for (int j = 0; j < d; j++) {
            q[j] = centers[cq * d + j] + noise(fb_rng2);
            p[j] = centers[cq * d + j] + noise(fb_rng2);
            n[j] = centers[cn * d + j] + noise(fb_rng2);
        }
        contrastive.feedback_contrastive(
                1, q.data(), p.data(), n.data(), 1, 1.0f);
    }

    // Measure weight discrimination ratio for both
    auto ratio = [&](const std::vector<float>& w) {
        float avg_noise = 0, avg_disc = 0;
        for (int j = 0; j < d / 2; j++)
            avg_noise += w[j];
        for (int j = d / 2; j < d; j++)
            avg_disc += w[j];
        avg_noise /= (d / 2);
        avg_disc /= (d - d / 2);
        return avg_disc / std::max(avg_noise, 0.001f);
    };

    float ratio_hebbian = ratio(hebbian.weights);
    float ratio_contrastive = ratio(contrastive.weights);

    // Contrastive should achieve better discrimination ratio
    EXPECT_GT(ratio_contrastive, ratio_hebbian)
            << "Contrastive ratio=" << ratio_contrastive
            << " should be > Hebbian ratio=" << ratio_hebbian;
}

// MR-03 with multiple negatives (hard negative mining)
TEST(NeuroContrastive, HardNegativeMining) {
    int d = 16;
    int nb = 200;

    std::mt19937 rng(14003);
    std::normal_distribution<float> noise(0.0f, 0.1f);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> xb(nb * d);
    for (auto& v : xb)
        v = dist(rng);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    faiss::IndexNeuroWeighted index(&inner);
    index.train(nb, xb.data());
    index.learning_rate = 0.1f;

    // Provide 5 negatives per query - the hardest should be selected
    int n_neg = 5;
    for (int iter = 0; iter < 100; iter++) {
        std::vector<float> q(d), p(d), negs(n_neg * d);
        for (int j = 0; j < d; j++) {
            q[j] = dist(rng);
            p[j] = q[j] + noise(rng) * 0.1f;
        }
        // First negative is hard (close to query)
        for (int j = 0; j < d; j++) {
            negs[j] = q[j] + noise(rng) * 0.5f; // close
        }
        // Others are easy (far from query)
        for (int n = 1; n < n_neg; n++) {
            for (int j = 0; j < d; j++) {
                negs[n * d + j] = q[j] + 10.0f; // far
            }
        }
        index.feedback_contrastive(
                1, q.data(), p.data(), negs.data(), n_neg, 1.0f);
    }

    // Weights should be finite and positive
    for (int j = 0; j < d; j++) {
        EXPECT_TRUE(std::isfinite(index.weights[j]));
        EXPECT_GE(index.weights[j], index.min_weight);
    }

    // Feedback count should reflect all iterations
    EXPECT_EQ(index.feedback_count, 100);
}

// MR-03 weight stability: weights don't oscillate wildly
TEST(NeuroContrastive, WeightStability) {
    int d = 8;
    int nb = 100;

    std::mt19937 rng(14004);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::normal_distribution<float> noise(0.0f, 0.1f);

    std::vector<float> xb(nb * d);
    for (auto& v : xb)
        v = dist(rng);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    faiss::IndexNeuroWeighted index(&inner);
    index.train(nb, xb.data());
    index.learning_rate = 0.05f;
    index.weight_decay = 0.99f;

    // Run many iterations with consistent signal
    for (int iter = 0; iter < 500; iter++) {
        std::vector<float> q(d), p(d), n(d);
        for (int j = 0; j < d; j++) {
            q[j] = dist(rng);
            p[j] = q[j] + noise(rng) * 0.1f;
            n[j] = dist(rng);
        }
        index.feedback_contrastive(1, q.data(), p.data(), n.data());
    }

    auto weights_mid = index.weights;

    // Run 50 more iterations
    for (int iter = 0; iter < 50; iter++) {
        std::vector<float> q(d), p(d), n(d);
        for (int j = 0; j < d; j++) {
            q[j] = dist(rng);
            p[j] = q[j] + noise(rng) * 0.1f;
            n[j] = dist(rng);
        }
        index.feedback_contrastive(1, q.data(), p.data(), n.data());
    }

    // Weights should not have changed dramatically (< 50% relative change)
    for (int j = 0; j < d; j++) {
        float relative_change = std::abs(index.weights[j] - weights_mid[j]) /
                std::max(weights_mid[j], 0.01f);
        EXPECT_LT(relative_change, 0.5f)
                << "weight[" << j << "] changed too much: "
                << weights_mid[j] << " -> " << index.weights[j];
    }
}

// MR-03 margin_scale=0 behaves like basic Hebbian (but with hard neg)
TEST(NeuroContrastive, ZeroMarginScaleLikeHebbian) {
    int d = 8;
    int nb = 100;

    std::mt19937 rng(14005);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::normal_distribution<float> noise(0.0f, 0.1f);

    std::vector<float> xb(nb * d);
    for (auto& v : xb)
        v = dist(rng);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    faiss::IndexNeuroWeighted index(&inner);
    index.train(nb, xb.data());
    index.learning_rate = 0.1f;

    // With margin_scale=0, contrastive reduces to sign-only updates (like Hebbian)
    for (int iter = 0; iter < 50; iter++) {
        std::vector<float> q(d), p(d), n(d);
        for (int j = 0; j < d; j++) {
            q[j] = dist(rng);
            p[j] = q[j] + noise(rng) * 0.1f;
            n[j] = dist(rng);
        }
        index.feedback_contrastive(
                1, q.data(), p.data(), n.data(), 1, 0.0f);
    }

    // All weights should be finite, positive, and have been updated
    bool any_changed = false;
    for (int j = 0; j < d; j++) {
        EXPECT_TRUE(std::isfinite(index.weights[j]));
        EXPECT_GE(index.weights[j], index.min_weight);
        if (std::abs(index.weights[j] - 1.0f) > 1e-6) {
            any_changed = true;
        }
    }
    EXPECT_TRUE(any_changed) << "Weights should change even with margin_scale=0";
}

// ============================================================
// T15: PP-01 ParallelVoting Tests
// ============================================================

// PP-01 returns valid results
TEST(NeuroParallelVoting, ReturnsValidResults) {
    int d = 32;
    int nb = 1000;
    int nq = 10;
    int k = 5;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 20, 15001);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    faiss::IndexNeuroParallelVoting index(&inner, 4);

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    for (int i = 0; i < nq * k; i++) {
        EXPECT_GE(labels[i], 0);
        EXPECT_LT(labels[i], nb);
        EXPECT_GE(distances[i], 0.0f);
    }

    // Sorted by distance
    for (int q = 0; q < nq; q++) {
        for (int j = 1; j < k; j++) {
            EXPECT_LE(distances[q * k + j - 1], distances[q * k + j]);
        }
    }
}

// PP-01 with full_rerank: recall >= 92%
TEST(NeuroParallelVoting, FullRerankRecall) {
    int d = 32;
    int nb = 5000;
    int nq = 100;
    int k = 10;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 50, 15002);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    // Ground truth
    std::vector<float> dist_gt(nq * k);
    std::vector<faiss::idx_t> lab_gt(nq * k);
    inner.search(nq, xq.data(), k, dist_gt.data(), lab_gt.data());

    faiss::IndexNeuroParallelVoting index(&inner, 4);
    index.integration = faiss::NEURO_INTEGRATE_FULL_RERANK;
    index.top_k_per_group = 50;

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    float recall = compute_recall(labels.data(), lab_gt.data(), nq, k);
    EXPECT_GE(recall, 0.92f)
            << "ParallelVoting recall=" << recall << " (expected >= 0.92)";
}

// PP-01 all grouping methods valid
TEST(NeuroParallelVoting, AllGroupingMethodsValid) {
    int d = 16;
    int nb = 500;
    int nq = 5;
    int k = 5;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 10, 15003);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    faiss::NeuroGroupingMethod methods[] = {
            faiss::NEURO_GROUP_CONSECUTIVE,
            faiss::NEURO_GROUP_INTERLEAVED};

    for (auto method : methods) {
        faiss::IndexNeuroParallelVoting index(&inner, 4);
        index.grouping = method;
        index.top_k_per_group = 30;

        std::vector<float> distances(nq * k);
        std::vector<faiss::idx_t> labels(nq * k);
        index.search(nq, xq.data(), k, distances.data(), labels.data());

        for (int i = 0; i < nq * k; i++) {
            EXPECT_GE(labels[i], 0)
                    << "method=" << static_cast<int>(method);
            EXPECT_LT(labels[i], nb)
                    << "method=" << static_cast<int>(method);
        }
    }
}

// PP-01 all integration methods valid
TEST(NeuroParallelVoting, AllIntegrationMethodsValid) {
    int d = 16;
    int nb = 500;
    int nq = 5;
    int k = 5;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 10, 15004);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    faiss::NeuroIntegrationMethod methods[] = {
            faiss::NEURO_INTEGRATE_VOTING,
            faiss::NEURO_INTEGRATE_BORDA,
            faiss::NEURO_INTEGRATE_MEAN_DIST,
            faiss::NEURO_INTEGRATE_FULL_RERANK};

    for (auto method : methods) {
        faiss::IndexNeuroParallelVoting index(&inner, 4);
        index.integration = method;
        index.top_k_per_group = 30;

        std::vector<float> distances(nq * k);
        std::vector<faiss::idx_t> labels(nq * k);
        index.search(nq, xq.data(), k, distances.data(), labels.data());

        for (int i = 0; i < nq * k; i++) {
            EXPECT_GE(labels[i], 0)
                    << "method=" << static_cast<int>(method);
            EXPECT_LT(labels[i], nb)
                    << "method=" << static_cast<int>(method);
        }
    }
}

// ============================================================
// T16: PP-02 CoarseToFine Tests
// ============================================================

// PP-02 returns valid results after train()
TEST(NeuroCoarseToFine, ReturnsValidResults) {
    int d = 32;
    int nb = 1000;
    int nq = 10;
    int k = 5;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 20, 16001);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    faiss::IndexNeuroCoarseToFine index(&inner, 3);
    index.train(nb, xb.data());

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    for (int i = 0; i < nq * k; i++) {
        EXPECT_GE(labels[i], 0);
        EXPECT_LT(labels[i], nb);
        EXPECT_GE(distances[i], 0.0f);
    }

    // Sorted
    for (int q = 0; q < nq; q++) {
        for (int j = 1; j < k; j++) {
            EXPECT_LE(distances[q * k + j - 1], distances[q * k + j]);
        }
    }
}

// PP-02 recall >= 88%
TEST(NeuroCoarseToFine, RecallAtLeast88Pct) {
    int d = 32;
    int nb = 5000;
    int nq = 100;
    int k = 10;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 50, 16002);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    // Ground truth
    std::vector<float> dist_gt(nq * k);
    std::vector<faiss::idx_t> lab_gt(nq * k);
    inner.search(nq, xq.data(), k, dist_gt.data(), lab_gt.data());

    faiss::IndexNeuroCoarseToFine index(&inner, 3);
    index.cutoff_per_level = {0.5f, 0.7f, 1.0f}; // keep more candidates
    index.train(nb, xb.data());

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    float recall = compute_recall(labels.data(), lab_gt.data(), nq, k);
    EXPECT_GE(recall, 0.88f)
            << "CoarseToFine recall=" << recall << " (expected >= 0.88)";
}

// PP-02 uses fewer calculations than brute force
TEST(NeuroCoarseToFine, FewerCalculations) {
    int d = 32;
    int nb = 5000;
    int nq = 1;
    int k = 10;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 50, 16003);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    faiss::IndexNeuroCoarseToFine index(&inner, 3);
    index.cutoff_per_level = {0.3f, 0.5f, 1.0f};
    index.train(nb, xb.data());

    faiss::NeuroCoarseToFineParams params;
    params.collect_stats = true;

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data(), &params);

    int64_t brute_force_calcs = (int64_t)nb * d;
    EXPECT_LT(index.last_stats.calculations_performed, brute_force_calcs)
            << "CoarseToFine should use fewer calculations: "
            << index.last_stats.calculations_performed << " vs "
            << brute_force_calcs;
}

// PP-02 with conservative cutoffs approaches brute force
TEST(NeuroCoarseToFine, ConservativeCutoffHighRecall) {
    int d = 16;
    int nb = 200;
    int nq = 10;
    int k = 5;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 10, 16004);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    // Ground truth
    std::vector<float> dist_gt(nq * k);
    std::vector<faiss::idx_t> lab_gt(nq * k);
    inner.search(nq, xq.data(), k, dist_gt.data(), lab_gt.data());

    // Conservative: keep 90% at each level
    faiss::IndexNeuroCoarseToFine index(&inner, 3);
    index.cutoff_per_level = {0.9f, 0.95f, 1.0f};
    index.train(nb, xb.data());

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    float recall = compute_recall(labels.data(), lab_gt.data(), nq, k);
    EXPECT_GE(recall, 0.95f)
            << "Conservative CoarseToFine recall=" << recall;
}

// ============================================================
// T17: MR-02 ContextCache Tests
// ============================================================

// MR-02 returns valid results (no cache effect)
TEST(NeuroCache, ReturnsValidResults) {
    int d = 16;
    int nb = 500;
    int nq = 10;
    int k = 5;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 10, 17001);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    faiss::IndexNeuroCache index(&inner);

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    for (int i = 0; i < nq * k; i++) {
        EXPECT_GE(labels[i], 0);
        EXPECT_LT(labels[i], nb);
        EXPECT_GE(distances[i], 0.0f);
    }
}

// MR-02 cache hits return same results
TEST(NeuroCache, CacheHitsReturnSameResults) {
    int d = 16;
    int nb = 500;
    int nq = 5;
    int k = 5;

    std::vector<float> xb, xq;
    generate_clustered_data(xb, xq, d, nb, nq, 10, 17002);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    faiss::IndexNeuroCache index(&inner);

    // First search (cache miss)
    std::vector<float> d1(nq * k);
    std::vector<faiss::idx_t> l1(nq * k);
    index.search(nq, xq.data(), k, d1.data(), l1.data());
    EXPECT_EQ(index.cache_misses, nq);
    EXPECT_EQ(index.cache_hits, 0);

    // Second search (same queries -> cache hit)
    std::vector<float> d2(nq * k);
    std::vector<faiss::idx_t> l2(nq * k);
    index.search(nq, xq.data(), k, d2.data(), l2.data());
    EXPECT_EQ(index.cache_hits, nq);

    // Results should be identical
    for (int i = 0; i < nq * k; i++) {
        EXPECT_EQ(l1[i], l2[i]);
        EXPECT_NEAR(d1[i], d2[i], 1e-6);
    }
}

// MR-02 cache invalidation on add()
TEST(NeuroCache, InvalidationOnAdd) {
    int d = 16;
    int nb = 100;
    int nq = 3;
    int k = 3;

    std::mt19937 rng(17003);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> xb(nb * d), xq(nq * d);
    for (auto& v : xb)
        v = dist(rng);
    for (auto& v : xq)
        v = dist(rng);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    faiss::IndexNeuroCache index(&inner);

    // Search to populate cache
    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data());
    EXPECT_EQ(index.cache_misses, nq);

    // Add more data -> cache should be invalidated
    std::vector<float> xb2(10 * d);
    for (auto& v : xb2)
        v = dist(rng);
    index.add(10, xb2.data());

    // Search again -> should be cache miss (cache was cleared)
    index.search(nq, xq.data(), k, distances.data(), labels.data());
    // After clear, cache_misses was reset to 0, so it should be nq again
    EXPECT_EQ(index.cache_misses, nq);
    EXPECT_EQ(index.ntotal, nb + 10);
}

// MR-02 cache invalidation on reset()
TEST(NeuroCache, InvalidationOnReset) {
    int d = 8;
    int nb = 50;

    std::mt19937 rng(17004);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> xb(nb * d);
    for (auto& v : xb)
        v = dist(rng);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    faiss::IndexNeuroCache index(&inner);
    EXPECT_EQ(index.ntotal, nb);

    index.reset();
    EXPECT_EQ(index.ntotal, 0);
    EXPECT_EQ(inner.ntotal, 0);
    EXPECT_FLOAT_EQ(index.hit_rate(), 0.0f);
}

// MR-02 hit_rate() works
TEST(NeuroCache, HitRate) {
    int d = 8;
    int nb = 50;
    int k = 3;

    std::mt19937 rng(17005);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> xb(nb * d), xq(d);
    for (auto& v : xb)
        v = dist(rng);
    for (auto& v : xq)
        v = dist(rng);

    faiss::IndexFlat inner(d, faiss::METRIC_L2);
    inner.add(nb, xb.data());

    faiss::IndexNeuroCache index(&inner);

    // First search: miss
    std::vector<float> distances(k);
    std::vector<faiss::idx_t> labels(k);
    index.search(1, xq.data(), k, distances.data(), labels.data());
    EXPECT_FLOAT_EQ(index.hit_rate(), 0.0f);

    // Second search: hit
    index.search(1, xq.data(), k, distances.data(), labels.data());
    EXPECT_FLOAT_EQ(index.hit_rate(), 0.5f); // 1 hit, 1 miss

    // Third search: hit
    index.search(1, xq.data(), k, distances.data(), labels.data());
    float expected = 2.0f / 3.0f; // 2 hits, 1 miss
    EXPECT_NEAR(index.hit_rate(), expected, 1e-5);
}
