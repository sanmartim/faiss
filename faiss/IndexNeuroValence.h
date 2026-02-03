/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <vector>

namespace faiss {

/** DR-03: Valence Modulation Decorator.
 *
 * Wraps any Index and applies learned valence (emotion-like) weights
 * to queries before delegating search to the sub-index.
 *
 * Inspired by how Drosophila MBON compartments encode different
 * valences (approach vs avoid) that modulate behavior.
 *
 * Use cases:
 *   - Preference learning: upweight certain feature dimensions
 *   - Aversion learning: downweight dimensions associated with negatives
 *   - Multi-objective search: different valence vectors for different goals
 *
 * This is a DECORATOR pattern: wraps any Index, not just IndexNeuro.
 */
struct IndexNeuroValence : Index {
    /// The wrapped index
    Index* sub_index = nullptr;

    /// Whether to delete sub_index in destructor
    bool own_fields = false;

    /// Number of valence types
    int n_valences = 2;

    /// Current active valence (0 = first valence)
    int active_valence = 0;

    /// Valence weight vectors: n_valences * d
    std::vector<float> valence_weights;

    /// Learning rate for valence updates
    float learning_rate = 0.1f;

    /// Weight decay
    float weight_decay = 0.99f;

    /// Transformation mode: "multiply" or "add"
    /// multiply: query[j] *= valence[j]
    /// add: query[j] += valence[j]
    bool multiply_mode = true;

    IndexNeuroValence() = default;

    /** Construct wrapping any Index.
     * @param sub_index   the index to wrap
     * @param n_valences  number of valence types (default 2: approach/avoid)
     * @param own_fields  take ownership of sub_index
     */
    IndexNeuroValence(Index* sub_index, int n_valences = 2, bool own_fields = false);

    void train(idx_t n, const float* x) override;
    void add(idx_t n, const float* x) override;
    void reset() override;
    void reconstruct(idx_t key, float* recons) const override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /// Set the active valence for subsequent searches
    void set_active_valence(int valence);

    /// Get the weight vector for a specific valence
    float* get_valence_weights(int valence);
    const float* get_valence_weights(int valence) const;

    /** Learn valence from examples.
     *
     * @param valence     which valence to update
     * @param n           number of example vectors
     * @param positive    vectors to approach (increase weights)
     * @param negative    vectors to avoid (decrease weights)
     */
    void learn_valence(
            int valence,
            idx_t n,
            const float* positive,
            const float* negative);

    /** Single-example incremental update.
     *
     * @param valence     which valence to update
     * @param example     the example vector
     * @param is_positive true = approach, false = avoid
     */
    void update_valence(int valence, const float* example, bool is_positive);

    /// Reset a valence to neutral (uniform weights)
    void reset_valence(int valence);

    /// Reset all valences to neutral
    void reset_all_valences();

    ~IndexNeuroValence() override;

private:
    /// Apply valence transformation to queries
    void apply_valence(
            const float* queries,
            idx_t n,
            std::vector<float>& transformed) const;
};

} // namespace faiss
