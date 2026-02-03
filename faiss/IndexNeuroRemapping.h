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

/** HP-04: Context Remapping Decorator.
 *
 * Inspired by hippocampal remapping where the same neurons
 * represent different information in different contexts.
 *
 * Maintains multiple sub-indices (one per context) and optionally
 * a transfer matrix for cross-context search.
 *
 * Use cases:
 *   - Multi-domain search (same embedding space, different domains)
 *   - Temporal contexts (same objects at different times)
 *   - Environment-specific search (indoor vs outdoor)
 *
 * This is a DECORATOR pattern: wraps multiple indices.
 */
struct IndexNeuroRemapping : Index {
    /// Number of contexts
    int n_contexts = 0;

    /// Sub-indices (one per context)
    std::vector<Index*> context_indices;

    /// Whether to delete sub-indices in destructor
    bool own_fields = false;

    /// Current active context for add/search
    int active_context = 0;

    /// Transfer matrix: n_contexts * n_contexts
    /// transfer[i * n_contexts + j] = weight for transferring from context i to j
    std::vector<float> transfer_matrix;

    /// Whether to enable cross-context search
    bool cross_context_search = false;

    /// Learning rate for transfer matrix updates
    float learning_rate = 0.1f;

    IndexNeuroRemapping() = default;

    /** Construct with multiple context indices.
     * @param context_indices  vector of indices (one per context)
     * @param own_fields       take ownership of indices
     */
    IndexNeuroRemapping(
            const std::vector<Index*>& context_indices,
            bool own_fields = false);

    /** Construct with single index type, replicated per context.
     * @param template_index  template to clone for each context
     * @param n_contexts      number of contexts
     * @param own_fields      take ownership
     */
    IndexNeuroRemapping(Index* template_index, int n_contexts, bool own_fields = false);

    void train(idx_t n, const float* x) override;

    /// Add vectors to active context
    void add(idx_t n, const float* x) override;

    /// Add vectors to specific context
    void add_to_context(int context, idx_t n, const float* x);

    void reset() override;

    /// Reset specific context
    void reset_context(int context);

    void reconstruct(idx_t key, float* recons) const override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /// Search in specific context
    void search_context(
            int context,
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const;

    /// Search across all contexts (returns combined results)
    void search_all_contexts(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            std::vector<int>* result_contexts = nullptr,
            const SearchParameters* params = nullptr) const;

    /// Set active context
    void set_active_context(int context);

    /// Get active context
    int get_active_context() const { return active_context; }

    /// Get number of vectors in a context
    idx_t ntotal_context(int context) const;

    /** Learn transfer weights from cross-context pairs.
     *
     * @param n_pairs     number of training pairs
     * @param ctx_from    source context for each pair (n_pairs)
     * @param ctx_to      target context for each pair (n_pairs)
     * @param vec_from    source vectors (n_pairs * d)
     * @param vec_to      target vectors (n_pairs * d)
     */
    void learn_transfer(
            idx_t n_pairs,
            const int* ctx_from,
            const int* ctx_to,
            const float* vec_from,
            const float* vec_to);

    ~IndexNeuroRemapping() override;

private:
    /// Encode global label from context and local label
    idx_t encode_label(int context, idx_t local_label) const;

    /// Decode global label to context and local label
    void decode_label(idx_t global_label, int& context, idx_t& local_label) const;
};

} // namespace faiss
