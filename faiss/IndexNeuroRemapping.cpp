/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroRemapping.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/clone_index.h>

#include <algorithm>
#include <cmath>

namespace faiss {

IndexNeuroRemapping::IndexNeuroRemapping(
        const std::vector<Index*>& indices,
        bool own_fields)
        : Index(indices.empty() ? 0 : indices[0]->d,
                indices.empty() ? METRIC_L2 : indices[0]->metric_type),
          n_contexts(static_cast<int>(indices.size())),
          context_indices(indices),
          own_fields(own_fields) {
    // Initialize transfer matrix to identity
    transfer_matrix.assign(n_contexts * n_contexts, 0.0f);
    for (int i = 0; i < n_contexts; i++) {
        transfer_matrix[i * n_contexts + i] = 1.0f;
    }

    // Compute total
    ntotal = 0;
    for (auto* idx : context_indices) {
        ntotal += idx->ntotal;
    }

    is_trained = true;
    for (auto* idx : context_indices) {
        if (!idx->is_trained) {
            is_trained = false;
            break;
        }
    }
}

IndexNeuroRemapping::IndexNeuroRemapping(
        Index* template_index,
        int n_contexts,
        bool own_fields)
        : Index(template_index->d, template_index->metric_type),
          n_contexts(n_contexts),
          own_fields(true) {  // Always own cloned indices
    context_indices.resize(n_contexts);
    for (int c = 0; c < n_contexts; c++) {
        context_indices[c] = clone_index(template_index);
        context_indices[c]->reset();
    }

    // Initialize transfer matrix to identity
    transfer_matrix.assign(n_contexts * n_contexts, 0.0f);
    for (int i = 0; i < n_contexts; i++) {
        transfer_matrix[i * n_contexts + i] = 1.0f;
    }

    ntotal = 0;
    is_trained = template_index->is_trained;

    // If we don't own the template, and own_fields was false, clean up
    // (but we always own cloned indices)
    (void)own_fields;
}

IndexNeuroRemapping::~IndexNeuroRemapping() {
    if (own_fields) {
        for (auto* idx : context_indices) {
            delete idx;
        }
    }
}

void IndexNeuroRemapping::train(idx_t n, const float* x) {
    for (auto* idx : context_indices) {
        idx->train(n, x);
    }
    is_trained = true;
}

void IndexNeuroRemapping::set_active_context(int context) {
    FAISS_THROW_IF_NOT_FMT(
            context >= 0 && context < n_contexts,
            "context %d out of range [0, %d)",
            context,
            n_contexts);
    active_context = context;
}

void IndexNeuroRemapping::add(idx_t n, const float* x) {
    add_to_context(active_context, n, x);
}

void IndexNeuroRemapping::add_to_context(int context, idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_FMT(
            context >= 0 && context < n_contexts,
            "context %d out of range [0, %d)",
            context,
            n_contexts);
    context_indices[context]->add(n, x);
    ntotal += n;
}

void IndexNeuroRemapping::reset() {
    for (auto* idx : context_indices) {
        idx->reset();
    }
    ntotal = 0;
}

void IndexNeuroRemapping::reset_context(int context) {
    FAISS_THROW_IF_NOT_FMT(
            context >= 0 && context < n_contexts,
            "context %d out of range [0, %d)",
            context,
            n_contexts);
    ntotal -= context_indices[context]->ntotal;
    context_indices[context]->reset();
}

idx_t IndexNeuroRemapping::ntotal_context(int context) const {
    FAISS_THROW_IF_NOT_FMT(
            context >= 0 && context < n_contexts,
            "context %d out of range [0, %d)",
            context,
            n_contexts);
    return context_indices[context]->ntotal;
}

idx_t IndexNeuroRemapping::encode_label(int context, idx_t local_label) const {
    // Encode as: context * max_per_context + local_label
    // Using a large multiplier to avoid collisions
    return static_cast<idx_t>(context) * 1000000000LL + local_label;
}

void IndexNeuroRemapping::decode_label(
        idx_t global_label,
        int& context,
        idx_t& local_label) const {
    context = static_cast<int>(global_label / 1000000000LL);
    local_label = global_label % 1000000000LL;
}

void IndexNeuroRemapping::reconstruct(idx_t key, float* recons) const {
    int context;
    idx_t local_label;
    decode_label(key, context, local_label);
    FAISS_THROW_IF_NOT_FMT(
            context >= 0 && context < n_contexts,
            "context %d out of range",
            context);
    context_indices[context]->reconstruct(local_label, recons);
}

void IndexNeuroRemapping::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    if (cross_context_search) {
        search_all_contexts(n, x, k, distances, labels, nullptr, params);
    } else {
        search_context(active_context, n, x, k, distances, labels, params);
    }
}

void IndexNeuroRemapping::search_context(
        int context,
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_FMT(
            context >= 0 && context < n_contexts,
            "context %d out of range [0, %d)",
            context,
            n_contexts);

    std::vector<float> ctx_distances(n * k);
    std::vector<idx_t> ctx_labels(n * k);

    context_indices[context]->search(
            n, x, k, ctx_distances.data(), ctx_labels.data(), params);

    // Encode labels with context
    for (idx_t i = 0; i < n * k; i++) {
        distances[i] = ctx_distances[i];
        if (ctx_labels[i] >= 0) {
            labels[i] = encode_label(context, ctx_labels[i]);
        } else {
            labels[i] = -1;
        }
    }
}

void IndexNeuroRemapping::search_all_contexts(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        std::vector<int>* result_contexts,
        const SearchParameters* params) const {
    // Search each context and merge results
    std::vector<std::vector<std::pair<float, idx_t>>> all_results(n);

    for (int c = 0; c < n_contexts; c++) {
        if (context_indices[c]->ntotal == 0) continue;

        std::vector<float> ctx_distances(n * k);
        std::vector<idx_t> ctx_labels(n * k);

        context_indices[c]->search(
                n, x, k, ctx_distances.data(), ctx_labels.data(), params);

        // Apply transfer weight from active context to this context
        float transfer_weight = transfer_matrix[active_context * n_contexts + c];

        for (idx_t q = 0; q < n; q++) {
            for (idx_t i = 0; i < k; i++) {
                idx_t local_label = ctx_labels[q * k + i];
                if (local_label >= 0) {
                    float dist = ctx_distances[q * k + i] / (transfer_weight + 1e-10f);
                    all_results[q].emplace_back(dist, encode_label(c, local_label));
                }
            }
        }
    }

    // Sort and output top-k per query
    if (result_contexts) {
        result_contexts->resize(n * k);
    }

    for (idx_t q = 0; q < n; q++) {
        auto& results = all_results[q];
        std::sort(results.begin(), results.end());

        size_t actual_k = std::min(static_cast<size_t>(k), results.size());
        for (size_t i = 0; i < actual_k; i++) {
            distances[q * k + i] = results[i].first;
            labels[q * k + i] = results[i].second;
            if (result_contexts) {
                int ctx;
                idx_t local;
                decode_label(results[i].second, ctx, local);
                (*result_contexts)[q * k + i] = ctx;
            }
        }
        for (size_t i = actual_k; i < static_cast<size_t>(k); i++) {
            distances[q * k + i] = std::numeric_limits<float>::max();
            labels[q * k + i] = -1;
            if (result_contexts) {
                (*result_contexts)[q * k + i] = -1;
            }
        }
    }
}

void IndexNeuroRemapping::learn_transfer(
        idx_t n_pairs,
        const int* ctx_from,
        const int* ctx_to,
        const float* vec_from,
        const float* vec_to) {
    // Learn transfer weights based on similarity of cross-context pairs
    // If vec_from in ctx_from is similar to vec_to in ctx_to,
    // increase transfer[ctx_from, ctx_to]

    for (idx_t p = 0; p < n_pairs; p++) {
        int cf = ctx_from[p];
        int ct = ctx_to[p];
        FAISS_THROW_IF_NOT_FMT(
                cf >= 0 && cf < n_contexts && ct >= 0 && ct < n_contexts,
                "context out of range in pair %ld",
                p);

        // Compute similarity of the pair
        const float* vf = vec_from + p * d;
        const float* vt = vec_to + p * d;

        float dist_sq = 0.0f;
        for (int j = 0; j < d; j++) {
            float diff = vf[j] - vt[j];
            dist_sq += diff * diff;
        }

        // Convert distance to similarity (exponential decay)
        float similarity = std::exp(-dist_sq);

        // Update transfer matrix
        transfer_matrix[cf * n_contexts + ct] += learning_rate * similarity;
        transfer_matrix[cf * n_contexts + ct] *= 0.99f;  // Decay
    }
}

} // namespace faiss
