/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/NeuroDistance.h>
#include <faiss/impl/FaissAssert.h>

namespace faiss {

IndexNeuro::IndexNeuro(Index* inner_index, bool own_inner)
        : Index(inner_index->d, inner_index->metric_type),
          inner_index(inner_index),
          own_inner(own_inner) {
    ntotal = inner_index->ntotal;
    is_trained = inner_index->is_trained;
    metric_arg = inner_index->metric_arg;
}

void IndexNeuro::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");
    inner_index->add(n, x);
    ntotal = inner_index->ntotal;
    is_trained = inner_index->is_trained;
}

void IndexNeuro::reset() {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");
    inner_index->reset();
    ntotal = 0;
}

void IndexNeuro::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is not set");
    inner_index->reconstruct(key, recons);
}

IndexNeuro::~IndexNeuro() {
    if (own_inner) {
        delete inner_index;
    }
}

} // namespace faiss
