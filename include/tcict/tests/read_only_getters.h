#pragma once

#include <tcict/assertion.h>
#include <tcict/elem_helper.h>
#include <tcict/fixture.h>
#include <tcict/skip.h>

#include <cmath>

namespace tcict { namespace tests {

// --- order ---

template <typename TenT>
void test_order(tci_test_fixture<TenT>& fix) {
#ifndef TCICT_SKIP_ORDER
  auto& ctx = fix.context();
  auto tensor = tci::zeros<TenT>(ctx, {2, 3, 4});
  TCICT_ASSERT(tci::order(ctx, tensor) == 3);
#else
  (void)fix;
#endif
}

// --- shape ---

template <typename TenT>
void test_shape(tci_test_fixture<TenT>& fix) {
#ifndef TCICT_SKIP_SHAPE
  auto& ctx = fix.context();
  auto tensor = tci::zeros<TenT>(ctx, {2, 3, 4});
  auto result_shape = tci::shape(ctx, tensor);
  TCICT_ASSERT(result_shape.size() == 3);
  TCICT_ASSERT(result_shape[0] == 2);
  TCICT_ASSERT(result_shape[1] == 3);
  TCICT_ASSERT(result_shape[2] == 4);
#else
  (void)fix;
#endif
}

// --- size ---

template <typename TenT>
void test_size(tci_test_fixture<TenT>& fix) {
#ifndef TCICT_SKIP_SIZE
  auto& ctx = fix.context();
  auto tensor = tci::zeros<TenT>(ctx, {2, 3, 4});
  TCICT_ASSERT(tci::size(ctx, tensor) == 24);
#else
  (void)fix;
#endif
}

// --- size_bytes ---

template <typename TenT>
void test_size_bytes(tci_test_fixture<TenT>& fix) {
#ifndef TCICT_SKIP_SIZE_BYTES
  auto& ctx = fix.context();
  auto tensor = tci::zeros<TenT>(ctx, {2, 3, 4});
  auto bytes = tci::size_bytes(ctx, tensor);
  TCICT_ASSERT(bytes > 0);
  // Verify: 24 elements * sizeof(elem_t<TenT>)
  TCICT_ASSERT(bytes == 24 * sizeof(tci::elem_t<TenT>));
#else
  (void)fix;
#endif
}

// --- get_elem / set_elem ---

template <typename TenT>
void test_set_get_elem(tci_test_fixture<TenT>& fix) {
#if !defined(TCICT_SKIP_GET_ELEM) && !defined(TCICT_SKIP_SET_ELEM)
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  auto tensor = tci::zeros<TenT>(ctx, {2, 2});

  tci::elem_coors_t<TenT> coord = {0, 0};
  auto expected = make_elem<TenT>(2.5, 1.5);
  tci::set_elem(ctx, tensor, coord, expected);

  auto retrieved = tci::get_elem(ctx, tensor, coord);
  TCICT_ASSERT_CLOSE(real_part<TenT>(retrieved), real_part<TenT>(expected), eps);
  if constexpr (is_complex_v<TenT>) {
    TCICT_ASSERT_CLOSE(imag_part<TenT>(retrieved), imag_part<TenT>(expected), eps);
  }
#else
  (void)fix;
#endif
}

// --- size_bytes for small tensor ---

template <typename TenT>
void test_size_bytes_2x2(tci_test_fixture<TenT>& fix) {
#ifndef TCICT_SKIP_SIZE_BYTES
  auto& ctx = fix.context();
  auto tensor = tci::zeros<TenT>(ctx, {2, 2});
  auto size = tci::size_bytes(ctx, tensor);
  TCICT_ASSERT(size > 0);
#else
  (void)fix;
#endif
}

}}  // namespace tcict::tests

// Bulk registration helper: invokes X(..., "category", test_fn) once per test.
// See include/tcict/adapters/doctest.h for usage.
#define TCICT_FOREACH_READ_ONLY_GETTERS_TEST_ALL_TYPES(X, ...) \
  X(__VA_ARGS__, "read_only_getters", test_order) \
  X(__VA_ARGS__, "read_only_getters", test_shape) \
  X(__VA_ARGS__, "read_only_getters", test_size) \
  X(__VA_ARGS__, "read_only_getters", test_size_bytes) \
  X(__VA_ARGS__, "read_only_getters", test_set_get_elem) \
  X(__VA_ARGS__, "read_only_getters", test_size_bytes_2x2)
