#pragma once

#include <tcict/assertion.h>
#include <tcict/elem_helper.h>
#include <tcict/fixture.h>
#include <tcict/skip.h>
#include <tcict/zeroth_order.h>

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
  // V1's entry for it is "Reports memory consumption in bytes": no formula, no
  // lower bound. Callable-and-returns is the whole of what is portably
  // assertable, so what this test and its second-order and zeroth-order
  // counterparts below give between them is that none of the three ranks makes
  // the call fail.
  TCICT_ASSERT_NOTHROW(tci::size_bytes(ctx, tensor));
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

// --- get_elem / set_elem: zeroth-order ---

// TCAPI V1: for a zeroth-order tensor, `coors` is the empty sequence {}, and
// get_elem returns and set_elem updates the sole element.
template <typename TenT>
void test_set_get_elem_zeroth_order(tci_test_fixture<TenT>& fix) {
#if !defined(TCICT_SKIP_GET_ELEM) && !defined(TCICT_SKIP_SET_ELEM) && \
    defined(TCICT_HAS_ZEROTH_ORDER_SWEEP)
  auto& ctx = fix.context();
  auto tensor = tci::zeros<TenT>(ctx, {});
  // set_elem and get_elem are the APIs under test, so the input is checked to
  // be zeroth-order before either is called on it.
  TCICT_EXPECT_ZEROTH_ORDER_SHAPE(ctx, tensor);

  tci::set_elem(ctx, tensor, {}, written_elem<TenT>(6.25));
  TCICT_EXPECT_ZEROTH_ORDER(fix, tensor, 6.25);
  tci::set_elem(ctx, tensor, {}, written_elem<TenT>(-0.75));
  TCICT_EXPECT_ZEROTH_ORDER(fix, tensor, -0.75);
#else
  (void)fix;
#endif
}

// --- size_bytes for small tensor ---

template <typename TenT>
void test_size_bytes_2x2(tci_test_fixture<TenT>& fix) {
#ifndef TCICT_SKIP_SIZE_BYTES
  auto& ctx = fix.context();
  // The second-order one of the three; see the third-order test above for what
  // V1 leaves assertable.
  auto tensor = tci::zeros<TenT>(ctx, {2, 2});
  TCICT_ASSERT_NOTHROW(tci::size_bytes(ctx, tensor));
#else
  (void)fix;
#endif
}

// --- size_bytes: zeroth-order ---

template <typename TenT>
void test_size_bytes_zeroth_order(tci_test_fixture<TenT>& fix) {
#ifndef TCICT_SKIP_SIZE_BYTES
  auto& ctx = fix.context();
  // As for the third-order test above, the call returning is all V1 leaves
  // assertable; this adds the zeroth order to the ranks it is made on. The
  // shape check keeps the test from passing on a tensor that is not one.
  auto tensor = tci::zeros<TenT>(ctx, {});
  TCICT_ASSERT(tci::shape(ctx, tensor).empty());
  TCICT_ASSERT_NOTHROW(tci::size_bytes(ctx, tensor));
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
  X(__VA_ARGS__, "read_only_getters", test_set_get_elem_zeroth_order) \
  X(__VA_ARGS__, "read_only_getters", test_size_bytes_2x2) \
  X(__VA_ARGS__, "read_only_getters", test_size_bytes_zeroth_order)
