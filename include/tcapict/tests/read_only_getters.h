#pragma once

#include <tcapict/assertion.h>
#include <tcapict/elem_helper.h>
#include <tcapict/fixture.h>
#include <tcapict/skip.h>
#include <tcapict/zeroth_order.h>

#include <cmath>

namespace tcapict { namespace tests {

// --- order ---

template <typename TenT>
void test_order(tcapi_test_fixture<TenT>& fix) {
#ifndef TCAPICT_SKIP_ORDER
  auto& ctx = fix.context();
  auto tensor = tcapi::zeros<TenT>(ctx, {2, 3, 4});
  TCAPICT_ASSERT(tcapi::order(ctx, tensor) == 3);
#else
  (void)fix;
#endif
}

// --- shape ---

template <typename TenT>
void test_shape(tcapi_test_fixture<TenT>& fix) {
#ifndef TCAPICT_SKIP_SHAPE
  auto& ctx = fix.context();
  auto tensor = tcapi::zeros<TenT>(ctx, {2, 3, 4});
  auto result_shape = tcapi::shape(ctx, tensor);
  TCAPICT_ASSERT(result_shape.size() == 3);
  TCAPICT_ASSERT(result_shape[0] == 2);
  TCAPICT_ASSERT(result_shape[1] == 3);
  TCAPICT_ASSERT(result_shape[2] == 4);
#else
  (void)fix;
#endif
}

// --- size ---

template <typename TenT>
void test_size(tcapi_test_fixture<TenT>& fix) {
#ifndef TCAPICT_SKIP_SIZE
  auto& ctx = fix.context();
  auto tensor = tcapi::zeros<TenT>(ctx, {2, 3, 4});
  TCAPICT_ASSERT(tcapi::size(ctx, tensor) == 24);
#else
  (void)fix;
#endif
}

// --- size_bytes ---

template <typename TenT>
void test_size_bytes(tcapi_test_fixture<TenT>& fix) {
#ifndef TCAPICT_SKIP_SIZE_BYTES
  auto& ctx = fix.context();
  auto tensor = tcapi::zeros<TenT>(ctx, {2, 3, 4});
  // V1's entry for it is "Return the physical memory footprint of `a` in
  // bytes. A compactly stored diagonal tensor may use fewer bytes than a dense
  // tensor with the same shape." Its example gives 96 bytes for a float32
  // tensor of shape {3, 4, 2}; the suite reads that as the example's value,
  // not as a formula every backend must meet, since the entry states none and
  // no lower bound. Callable-and-returns is the whole of what is portably
  // assertable, so what this test and its second-order and zeroth-order
  // counterparts below give between them is that none of the three ranks makes
  // the call fail.
  TCAPICT_ASSERT_NOTHROW(tcapi::size_bytes(ctx, tensor));
#else
  (void)fix;
#endif
}

// --- get_elem / set_elem ---

template <typename TenT>
void test_set_get_elem(tcapi_test_fixture<TenT>& fix) {
#if !defined(TCAPICT_SKIP_GET_ELEM) && !defined(TCAPICT_SKIP_SET_ELEM)
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  auto tensor = tcapi::zeros<TenT>(ctx, {2, 2});

  tcapi::elem_coors_t<TenT> coord = {0, 0};
  auto expected = make_elem<TenT>(2.5, 1.5);
  tcapi::set_elem(ctx, tensor, coord, expected);

  auto retrieved = tcapi::get_elem(ctx, tensor, coord);
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(retrieved), real_part<TenT>(expected), eps);
  if constexpr (is_complex_v<TenT>) {
    TCAPICT_ASSERT_CLOSE(imag_part<TenT>(retrieved), imag_part<TenT>(expected), eps);
  }
#else
  (void)fix;
#endif
}

// --- get_elem / set_elem: zeroth-order ---

// TCAPI V1, of get_elem: "For a zeroth-order tensor, `coors` is the empty
// coordinate sequence, and the function returns the sole element." Of
// set_elem, the same sentence ends "the function updates the sole element".
template <typename TenT>
void test_set_get_elem_zeroth_order(tcapi_test_fixture<TenT>& fix) {
#if !defined(TCAPICT_SKIP_GET_ELEM) && !defined(TCAPICT_SKIP_SET_ELEM) && \
    defined(TCAPICT_HAS_ZEROTH_ORDER_SWEEP)
  auto& ctx = fix.context();
  auto tensor = tcapi::zeros<TenT>(ctx, {});
  // set_elem and get_elem are the APIs under test, so the input is checked to
  // be zeroth-order before either is called on it.
  TCAPICT_EXPECT_ZEROTH_ORDER_SHAPE(ctx, tensor);

  tcapi::set_elem(ctx, tensor, {}, written_elem<TenT>(6.25));
  TCAPICT_EXPECT_ZEROTH_ORDER(fix, tensor, 6.25);
  tcapi::set_elem(ctx, tensor, {}, written_elem<TenT>(-0.75));
  TCAPICT_EXPECT_ZEROTH_ORDER(fix, tensor, -0.75);
#else
  (void)fix;
#endif
}

// --- size_bytes for small tensor ---

template <typename TenT>
void test_size_bytes_2x2(tcapi_test_fixture<TenT>& fix) {
#ifndef TCAPICT_SKIP_SIZE_BYTES
  auto& ctx = fix.context();
  // The second-order one of the three; see the third-order test above for what
  // V1 leaves assertable.
  auto tensor = tcapi::zeros<TenT>(ctx, {2, 2});
  TCAPICT_ASSERT_NOTHROW(tcapi::size_bytes(ctx, tensor));
#else
  (void)fix;
#endif
}

// --- size_bytes: zeroth-order ---

template <typename TenT>
void test_size_bytes_zeroth_order(tcapi_test_fixture<TenT>& fix) {
#ifndef TCAPICT_SKIP_SIZE_BYTES
  auto& ctx = fix.context();
  // As for the third-order test above, the call returning is all V1 leaves
  // assertable; this adds the zeroth order to the ranks it is made on. The
  // shape check keeps the test from passing on a tensor that is not one.
  auto tensor = tcapi::zeros<TenT>(ctx, {});
  TCAPICT_ASSERT(tcapi::shape(ctx, tensor).empty());
  TCAPICT_ASSERT_NOTHROW(tcapi::size_bytes(ctx, tensor));
#else
  (void)fix;
#endif
}

}}  // namespace tcapict::tests

// Bulk registration helper: invokes X(..., "category", test_fn) once per test.
// See include/tcapict/adapters/doctest.h for usage.
#define TCAPICT_FOREACH_READ_ONLY_GETTERS_TEST_ALL_TYPES(X, ...) \
  X(__VA_ARGS__, "read_only_getters", test_order) \
  X(__VA_ARGS__, "read_only_getters", test_shape) \
  X(__VA_ARGS__, "read_only_getters", test_size) \
  X(__VA_ARGS__, "read_only_getters", test_size_bytes) \
  X(__VA_ARGS__, "read_only_getters", test_set_get_elem) \
  X(__VA_ARGS__, "read_only_getters", test_set_get_elem_zeroth_order) \
  X(__VA_ARGS__, "read_only_getters", test_size_bytes_2x2) \
  X(__VA_ARGS__, "read_only_getters", test_size_bytes_zeroth_order)
