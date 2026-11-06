#pragma once

#include <tcapict/assertion.h>
#include <tcapict/diagonal.h>
#include <tcapict/elem_helper.h>
#include <tcapict/elem_sweep.h>
#include <tcapict/fixture.h>
#include <tcapict/range_2d.h>
#include <tcapict/skip.h>
#include <tcapict/zeroth_order.h>

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace tcapict { namespace tests {

// --- zeros ---

template <typename TenT>
void test_zeros(tcapi_test_fixture<TenT>& fix) {
#ifndef TCAPICT_SKIP_ZEROS
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  tcapi::shape_t<TenT> shape = {2, 3};
  TenT tensor;
  TCAPICT_ASSERT_NOTHROW(tensor = tcapi::zeros<TenT>(ctx, shape));
  TCAPICT_ASSERT(tcapi::order(ctx, tensor) == 2);
  auto result_shape = tcapi::shape(ctx, tensor);
  TCAPICT_ASSERT(result_shape.size() == 2);
  TCAPICT_ASSERT(result_shape[0] == 2);
  TCAPICT_ASSERT(result_shape[1] == 3);

  // Verify all elements are zero
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      auto elem = tcapi::get_elem(ctx, tensor, coors_of<TenT>(i, j));
      TCAPICT_ASSERT_CLOSE(real_part<TenT>(elem), 0.0, tol);
      if constexpr (is_complex_v<TenT>) {
        TCAPICT_ASSERT_CLOSE(imag_part<TenT>(elem), 0.0, tol);
      }
    }
  }
#else
  (void)fix;
#endif
}

// --- zeros: zeroth-order ---

// TCAPI V1: "The empty shape is valid; in that case, the function returns a
// zeroth-order tensor whose sole element is initialized to zero."
template <typename TenT>
void test_zeros_zeroth_order(tcapi_test_fixture<TenT>& fix) {
#if !defined(TCAPICT_SKIP_ZEROS) && defined(TCAPICT_HAS_ZEROTH_ORDER_SWEEP)
  auto& ctx = fix.context();
  TenT tensor;
  TCAPICT_ASSERT_NOTHROW(tensor = tcapi::template zeros<TenT>(ctx, {}));
  TCAPICT_EXPECT_ZEROTH_ORDER(fix, tensor, 0.0);
#else
  (void)fix;
#endif
}

// --- fill ---

template <typename TenT>
void test_fill(tcapi_test_fixture<TenT>& fix) {
#ifndef TCAPICT_SKIP_FILL
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto val = make_elem<TenT>(42.0, -7.5);
  auto tensor = tcapi::fill<TenT>(ctx, {3, 4}, val);

  TCAPICT_ASSERT(tcapi::order(ctx, tensor) == 2);
  auto s = tcapi::shape(ctx, tensor);
  TCAPICT_ASSERT(s[0] == 3);
  TCAPICT_ASSERT(s[1] == 4);

  // Verify every element equals the fill value
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      auto elem = tcapi::get_elem(ctx, tensor, coors_of<TenT>(i, j));
      TCAPICT_ASSERT_CLOSE(real_part<TenT>(elem), real_part<TenT>(val), tol);
      if constexpr (is_complex_v<TenT>) {
        TCAPICT_ASSERT_CLOSE(imag_part<TenT>(elem), imag_part<TenT>(val), tol);
      }
    }
  }
#else
  (void)fix;
#endif
}

// --- fill: zeroth-order ---

// TCAPI V1: "The empty shape is valid; in that case, the function returns a
// zeroth-order tensor whose sole element is initialized to `v`."
template <typename TenT>
void test_fill_zeroth_order(tcapi_test_fixture<TenT>& fix) {
#if !defined(TCAPICT_SKIP_FILL) && defined(TCAPICT_HAS_ZEROTH_ORDER_SWEEP)
  auto& ctx = fix.context();
  TenT tensor;
  TCAPICT_ASSERT_NOTHROW(
      tensor = tcapi::template fill<TenT>(ctx, {}, written_elem<TenT>(2.5)));
  TCAPICT_EXPECT_ZEROTH_ORDER(fix, tensor, 2.5);
#else
  (void)fix;
#endif
}

// --- eye ---

template <typename TenT>
void test_eye(tcapi_test_fixture<TenT>& fix) {
#if !defined(TCAPICT_SKIP_EYE) && defined(TCAPICT_HAS_DIAGONAL_CHECK)
  auto& ctx = fix.context();
  TenT identity;
  TCAPICT_ASSERT_NOTHROW(identity = tcapi::eye<TenT>(ctx, 3));
  // Order 2, shape {3, 3} and size 9, since a diagonal tensor's off-diagonal
  // zeros are logical elements however few the backend stores, then exact
  // ones on the diagonal (tol_category::exact: eye does no floating-point
  // arithmetic) and exact zeros off it (see "Diagonal tensors" in
  // CONTRIBUTING.md).
  TCAPICT_EXPECT_DIAGONAL(ctx, identity, {1.0, 1.0, 1.0},
                          tolerance(fix, tol_category::exact));
#else
  (void)fix;
#endif
}

// --- eye: to_range writes every logical element ---

// TCAPI V1's entry for `to_range`: "For a diagonal tensor, this routine writes
// all tensor elements, including off-diagonal zeros." A backend that walked
// only its N stored entries would leave the sentinel standing in the slots it
// skipped.
template <typename TenT>
void test_eye_to_range(tcapi_test_fixture<TenT>& fix) {
#if !defined(TCAPICT_SKIP_EYE) && !defined(TCAPICT_SKIP_TO_RANGE)
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto identity = tcapi::eye<TenT>(ctx, 3);

  auto sentinel = make_elem<TenT>(-999.0);
  std::vector<tcapi::elem_t<TenT>> container(9, sentinel);

  to_range_2d(ctx, identity, {3, 3}, container.begin(), layout_2d::row_major);

  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      auto elem = container[i * 3 + j];
      // Unwritten slot, as distinct from a slot written with a wrong value.
      TCAPICT_ASSERT(std::abs(real_part<TenT>(elem) - real_part<TenT>(sentinel))
                     > tol);
      // eye's elements are exact: eye does no floating-point arithmetic
      // (tol_category::exact in fixture.h), and its off-diagonal zeros are
      // exact as "Diagonal tensors" in CONTRIBUTING.md says.
      TCAPICT_ASSERT(real_part<TenT>(elem) == (i == j ? 1.0 : 0.0));
      if constexpr (is_complex_v<TenT>) {
        TCAPICT_ASSERT(imag_part<TenT>(elem) == 0.0);
      }
    }
  }
#else
  (void)fix;
#endif
}

// --- random (shape {2,3}, every element came out of the generator) ---

// TCAPI V1's entry for `random`: "Construct and return a tensor of shape
// `shape` whose elements are generated by repeatedly invoking the random number
// generator `gen`." For a non-empty shape it fixes neither how many times gen
// is invoked nor which invocation fills which coordinate, so neither an exact
// call count nor a coordinate-to-value expectation is portably assertable. What
// survives is containment: every logical element must equal one of the values
// gen actually emitted.
template <typename TenT>
void test_random_inplace(tcapi_test_fixture<TenT>& fix) {
#ifndef TCAPICT_SKIP_RANDOM
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  tcapi::shape_t<TenT> shape = {2, 3};

  // gen's k-th invocation (counting from 0) returns emission(k), so the call
  // count alone determines the emitted set. gen keeps no record of its values,
  // since recording can allocate; see "Callbacks the backend invokes" in
  // CONTRIBUTING.md. The values start at 1, so no emission equals zero, which
  // an element a zero-filling backend left unwritten would hold.
  auto emission = [](std::size_t k) {
    const double val = static_cast<double>(k) + 1.0;
    return make_elem<TenT>(val, val + 0.5);
  };
  std::size_t calls = 0;
  auto gen = [&]() -> tcapi::elem_t<TenT> { return emission(calls++); };

  TenT tensor;
  TCAPICT_ASSERT_NOTHROW(tensor = tcapi::template random<TenT>(ctx, shape, gen));
  TCAPICT_ASSERT(tcapi::shape(ctx, tensor) == shape);
  TCAPICT_ASSERT(tcapi::size(ctx, tensor) == 6);
  // Separates "gen was never invoked" from "an element does not match": with
  // no calls the containment check below throws at the first coordinate,
  // reporting the mismatch rather than the cause.
  TCAPICT_ASSERT(calls > 0);

  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      auto elem = tcapi::get_elem(ctx, tensor, coors_of<TenT>(i, j));
      bool contained = false;
      for (std::size_t k = 0; k < calls; ++k) {
        const auto candidate = emission(k);
        if (std::abs(real_part<TenT>(elem) - real_part<TenT>(candidate)) < tol
            && std::abs(imag_part<TenT>(elem) - imag_part<TenT>(candidate)) < tol) {
          contained = true;
          break;
        }
      }
      TCAPICT_ASSERT(contained);
    }
  }
#else
  (void)fix;
#endif
}

// --- random (constant generator) ---

// The singleton case of the containment oracle above: with a constant
// generator the emission set has one member, so every element is pinned to a
// known value without assuming an invocation order or an exact count.
template <typename TenT>
void test_random_outofplace(tcapi_test_fixture<TenT>& fix) {
#ifndef TCAPICT_SKIP_RANDOM
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  tcapi::shape_t<TenT> shape = {2, 2};
  auto constant = make_elem<TenT>(2.5, -1.25);
  std::size_t calls = 0;

  auto gen = [&]() -> tcapi::elem_t<TenT> {
    ++calls;
    return constant;
  };

  TenT tensor;
  TCAPICT_ASSERT_NOTHROW(tensor = tcapi::template random<TenT>(ctx, shape, gen));
  TCAPICT_ASSERT(tcapi::shape(ctx, tensor) == shape);
  TCAPICT_ASSERT(tcapi::size(ctx, tensor) == 4);
  // The value checks below compare against `constant`, so on their own they
  // cannot tell a backend that consulted gen from one that filled with the
  // same value without consulting it. The counter is what separates the two.
  TCAPICT_ASSERT(calls > 0);

  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 2; ++j) {
      auto elem = tcapi::get_elem(ctx, tensor, coors_of<TenT>(i, j));
      TCAPICT_ASSERT_CLOSE(real_part<TenT>(elem), real_part<TenT>(constant), tol);
      if constexpr (is_complex_v<TenT>) {
        TCAPICT_ASSERT_CLOSE(imag_part<TenT>(elem), imag_part<TenT>(constant), tol);
      }
    }
  }
#else
  (void)fix;
#endif
}

// --- random: zeroth-order ---

// TCAPI V1: "The empty shape is valid; in that case, the function returns a
// zeroth-order tensor and invokes the random-number generator once." The test
// asserts that one call and that the sole element is the value it returned.
template <typename TenT>
void test_random_zeroth_order(tcapi_test_fixture<TenT>& fix) {
#if !defined(TCAPICT_SKIP_RANDOM) && defined(TCAPICT_HAS_ZEROTH_ORDER_SWEEP)
  auto& ctx = fix.context();
  // The k-th call (counting from 1) returns written_elem<TenT>(value_of_call(k)).
  auto value_of_call = [](std::size_t k) {
    return 10.0 + static_cast<double>(k);
  };
  std::size_t calls = 0;
  auto gen = [&calls, &value_of_call]() -> tcapi::elem_t<TenT> {
    ++calls;
    return written_elem<TenT>(value_of_call(calls));
  };

  TenT tensor;
  TCAPICT_ASSERT_NOTHROW(tensor = tcapi::template random<TenT>(ctx, {}, gen));
  // Checked before the call count, so a backend that ignores the empty shape
  // is reported as such rather than as a wrong number of calls.
  TCAPICT_EXPECT_ZEROTH_ORDER_SHAPE(ctx, tensor);
  TCAPICT_ASSERT_MSG(calls == 1, "random did not invoke gen() exactly once");
  TCAPICT_EXPECT_ELEMS(
      fix, tensor, tcapi::shape_t<TenT>{},
      [&](const tcapi::elem_coors_t<TenT> &) { return value_of_call(1); },
      tol_category::exact);
#else
  (void)fix;
#endif
}

// --- copy (in-place) ---

template <typename TenT>
void test_copy_inplace(tcapi_test_fixture<TenT>& fix) {
#ifndef TCAPICT_SKIP_COPY
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto a = tcapi::zeros<TenT>(ctx, {2, 3});
  tcapi::set_elem(ctx, a, {0, 0}, make_elem<TenT>(42.0, 13.0));
  tcapi::set_elem(ctx, a, {1, 2}, make_elem<TenT>(-5.5, 7.7));

  TenT b;
  TCAPICT_ASSERT_NOTHROW(b = tcapi::copy(ctx, a));

  auto val1 = tcapi::get_elem(ctx, b, {0, 0});
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(val1), 42.0, tol);

  auto val2 = tcapi::get_elem(ctx, b, {1, 2});
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(val2), -5.5, tol);

  if constexpr (is_complex_v<TenT>) {
    TCAPICT_ASSERT_CLOSE(imag_part<TenT>(val1), 13.0, tol);
    TCAPICT_ASSERT_CLOSE(imag_part<TenT>(val2), 7.7, tol);
  }

  TCAPICT_ASSERT(tcapi::shape(ctx, a) == tcapi::shape(ctx, b));
  TCAPICT_ASSERT(tcapi::order(ctx, a) == tcapi::order(ctx, b));
#else
  (void)fix;
#endif
}

// --- copy (out-of-place) ---

template <typename TenT>
void test_copy_outofplace(tcapi_test_fixture<TenT>& fix) {
#ifndef TCAPICT_SKIP_COPY
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto a = tcapi::zeros<TenT>(ctx, {3, 4, 2});
  tcapi::set_elem(ctx, a, {0, 1, 0}, make_elem<TenT>(1.23, 4.56));
  tcapi::set_elem(ctx, a, {2, 3, 1}, make_elem<TenT>(-9.87, 6.54));

  auto b = tcapi::copy(ctx, a);

  auto val1_a = tcapi::get_elem(ctx, a, {0, 1, 0});
  auto val1_b = tcapi::get_elem(ctx, b, {0, 1, 0});
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(val1_a), real_part<TenT>(val1_b), tol);
  if constexpr (is_complex_v<TenT>) {
    TCAPICT_ASSERT_CLOSE(imag_part<TenT>(val1_a), imag_part<TenT>(val1_b), tol);
  }

  TCAPICT_ASSERT(tcapi::shape(ctx, a) == tcapi::shape(ctx, b));
  TCAPICT_ASSERT(tcapi::order(ctx, a) == tcapi::order(ctx, b));
#else
  (void)fix;
#endif
}

// --- copy independence ---

template <typename TenT>
void test_copy_independence(tcapi_test_fixture<TenT>& fix) {
#ifndef TCAPICT_SKIP_COPY
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto a = tcapi::zeros<TenT>(ctx, {2, 2});
  tcapi::set_elem(ctx, a, {0, 0}, make_elem<TenT>(100.0));

  auto b = tcapi::copy(ctx, a);
  tcapi::set_elem(ctx, a, {0, 0}, make_elem<TenT>(999.0));

  auto copy_val = tcapi::get_elem(ctx, b, {0, 0});
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(copy_val), 100.0, tol);

  auto orig_val = tcapi::get_elem(ctx, a, {0, 0});
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(orig_val), 999.0, tol);
#else
  (void)fix;
#endif
}

// --- copy single element ---

template <typename TenT>
void test_copy_single_element(tcapi_test_fixture<TenT>& fix) {
#ifndef TCAPICT_SKIP_COPY
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto a = tcapi::zeros<TenT>(ctx, {1});
  tcapi::set_elem(ctx, a, {0}, make_elem<TenT>(3.14, 2.71));

  auto b = tcapi::copy(ctx, a);

  auto val = tcapi::get_elem(ctx, b, {0});
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(val), 3.14, tol);
  if constexpr (is_complex_v<TenT>) {
    TCAPICT_ASSERT_CLOSE(imag_part<TenT>(val), 2.71, tol);
  }
#else
  (void)fix;
#endif
}

// --- copy large tensor ---

template <typename TenT>
void test_copy_large(tcapi_test_fixture<TenT>& fix) {
#ifndef TCAPICT_SKIP_COPY
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto a = tcapi::zeros<TenT>(ctx, {10, 10, 10});
  tcapi::set_elem(ctx, a, {0, 0, 0}, make_elem<TenT>(1.0, 0.0));
  tcapi::set_elem(ctx, a, {9, 9, 9}, make_elem<TenT>(0.0, 1.0));

  auto b = tcapi::copy(ctx, a);

  auto corner1 = tcapi::get_elem(ctx, b, {0, 0, 0});
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(corner1), 1.0, tol);

  auto corner2 = tcapi::get_elem(ctx, b, {9, 9, 9});
  if constexpr (is_complex_v<TenT>) {
    TCAPICT_ASSERT_CLOSE(imag_part<TenT>(corner1), 0.0, tol);
    TCAPICT_ASSERT_CLOSE(real_part<TenT>(corner2), 0.0, tol);
    TCAPICT_ASSERT_CLOSE(imag_part<TenT>(corner2), 1.0, tol);
  } else {
    // For real types, make_elem(0.0, 1.0) yields 0.0
    TCAPICT_ASSERT_CLOSE(real_part<TenT>(corner2), 0.0, tol);
  }

  TCAPICT_ASSERT(tcapi::shape(ctx, a) == tcapi::shape(ctx, b));
#else
  (void)fix;
#endif
}

// --- copy: zeroth-order ---

// TCAPI V1: "The returned tensor has the same shape and element values as
// `orig`, and the two tensors do not share memory", so a write to the copy
// leaves the original as it was.
template <typename TenT>
void test_copy_zeroth_order(tcapi_test_fixture<TenT>& fix) {
#if !defined(TCAPICT_SKIP_COPY) && defined(TCAPICT_HAS_ZEROTH_ORDER_SWEEP)
  auto& ctx = fix.context();
  auto orig = tcapi::template zeros<TenT>(ctx, {});
  TCAPICT_SET_ZEROTH_ORDER(fix, orig, 3.5);

  TenT copied;
  TCAPICT_ASSERT_NOTHROW(copied = tcapi::copy(ctx, orig));
  TCAPICT_EXPECT_ZEROTH_ORDER(fix, copied, 3.5);

  TCAPICT_SET_ZEROTH_ORDER(fix, copied, -4.0);
  TCAPICT_EXPECT_ZEROTH_ORDER(fix, copied, -4.0);
  TCAPICT_EXPECT_ZEROTH_ORDER(fix, orig, 3.5);
#else
  (void)fix;
#endif
}

// --- assign_from_range (row-major) ---

template <typename TenT>
void test_assign_from_range_row_major(tcapi_test_fixture<TenT>& fix) {
#ifndef TCAPICT_SKIP_ASSIGN_FROM_RANGE
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);

  // {{1, 2, 3}, {4, 5, 6}} in row-major order. Each element's imaginary part is
  // -0.5 times its real part, so the complex instantiations check where the
  // imaginary part lands as well.
  std::vector<tcapi::elem_t<TenT>> container;
  for (double v : {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}) {
    container.push_back(make_elem<TenT>(v, -0.5 * v));
  }

  tcapi::shape_t<TenT> shape = {2, 3};
  TenT tensor;
  assign_from_range_2d(ctx, shape, container.begin(), layout_2d::row_major, tensor);

  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      const auto elem = tcapi::get_elem(ctx, tensor, coors_of<TenT>(i, j));
      const auto value = static_cast<double>(i * 3 + j + 1);
      TCAPICT_ASSERT_CLOSE(real_part<TenT>(elem), value, tol);
      if constexpr (is_complex_v<TenT>) {
        TCAPICT_ASSERT_CLOSE(imag_part<TenT>(elem), -0.5 * value, tol);
      }
    }
  }
#else
  (void)fix;
#endif
}

// --- assign_from_range (column-major) ---

template <typename TenT>
void test_assign_from_range_column_major(tcapi_test_fixture<TenT>& fix) {
#ifndef TCAPICT_SKIP_ASSIGN_FROM_RANGE
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);

  // {{1, 2, 3}, {4, 5, 6}} in column-major order, imaginary parts as in the
  // row-major test. Non-square, so a map that confused the row count with the
  // column count would misplace elements.
  std::vector<tcapi::elem_t<TenT>> container;
  for (double v : {1.0, 4.0, 2.0, 5.0, 3.0, 6.0}) {
    container.push_back(make_elem<TenT>(v, -0.5 * v));
  }

  tcapi::shape_t<TenT> shape = {2, 3};
  TenT tensor;
  assign_from_range_2d(ctx, shape, container.begin(), layout_2d::column_major, tensor);

  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      const auto elem = tcapi::get_elem(ctx, tensor, coors_of<TenT>(i, j));
      const auto value = static_cast<double>(i * 3 + j + 1);
      TCAPICT_ASSERT_CLOSE(real_part<TenT>(elem), value, tol);
      if constexpr (is_complex_v<TenT>) {
        TCAPICT_ASSERT_CLOSE(imag_part<TenT>(elem), -0.5 * value, tol);
      }
    }
  }
#else
  (void)fix;
#endif
}

// --- assign_from_range: zeroth-order ---

// TCAPI V1: "The element at coordinate tuple `coors` is initialized as
// `*(first + coors2idx(coors))` in C++". The only coordinate of shape {} is {},
// which the map sends to index 0; assign_from_range_0d asserts the map's one
// invocation.
template <typename TenT>
void test_assign_from_range_zeroth_order(tcapi_test_fixture<TenT>& fix) {
#if !defined(TCAPICT_SKIP_ASSIGN_FROM_RANGE) && defined(TCAPICT_HAS_ZEROTH_ORDER_SWEEP)
  std::vector<tcapi::elem_t<TenT>> source = {written_elem<TenT>(7.5)};
  TenT tensor;
  assign_from_range_0d(fix.context(), source.begin(), tensor);
  TCAPICT_EXPECT_ZEROTH_ORDER(fix, tensor, 7.5);
#else
  (void)fix;
#endif
}

// --- allocate ---

template <typename TenT>
void test_allocate_3d(tcapi_test_fixture<TenT>& fix) {
#ifndef TCAPICT_SKIP_ALLOCATE
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  tcapi::shape_t<TenT> shape = {3, 4, 5};
  auto tensor = tcapi::allocate<TenT>(ctx, shape);

  auto tensor_shape = tcapi::shape(ctx, tensor);
  TCAPICT_ASSERT(tensor_shape.size() == 3);
  TCAPICT_ASSERT(tensor_shape[0] == 3);
  TCAPICT_ASSERT(tensor_shape[1] == 4);
  TCAPICT_ASSERT(tensor_shape[2] == 5);
  TCAPICT_ASSERT(tcapi::size(ctx, tensor) == 60);

  // Verify element type by round-tripping a value
  auto val = make_elem<TenT>(1.5, 2.5);
  tcapi::set_elem(ctx, tensor, {0, 0, 0}, val);
  auto got = tcapi::get_elem(ctx, tensor, {0, 0, 0});
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(got), real_part<TenT>(val), tol);
  if constexpr (is_complex_v<TenT>) {
    TCAPICT_ASSERT_CLOSE(imag_part<TenT>(got), imag_part<TenT>(val), tol);
  }
#else
  (void)fix;
#endif
}

template <typename TenT>
void test_allocate_2d(tcapi_test_fixture<TenT>& fix) {
#ifndef TCAPICT_SKIP_ALLOCATE
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  tcapi::shape_t<TenT> shape = {2, 3};
  auto tensor = tcapi::allocate<TenT>(ctx, shape);

  auto tensor_shape = tcapi::shape(ctx, tensor);
  TCAPICT_ASSERT(tensor_shape.size() == 2);
  TCAPICT_ASSERT(tensor_shape[0] == 2);
  TCAPICT_ASSERT(tensor_shape[1] == 3);
  TCAPICT_ASSERT(tcapi::size(ctx, tensor) == 6);

  // Verify element type by round-tripping a value
  auto val = make_elem<TenT>(1.5, 2.5);
  tcapi::set_elem(ctx, tensor, {0, 0}, val);
  auto got = tcapi::get_elem(ctx, tensor, {0, 0});
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(got), real_part<TenT>(val), tol);
  if constexpr (is_complex_v<TenT>) {
    TCAPICT_ASSERT_CLOSE(imag_part<TenT>(got), imag_part<TenT>(val), tol);
  }
#else
  (void)fix;
#endif
}

template <typename TenT>
void test_allocate_1d(tcapi_test_fixture<TenT>& fix) {
#ifndef TCAPICT_SKIP_ALLOCATE
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  tcapi::shape_t<TenT> shape = {10};
  auto tensor = tcapi::allocate<TenT>(ctx, shape);

  auto tensor_shape = tcapi::shape(ctx, tensor);
  TCAPICT_ASSERT(tensor_shape.size() == 1);
  TCAPICT_ASSERT(tensor_shape[0] == 10);
  TCAPICT_ASSERT(tcapi::size(ctx, tensor) == 10);

  // Verify element type by round-tripping a value
  auto val = make_elem<TenT>(1.5, 2.5);
  tcapi::set_elem(ctx, tensor, {0}, val);
  auto got = tcapi::get_elem(ctx, tensor, {0});
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(got), real_part<TenT>(val), tol);
  if constexpr (is_complex_v<TenT>) {
    TCAPICT_ASSERT_CLOSE(imag_part<TenT>(got), imag_part<TenT>(val), tol);
  }
#else
  (void)fix;
#endif
}

// --- allocate: zeroth-order ---

// TCAPI V1: "The empty shape is valid; in that case, the function returns a
// zeroth-order tensor whose sole element is uninitialized." The element is
// read only after it is set.
template <typename TenT>
void test_allocate_zeroth_order(tcapi_test_fixture<TenT>& fix) {
#if !defined(TCAPICT_SKIP_ALLOCATE) && defined(TCAPICT_HAS_ZEROTH_ORDER_SWEEP)
  auto& ctx = fix.context();
  TenT tensor;
  TCAPICT_ASSERT_NOTHROW(tensor = tcapi::template allocate<TenT>(ctx, {}));
  TCAPICT_EXPECT_ZEROTH_ORDER_SHAPE(ctx, tensor);

  TCAPICT_SET_ZEROTH_ORDER(fix, tensor, -1.25);
  TCAPICT_EXPECT_ZEROTH_ORDER(fix, tensor, -1.25);
#else
  (void)fix;
#endif
}

// --- clear ---

template <typename TenT>
void test_clear_basic(tcapi_test_fixture<TenT>& fix) {
#ifndef TCAPICT_SKIP_CLEAR
  auto& ctx = fix.context();
  auto tensor = tcapi::eye<TenT>(ctx, 3);
  TCAPICT_ASSERT(tcapi::size(ctx, tensor) == 9);
  TCAPICT_ASSERT_NOTHROW(tcapi::clear(ctx, tensor));
  TCAPICT_ASSERT(tcapi::order(ctx, tensor) == 0);
  TCAPICT_ASSERT(tcapi::shape(ctx, tensor).empty());
#else
  (void)fix;
#endif
}

template <typename TenT>
void test_clear_empty(tcapi_test_fixture<TenT>& fix) {
#ifndef TCAPICT_SKIP_CLEAR
  auto& ctx = fix.context();
  TenT empty_tensor;
  TCAPICT_ASSERT_NOTHROW(tcapi::clear(ctx, empty_tensor));
  TCAPICT_ASSERT(tcapi::order(ctx, empty_tensor) == 0);
#else
  (void)fix;
#endif
}

template <typename TenT>
void test_clear_and_reallocate(tcapi_test_fixture<TenT>& fix) {
#ifndef TCAPICT_SKIP_CLEAR
  auto& ctx = fix.context();
  auto tensor = tcapi::eye<TenT>(ctx, 2);
  tcapi::clear(ctx, tensor);
  TCAPICT_ASSERT(tcapi::order(ctx, tensor) == 0);
  TCAPICT_ASSERT_NOTHROW(tensor = tcapi::allocate<TenT>(ctx, {2, 2}));
  TCAPICT_ASSERT(tcapi::order(ctx, tensor) == 2);
#else
  (void)fix;
#endif
}

// --- move (in-place) ---

template <typename TenT>
void test_move_inplace(tcapi_test_fixture<TenT>& fix) {
#ifndef TCAPICT_SKIP_MOVE
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto source = tcapi::eye<TenT>(ctx, 3);

  auto source_size = tcapi::size(ctx, source);
  auto source_shape = tcapi::shape(ctx, source);
  auto original_elem = tcapi::get_elem(ctx, source, {0, 0});

  auto destination = tcapi::move(ctx, source);

  TCAPICT_ASSERT(tcapi::size(ctx, destination) == source_size);
  TCAPICT_ASSERT(tcapi::shape(ctx, destination) == source_shape);
  auto dest_elem = tcapi::get_elem(ctx, destination, {0, 0});
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(dest_elem), real_part<TenT>(original_elem), tol);
  if constexpr (is_complex_v<TenT>) {
    TCAPICT_ASSERT_CLOSE(imag_part<TenT>(dest_elem), imag_part<TenT>(original_elem), tol);
  }

  // Verify source is invalidated after move
  TCAPICT_ASSERT(tcapi::order(ctx, source) == 0);
#else
  (void)fix;
#endif
}

// --- move (out-of-place) ---

template <typename TenT>
void test_move_outofplace(tcapi_test_fixture<TenT>& fix) {
#ifndef TCAPICT_SKIP_MOVE
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto source = tcapi::eye<TenT>(ctx, 2);

  auto source_size = tcapi::size(ctx, source);
  auto source_shape = tcapi::shape(ctx, source);
  auto original_elem = tcapi::get_elem(ctx, source, {1, 1});

  auto result = tcapi::move(ctx, source);

  TCAPICT_ASSERT(tcapi::size(ctx, result) == source_size);
  TCAPICT_ASSERT(tcapi::shape(ctx, result) == source_shape);
  auto result_elem = tcapi::get_elem(ctx, result, {1, 1});
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(result_elem), real_part<TenT>(original_elem), tol);
  if constexpr (is_complex_v<TenT>) {
    TCAPICT_ASSERT_CLOSE(imag_part<TenT>(result_elem), imag_part<TenT>(original_elem), tol);
  }

  // Verify source is invalidated after move
  TCAPICT_ASSERT(tcapi::order(ctx, source) == 0);
#else
  (void)fix;
#endif
}

// --- move empty tensor ---

template <typename TenT>
void test_move_empty(tcapi_test_fixture<TenT>& fix) {
#ifndef TCAPICT_SKIP_MOVE
  auto& ctx = fix.context();
  TenT empty_source, empty_result;
  TCAPICT_ASSERT_NOTHROW(empty_result = tcapi::move(ctx, empty_source));
  TCAPICT_ASSERT(tcapi::order(ctx, empty_source) == 0);
#else
  (void)fix;
#endif
}

// --- move preserves values ---

template <typename TenT>
void test_move_preserves_values(tcapi_test_fixture<TenT>& fix) {
#ifndef TCAPICT_SKIP_MOVE
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto source = tcapi::zeros<TenT>(ctx, {2, 3});

  auto val1 = make_elem<TenT>(2.5, 1.5);
  auto val2 = make_elem<TenT>(3.7, -2.1);
  tcapi::set_elem(ctx, source, {0, 1}, val1);
  tcapi::set_elem(ctx, source, {1, 2}, val2);

  auto destination = tcapi::move(ctx, source);

  auto moved_val1 = tcapi::get_elem(ctx, destination, {0, 1});
  auto moved_val2 = tcapi::get_elem(ctx, destination, {1, 2});

  TCAPICT_ASSERT_CLOSE(real_part<TenT>(moved_val1), real_part<TenT>(val1), tol);
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(moved_val2), real_part<TenT>(val2), tol);
  if constexpr (is_complex_v<TenT>) {
    TCAPICT_ASSERT_CLOSE(imag_part<TenT>(moved_val1), imag_part<TenT>(val1), tol);
    TCAPICT_ASSERT_CLOSE(imag_part<TenT>(moved_val2), imag_part<TenT>(val2), tol);
  }

  // Verify source is invalidated after move
  TCAPICT_ASSERT(tcapi::order(ctx, source) == 0);
#else
  (void)fix;
#endif
}

// --- move: zeroth-order ---

// The result carries the zeroth-order tensor. TCAPI V1 leaves `from` as
// `TenT{}`, whose queries it does not define, so nothing is asserted on it.
template <typename TenT>
void test_move_zeroth_order(tcapi_test_fixture<TenT>& fix) {
#if !defined(TCAPICT_SKIP_MOVE) && defined(TCAPICT_HAS_ZEROTH_ORDER_SWEEP)
  auto& ctx = fix.context();
  auto source = tcapi::template zeros<TenT>(ctx, {});
  TCAPICT_SET_ZEROTH_ORDER(fix, source, 4.5);

  TenT moved;
  TCAPICT_ASSERT_NOTHROW(moved = tcapi::move(ctx, source));
  TCAPICT_EXPECT_ZEROTH_ORDER(fix, moved, 4.5);
#else
  (void)fix;
#endif
}

}}  // namespace tcapict::tests

// Bulk registration helper: invokes X(..., "category", test_fn) once per test.
// See include/tcapict/adapters/doctest.h for usage.
#define TCAPICT_FOREACH_CONSTRUCTION_TEST_ALL_TYPES(X, ...) \
  X(__VA_ARGS__, "construction", test_zeros) \
  X(__VA_ARGS__, "construction", test_zeros_zeroth_order) \
  X(__VA_ARGS__, "construction", test_fill) \
  X(__VA_ARGS__, "construction", test_fill_zeroth_order) \
  X(__VA_ARGS__, "construction", test_eye) \
  X(__VA_ARGS__, "construction", test_eye_to_range) \
  X(__VA_ARGS__, "construction", test_random_inplace) \
  X(__VA_ARGS__, "construction", test_random_outofplace) \
  X(__VA_ARGS__, "construction", test_random_zeroth_order) \
  X(__VA_ARGS__, "construction", test_copy_inplace) \
  X(__VA_ARGS__, "construction", test_copy_outofplace) \
  X(__VA_ARGS__, "construction", test_copy_independence) \
  X(__VA_ARGS__, "construction", test_copy_single_element) \
  X(__VA_ARGS__, "construction", test_copy_large) \
  X(__VA_ARGS__, "construction", test_copy_zeroth_order) \
  X(__VA_ARGS__, "construction", test_assign_from_range_row_major) \
  X(__VA_ARGS__, "construction", test_assign_from_range_column_major) \
  X(__VA_ARGS__, "construction", test_assign_from_range_zeroth_order) \
  X(__VA_ARGS__, "construction", test_allocate_3d) \
  X(__VA_ARGS__, "construction", test_allocate_2d) \
  X(__VA_ARGS__, "construction", test_allocate_1d) \
  X(__VA_ARGS__, "construction", test_allocate_zeroth_order) \
  X(__VA_ARGS__, "construction", test_clear_basic) \
  X(__VA_ARGS__, "construction", test_clear_empty) \
  X(__VA_ARGS__, "construction", test_clear_and_reallocate) \
  X(__VA_ARGS__, "construction", test_move_inplace) \
  X(__VA_ARGS__, "construction", test_move_outofplace) \
  X(__VA_ARGS__, "construction", test_move_empty) \
  X(__VA_ARGS__, "construction", test_move_preserves_values) \
  X(__VA_ARGS__, "construction", test_move_zeroth_order)
