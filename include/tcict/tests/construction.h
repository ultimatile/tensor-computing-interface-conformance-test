#pragma once

#include <tcict/assertion.h>
#include <tcict/elem_helper.h>
#include <tcict/fixture.h>
#include <tcict/skip.h>

#include <cmath>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <vector>

namespace tcict { namespace tests {

// --- zeros ---

template <typename TenT>
void test_zeros(tci_test_fixture<TenT>& fix) {
#ifndef TCICT_SKIP_ZEROS
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  tci::shape_t<TenT> shape = {2, 3};
  TenT tensor;
  TCICT_ASSERT_NOTHROW(tensor = tci::zeros<TenT>(ctx, shape));
  TCICT_ASSERT(tci::order(ctx, tensor) == 2);
  auto result_shape = tci::shape(ctx, tensor);
  TCICT_ASSERT(result_shape.size() == 2);
  TCICT_ASSERT(result_shape[0] == 2);
  TCICT_ASSERT(result_shape[1] == 3);

  // Verify all elements are zero
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      auto elem = tci::get_elem(ctx, tensor, {i, j});
      TCICT_ASSERT_CLOSE(real_part<TenT>(elem), 0.0, tol);
      if constexpr (is_complex_v<TenT>) {
        TCICT_ASSERT_CLOSE(imag_part<TenT>(elem), 0.0, tol);
      }
    }
  }
#else
  (void)fix;
#endif
}

// --- fill ---

template <typename TenT>
void test_fill(tci_test_fixture<TenT>& fix) {
#ifndef TCICT_SKIP_FILL
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto val = make_elem<TenT>(42.0, -7.5);
  auto tensor = tci::fill<TenT>(ctx, {3, 4}, val);

  TCICT_ASSERT(tci::order(ctx, tensor) == 2);
  auto s = tci::shape(ctx, tensor);
  TCICT_ASSERT(s[0] == 3);
  TCICT_ASSERT(s[1] == 4);

  // Verify every element equals the fill value
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      auto elem = tci::get_elem(ctx, tensor, {i, j});
      TCICT_ASSERT_CLOSE(real_part<TenT>(elem), real_part<TenT>(val), tol);
      if constexpr (is_complex_v<TenT>) {
        TCICT_ASSERT_CLOSE(imag_part<TenT>(elem), imag_part<TenT>(val), tol);
      }
    }
  }
#else
  (void)fix;
#endif
}

// --- eye ---

template <typename TenT>
void test_eye(tci_test_fixture<TenT>& fix) {
#ifndef TCICT_SKIP_EYE
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  TenT identity;
  TCICT_ASSERT_NOTHROW(identity = tci::eye<TenT>(ctx, 3));
  TCICT_ASSERT(tci::order(ctx, identity) == 2);
  auto result_shape = tci::shape(ctx, identity);
  TCICT_ASSERT(result_shape.size() == 2);
  TCICT_ASSERT(result_shape[0] == 3);
  TCICT_ASSERT(result_shape[1] == 3);
  // A diagonal tensor's off-diagonal zeros are logical elements too, so an
  // {N, N} identity has N^2 logical elements however few the backend stores.
  TCICT_ASSERT(tci::size(ctx, identity) == 9);

  // Verify diagonal elements are 1 and off-diagonal elements are 0
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      auto elem = tci::get_elem(ctx, identity, {i, j});
      if (i == j) {
        TCICT_ASSERT_CLOSE(real_part<TenT>(elem), 1.0, tol);
      } else {
        TCICT_ASSERT_CLOSE(real_part<TenT>(elem), 0.0, tol);
      }
      if constexpr (is_complex_v<TenT>) {
        TCICT_ASSERT_CLOSE(imag_part<TenT>(elem), 0.0, tol);
      }
    }
  }
#else
  (void)fix;
#endif
}

// --- eye: to_range writes every logical element ---

// V1: "A diagonal tensor's logical elements — including off-diagonal zeros —
// define its behavior regardless of storage. Thus ... `to_range` writes all
// `N^2` elements". A backend that walked only its N stored entries would leave
// the sentinel standing in the slots it skipped.
template <typename TenT>
void test_eye_to_range(tci_test_fixture<TenT>& fix) {
#if !defined(TCICT_SKIP_EYE) && !defined(TCICT_SKIP_TO_RANGE)
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto identity = tci::eye<TenT>(ctx, 3);

  auto sentinel = make_elem<TenT>(-999.0);
  std::vector<tci::elem_t<TenT>> container(9, sentinel);

  auto row_major_map = row_major_2d<TenT>(3);

  TCICT_ASSERT_NOTHROW(
      tci::to_range(ctx, identity, container.begin(), row_major_map));

  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      auto elem = container[i * 3 + j];
      // Unwritten slot, as distinct from a slot written with a wrong value.
      TCICT_ASSERT(std::abs(real_part<TenT>(elem) - real_part<TenT>(sentinel))
                   > tol);
      TCICT_ASSERT_CLOSE(real_part<TenT>(elem), i == j ? 1.0 : 0.0, tol);
      if constexpr (is_complex_v<TenT>) {
        TCICT_ASSERT_CLOSE(imag_part<TenT>(elem), 0.0, tol);
      }
    }
  }
#else
  (void)fix;
#endif
}

// --- random (shape {2,3}, every element came out of the generator) ---

// V1's entry for `random` is "Constructs a tensor filled by repeatedly invoking
// `gen()`". It fixes neither how many times gen is invoked nor which invocation
// fills which coordinate, so neither the call count nor a coordinate-to-value
// expectation is portably assertable. What survives is containment: every
// logical element must equal one of the values gen actually emitted.
template <typename TenT>
void test_random_inplace(tci_test_fixture<TenT>& fix) {
#ifndef TCICT_SKIP_RANDOM
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  tci::shape_t<TenT> shape = {2, 3};

  std::vector<tci::elem_t<TenT>> emitted;
  auto gen = [&]() -> tci::elem_t<TenT> {
    double val = static_cast<double>(emitted.size());
    auto elem = make_elem<TenT>(val, val + 0.5);
    emitted.push_back(elem);
    return elem;
  };

  TenT tensor;
  TCICT_ASSERT_NOTHROW(tensor = tci::template random<TenT>(ctx, shape, gen));
  TCICT_ASSERT(tci::shape(ctx, tensor) == shape);
  TCICT_ASSERT(tci::size(ctx, tensor) == 6);
  // Separates "gen was never invoked" from "an element does not match": with
  // an empty record the containment check below throws at the first
  // coordinate, reporting the mismatch rather than the cause.
  TCICT_ASSERT(!emitted.empty());

  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      auto elem = tci::get_elem(ctx, tensor, {i, j});
      bool contained = false;
      for (const auto& candidate : emitted) {
        if (std::abs(real_part<TenT>(elem) - real_part<TenT>(candidate)) < tol
            && std::abs(imag_part<TenT>(elem) - imag_part<TenT>(candidate)) < tol) {
          contained = true;
          break;
        }
      }
      TCICT_ASSERT(contained);
    }
  }
#else
  (void)fix;
#endif
}

// --- random (constant generator) ---

// The singleton case of the containment oracle above: with a constant
// generator the emission set has one member, so every element is pinned to a
// known value without assuming anything about invocation order or count.
template <typename TenT>
void test_random_outofplace(tci_test_fixture<TenT>& fix) {
#ifndef TCICT_SKIP_RANDOM
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  tci::shape_t<TenT> shape = {2, 2};
  auto constant = make_elem<TenT>(2.5, -1.25);
  std::size_t calls = 0;

  auto gen = [&]() -> tci::elem_t<TenT> {
    ++calls;
    return constant;
  };

  TenT tensor;
  TCICT_ASSERT_NOTHROW(tensor = tci::template random<TenT>(ctx, shape, gen));
  TCICT_ASSERT(tci::shape(ctx, tensor) == shape);
  TCICT_ASSERT(tci::size(ctx, tensor) == 4);
  // The value checks below compare against `constant`, so on their own they
  // cannot tell a backend that consulted gen from one that filled with the
  // same value without consulting it. The counter is what separates the two.
  TCICT_ASSERT(calls > 0);

  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 2; ++j) {
      auto elem = tci::get_elem(ctx, tensor, {i, j});
      TCICT_ASSERT_CLOSE(real_part<TenT>(elem), real_part<TenT>(constant), tol);
      if constexpr (is_complex_v<TenT>) {
        TCICT_ASSERT_CLOSE(imag_part<TenT>(elem), imag_part<TenT>(constant), tol);
      }
    }
  }
#else
  (void)fix;
#endif
}

// --- copy (in-place) ---

template <typename TenT>
void test_copy_inplace(tci_test_fixture<TenT>& fix) {
#ifndef TCICT_SKIP_COPY
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto a = tci::zeros<TenT>(ctx, {2, 3});
  tci::set_elem(ctx, a, {0, 0}, make_elem<TenT>(42.0, 13.0));
  tci::set_elem(ctx, a, {1, 2}, make_elem<TenT>(-5.5, 7.7));

  TenT b;
  TCICT_ASSERT_NOTHROW(b = tci::copy(ctx, a));

  auto val1 = tci::get_elem(ctx, b, {0, 0});
  TCICT_ASSERT_CLOSE(real_part<TenT>(val1), 42.0, tol);

  auto val2 = tci::get_elem(ctx, b, {1, 2});
  TCICT_ASSERT_CLOSE(real_part<TenT>(val2), -5.5, tol);

  if constexpr (is_complex_v<TenT>) {
    TCICT_ASSERT_CLOSE(imag_part<TenT>(val1), 13.0, tol);
    TCICT_ASSERT_CLOSE(imag_part<TenT>(val2), 7.7, tol);
  }

  TCICT_ASSERT(tci::shape(ctx, a) == tci::shape(ctx, b));
  TCICT_ASSERT(tci::order(ctx, a) == tci::order(ctx, b));
#else
  (void)fix;
#endif
}

// --- copy (out-of-place) ---

template <typename TenT>
void test_copy_outofplace(tci_test_fixture<TenT>& fix) {
#ifndef TCICT_SKIP_COPY
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto a = tci::zeros<TenT>(ctx, {3, 4, 2});
  tci::set_elem(ctx, a, {0, 1, 0}, make_elem<TenT>(1.23, 4.56));
  tci::set_elem(ctx, a, {2, 3, 1}, make_elem<TenT>(-9.87, 6.54));

  auto b = tci::copy(ctx, a);

  auto val1_a = tci::get_elem(ctx, a, {0, 1, 0});
  auto val1_b = tci::get_elem(ctx, b, {0, 1, 0});
  TCICT_ASSERT_CLOSE(real_part<TenT>(val1_a), real_part<TenT>(val1_b), tol);
  if constexpr (is_complex_v<TenT>) {
    TCICT_ASSERT_CLOSE(imag_part<TenT>(val1_a), imag_part<TenT>(val1_b), tol);
  }

  TCICT_ASSERT(tci::shape(ctx, a) == tci::shape(ctx, b));
  TCICT_ASSERT(tci::order(ctx, a) == tci::order(ctx, b));
#else
  (void)fix;
#endif
}

// --- copy independence ---

template <typename TenT>
void test_copy_independence(tci_test_fixture<TenT>& fix) {
#ifndef TCICT_SKIP_COPY
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto a = tci::zeros<TenT>(ctx, {2, 2});
  tci::set_elem(ctx, a, {0, 0}, make_elem<TenT>(100.0));

  auto b = tci::copy(ctx, a);
  tci::set_elem(ctx, a, {0, 0}, make_elem<TenT>(999.0));

  auto copy_val = tci::get_elem(ctx, b, {0, 0});
  TCICT_ASSERT_CLOSE(real_part<TenT>(copy_val), 100.0, tol);

  auto orig_val = tci::get_elem(ctx, a, {0, 0});
  TCICT_ASSERT_CLOSE(real_part<TenT>(orig_val), 999.0, tol);
#else
  (void)fix;
#endif
}

// --- copy single element ---

template <typename TenT>
void test_copy_single_element(tci_test_fixture<TenT>& fix) {
#ifndef TCICT_SKIP_COPY
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto a = tci::zeros<TenT>(ctx, {1});
  tci::set_elem(ctx, a, {0}, make_elem<TenT>(3.14, 2.71));

  auto b = tci::copy(ctx, a);

  auto val = tci::get_elem(ctx, b, {0});
  TCICT_ASSERT_CLOSE(real_part<TenT>(val), 3.14, tol);
  if constexpr (is_complex_v<TenT>) {
    TCICT_ASSERT_CLOSE(imag_part<TenT>(val), 2.71, tol);
  }
#else
  (void)fix;
#endif
}

// --- copy large tensor ---

template <typename TenT>
void test_copy_large(tci_test_fixture<TenT>& fix) {
#ifndef TCICT_SKIP_COPY
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto a = tci::zeros<TenT>(ctx, {10, 10, 10});
  tci::set_elem(ctx, a, {0, 0, 0}, make_elem<TenT>(1.0, 0.0));
  tci::set_elem(ctx, a, {9, 9, 9}, make_elem<TenT>(0.0, 1.0));

  auto b = tci::copy(ctx, a);

  auto corner1 = tci::get_elem(ctx, b, {0, 0, 0});
  TCICT_ASSERT_CLOSE(real_part<TenT>(corner1), 1.0, tol);

  auto corner2 = tci::get_elem(ctx, b, {9, 9, 9});
  if constexpr (is_complex_v<TenT>) {
    TCICT_ASSERT_CLOSE(imag_part<TenT>(corner1), 0.0, tol);
    TCICT_ASSERT_CLOSE(real_part<TenT>(corner2), 0.0, tol);
    TCICT_ASSERT_CLOSE(imag_part<TenT>(corner2), 1.0, tol);
  } else {
    // For real types, make_elem(0.0, 1.0) yields 0.0
    TCICT_ASSERT_CLOSE(real_part<TenT>(corner2), 0.0, tol);
  }

  TCICT_ASSERT(tci::shape(ctx, a) == tci::shape(ctx, b));
#else
  (void)fix;
#endif
}

// --- assign_from_range (row-major) ---

template <typename TenT>
void test_assign_from_range_row_major(tci_test_fixture<TenT>& fix) {
#ifndef TCICT_SKIP_ASSIGN_FROM_RANGE
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);

  std::vector<tci::elem_t<TenT>> container = {
      make_elem<TenT>(1.0), make_elem<TenT>(2.0), make_elem<TenT>(3.0),
      make_elem<TenT>(4.0), make_elem<TenT>(5.0), make_elem<TenT>(6.0)};

  auto coors2idx = row_major_2d<TenT>(3);

  tci::shape_t<TenT> shape = {2, 3};
  TenT tensor;
  TCICT_ASSERT_NOTHROW(
      tensor = tci::template assign_from_range<TenT>(ctx, shape, container.begin(),
                                                     std::move(coors2idx)));

  TCICT_ASSERT(tci::order(ctx, tensor) == 2);
  auto result_shape = tci::shape(ctx, tensor);
  TCICT_ASSERT(result_shape[0] == 2);
  TCICT_ASSERT(result_shape[1] == 3);

  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 0})), 1.0, tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 1})), 2.0, tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {1, 0})), 4.0, tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {1, 2})), 6.0, tol);
#else
  (void)fix;
#endif
}

// --- assign_from_range (column-major) ---

template <typename TenT>
void test_assign_from_range_column_major(tci_test_fixture<TenT>& fix) {
#ifndef TCICT_SKIP_ASSIGN_FROM_RANGE
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);

  // Column-major layout: [1, 3, 2, 4]
  std::vector<tci::elem_t<TenT>> container = {make_elem<TenT>(1.0), make_elem<TenT>(3.0),
                                               make_elem<TenT>(2.0), make_elem<TenT>(4.0)};

  std::function<std::ptrdiff_t(const tci::elem_coors_t<TenT>&)> coors2idx
      = [](const tci::elem_coors_t<TenT>& coors) -> std::ptrdiff_t {
    return coors[1] * 2 + coors[0];
  };

  tci::shape_t<TenT> shape = {2, 2};
  TenT tensor;
  TCICT_ASSERT_NOTHROW(
      tensor = tci::template assign_from_range<TenT>(ctx, shape, container.begin(),
                                                     std::move(coors2idx)));

  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 0})), 1.0, tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 1})), 2.0, tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {1, 0})), 3.0, tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {1, 1})), 4.0, tol);
#else
  (void)fix;
#endif
}

// --- allocate ---

template <typename TenT>
void test_allocate_3d(tci_test_fixture<TenT>& fix) {
#ifndef TCICT_SKIP_ALLOCATE
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  tci::shape_t<TenT> shape = {3, 4, 5};
  auto tensor = tci::allocate<TenT>(ctx, shape);

  auto tensor_shape = tci::shape(ctx, tensor);
  TCICT_ASSERT(tensor_shape.size() == 3);
  TCICT_ASSERT(tensor_shape[0] == 3);
  TCICT_ASSERT(tensor_shape[1] == 4);
  TCICT_ASSERT(tensor_shape[2] == 5);
  TCICT_ASSERT(tci::size(ctx, tensor) == 60);

  // Verify element type by round-tripping a value
  auto val = make_elem<TenT>(1.5, 2.5);
  tci::set_elem(ctx, tensor, {0, 0, 0}, val);
  auto got = tci::get_elem(ctx, tensor, {0, 0, 0});
  TCICT_ASSERT_CLOSE(real_part<TenT>(got), real_part<TenT>(val), tol);
  if constexpr (is_complex_v<TenT>) {
    TCICT_ASSERT_CLOSE(imag_part<TenT>(got), imag_part<TenT>(val), tol);
  }
#else
  (void)fix;
#endif
}

template <typename TenT>
void test_allocate_2d(tci_test_fixture<TenT>& fix) {
#ifndef TCICT_SKIP_ALLOCATE
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  tci::shape_t<TenT> shape = {2, 3};
  auto tensor = tci::allocate<TenT>(ctx, shape);

  auto tensor_shape = tci::shape(ctx, tensor);
  TCICT_ASSERT(tensor_shape.size() == 2);
  TCICT_ASSERT(tensor_shape[0] == 2);
  TCICT_ASSERT(tensor_shape[1] == 3);
  TCICT_ASSERT(tci::size(ctx, tensor) == 6);

  // Verify element type by round-tripping a value
  auto val = make_elem<TenT>(1.5, 2.5);
  tci::set_elem(ctx, tensor, {0, 0}, val);
  auto got = tci::get_elem(ctx, tensor, {0, 0});
  TCICT_ASSERT_CLOSE(real_part<TenT>(got), real_part<TenT>(val), tol);
  if constexpr (is_complex_v<TenT>) {
    TCICT_ASSERT_CLOSE(imag_part<TenT>(got), imag_part<TenT>(val), tol);
  }
#else
  (void)fix;
#endif
}

template <typename TenT>
void test_allocate_1d(tci_test_fixture<TenT>& fix) {
#ifndef TCICT_SKIP_ALLOCATE
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  tci::shape_t<TenT> shape = {10};
  auto tensor = tci::allocate<TenT>(ctx, shape);

  auto tensor_shape = tci::shape(ctx, tensor);
  TCICT_ASSERT(tensor_shape.size() == 1);
  TCICT_ASSERT(tensor_shape[0] == 10);
  TCICT_ASSERT(tci::size(ctx, tensor) == 10);

  // Verify element type by round-tripping a value
  auto val = make_elem<TenT>(1.5, 2.5);
  tci::set_elem(ctx, tensor, {0}, val);
  auto got = tci::get_elem(ctx, tensor, {0});
  TCICT_ASSERT_CLOSE(real_part<TenT>(got), real_part<TenT>(val), tol);
  if constexpr (is_complex_v<TenT>) {
    TCICT_ASSERT_CLOSE(imag_part<TenT>(got), imag_part<TenT>(val), tol);
  }
#else
  (void)fix;
#endif
}

// --- clear ---

template <typename TenT>
void test_clear_basic(tci_test_fixture<TenT>& fix) {
#ifndef TCICT_SKIP_CLEAR
  auto& ctx = fix.context();
  auto tensor = tci::eye<TenT>(ctx, 3);
  TCICT_ASSERT(tci::size(ctx, tensor) == 9);
  TCICT_ASSERT_NOTHROW(tci::clear(ctx, tensor));
  TCICT_ASSERT(tci::order(ctx, tensor) == 0);
  TCICT_ASSERT(tci::shape(ctx, tensor).empty());
#else
  (void)fix;
#endif
}

template <typename TenT>
void test_clear_empty(tci_test_fixture<TenT>& fix) {
#ifndef TCICT_SKIP_CLEAR
  auto& ctx = fix.context();
  TenT empty_tensor;
  TCICT_ASSERT_NOTHROW(tci::clear(ctx, empty_tensor));
  TCICT_ASSERT(tci::order(ctx, empty_tensor) == 0);
#else
  (void)fix;
#endif
}

template <typename TenT>
void test_clear_and_reallocate(tci_test_fixture<TenT>& fix) {
#ifndef TCICT_SKIP_CLEAR
  auto& ctx = fix.context();
  auto tensor = tci::eye<TenT>(ctx, 2);
  tci::clear(ctx, tensor);
  TCICT_ASSERT(tci::order(ctx, tensor) == 0);
  TCICT_ASSERT_NOTHROW(tensor = tci::allocate<TenT>(ctx, {2, 2}));
  TCICT_ASSERT(tci::order(ctx, tensor) == 2);
#else
  (void)fix;
#endif
}

// --- move (in-place) ---

template <typename TenT>
void test_move_inplace(tci_test_fixture<TenT>& fix) {
#ifndef TCICT_SKIP_MOVE
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto source = tci::eye<TenT>(ctx, 3);

  auto source_size = tci::size(ctx, source);
  auto source_shape = tci::shape(ctx, source);
  auto original_elem = tci::get_elem(ctx, source, {0, 0});

  auto destination = tci::move(ctx, source);

  TCICT_ASSERT(tci::size(ctx, destination) == source_size);
  TCICT_ASSERT(tci::shape(ctx, destination) == source_shape);
  auto dest_elem = tci::get_elem(ctx, destination, {0, 0});
  TCICT_ASSERT_CLOSE(real_part<TenT>(dest_elem), real_part<TenT>(original_elem), tol);
  if constexpr (is_complex_v<TenT>) {
    TCICT_ASSERT_CLOSE(imag_part<TenT>(dest_elem), imag_part<TenT>(original_elem), tol);
  }

  // Verify source is invalidated after move
  TCICT_ASSERT(tci::order(ctx, source) == 0);
#else
  (void)fix;
#endif
}

// --- move (out-of-place) ---

template <typename TenT>
void test_move_outofplace(tci_test_fixture<TenT>& fix) {
#ifndef TCICT_SKIP_MOVE
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto source = tci::eye<TenT>(ctx, 2);

  auto source_size = tci::size(ctx, source);
  auto source_shape = tci::shape(ctx, source);
  auto original_elem = tci::get_elem(ctx, source, {1, 1});

  auto result = tci::move(ctx, source);

  TCICT_ASSERT(tci::size(ctx, result) == source_size);
  TCICT_ASSERT(tci::shape(ctx, result) == source_shape);
  auto result_elem = tci::get_elem(ctx, result, {1, 1});
  TCICT_ASSERT_CLOSE(real_part<TenT>(result_elem), real_part<TenT>(original_elem), tol);
  if constexpr (is_complex_v<TenT>) {
    TCICT_ASSERT_CLOSE(imag_part<TenT>(result_elem), imag_part<TenT>(original_elem), tol);
  }

  // Verify source is invalidated after move
  TCICT_ASSERT(tci::order(ctx, source) == 0);
#else
  (void)fix;
#endif
}

// --- move empty tensor ---

template <typename TenT>
void test_move_empty(tci_test_fixture<TenT>& fix) {
#ifndef TCICT_SKIP_MOVE
  auto& ctx = fix.context();
  TenT empty_source, empty_result;
  TCICT_ASSERT_NOTHROW(empty_result = tci::move(ctx, empty_source));
  TCICT_ASSERT(tci::order(ctx, empty_source) == 0);
#else
  (void)fix;
#endif
}

// --- move preserves values ---

template <typename TenT>
void test_move_preserves_values(tci_test_fixture<TenT>& fix) {
#ifndef TCICT_SKIP_MOVE
  auto& ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto source = tci::zeros<TenT>(ctx, {2, 3});

  auto val1 = make_elem<TenT>(2.5, 1.5);
  auto val2 = make_elem<TenT>(3.7, -2.1);
  tci::set_elem(ctx, source, {0, 1}, val1);
  tci::set_elem(ctx, source, {1, 2}, val2);

  auto destination = tci::move(ctx, source);

  auto moved_val1 = tci::get_elem(ctx, destination, {0, 1});
  auto moved_val2 = tci::get_elem(ctx, destination, {1, 2});

  TCICT_ASSERT_CLOSE(real_part<TenT>(moved_val1), real_part<TenT>(val1), tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(moved_val2), real_part<TenT>(val2), tol);
  if constexpr (is_complex_v<TenT>) {
    TCICT_ASSERT_CLOSE(imag_part<TenT>(moved_val1), imag_part<TenT>(val1), tol);
    TCICT_ASSERT_CLOSE(imag_part<TenT>(moved_val2), imag_part<TenT>(val2), tol);
  }

  // Verify source is invalidated after move
  TCICT_ASSERT(tci::order(ctx, source) == 0);
#else
  (void)fix;
#endif
}

}}  // namespace tcict::tests

// Bulk registration helper: invokes X(..., "category", test_fn) once per test.
// See include/tcict/adapters/doctest.h for usage.
#define TCICT_FOREACH_CONSTRUCTION_TEST_ALL_TYPES(X, ...) \
  X(__VA_ARGS__, "construction", test_zeros) \
  X(__VA_ARGS__, "construction", test_fill) \
  X(__VA_ARGS__, "construction", test_eye) \
  X(__VA_ARGS__, "construction", test_eye_to_range) \
  X(__VA_ARGS__, "construction", test_random_inplace) \
  X(__VA_ARGS__, "construction", test_random_outofplace) \
  X(__VA_ARGS__, "construction", test_copy_inplace) \
  X(__VA_ARGS__, "construction", test_copy_outofplace) \
  X(__VA_ARGS__, "construction", test_copy_independence) \
  X(__VA_ARGS__, "construction", test_copy_single_element) \
  X(__VA_ARGS__, "construction", test_copy_large) \
  X(__VA_ARGS__, "construction", test_assign_from_range_row_major) \
  X(__VA_ARGS__, "construction", test_assign_from_range_column_major) \
  X(__VA_ARGS__, "construction", test_allocate_3d) \
  X(__VA_ARGS__, "construction", test_allocate_2d) \
  X(__VA_ARGS__, "construction", test_allocate_1d) \
  X(__VA_ARGS__, "construction", test_clear_basic) \
  X(__VA_ARGS__, "construction", test_clear_empty) \
  X(__VA_ARGS__, "construction", test_clear_and_reallocate) \
  X(__VA_ARGS__, "construction", test_move_inplace) \
  X(__VA_ARGS__, "construction", test_move_outofplace) \
  X(__VA_ARGS__, "construction", test_move_empty) \
  X(__VA_ARGS__, "construction", test_move_preserves_values)
