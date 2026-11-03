#pragma once

#include <tcict/assertion.h>
#include <tcict/elem_helper.h>
#include <tcict/fixture.h>

#include <cstddef>

namespace tcict {

// Compiled exactly when at least one of their callers is: the concatenate,
// stack and linear_combine tests.
#if !defined(TCICT_SKIP_CONCATENATE) || !defined(TCICT_SKIP_STACK) ||         \
    !defined(TCICT_SKIP_LINEAR_COMBINE)

namespace detail {

// Calls `f(coors)` for every coordinate of `shape`, the last bond advancing
// fastest.
template <typename TenT, typename F>
void for_each_coors(const tci::shape_t<TenT> &shape, F f) {
  const std::size_t order = shape.size();
  for (std::size_t b = 0; b < order; ++b) {
    if (shape[b] == 0) {
      return;
    }
  }
  tci::elem_coors_t<TenT> coors(order, 0);
  while (true) {
    f(static_cast<const tci::elem_coors_t<TenT> &>(coors));
    std::size_t b = order;
    while (true) {
      if (b == 0) {
        return;
      }
      --b;
      ++coors[b];
      if (static_cast<std::size_t>(coors[b]) <
          static_cast<std::size_t>(shape[b])) {
        break;
      }
      coors[b] = 0;
    }
  }
}

} // namespace detail

/// Returns the function sending a coordinate of `shape` to `offset` plus the
/// coordinate's position in a row-major walk of `shape`, so distinct
/// coordinates of `shape` get distinct values `offset`, `offset + 1`, ...
/// The position is a function of the coordinate alone; it says nothing about
/// how a backend lays the elements out. Asserts that the coordinate has
/// `shape`'s order and lies inside `shape`.
template <typename TenT>
auto ramp_over(const tci::shape_t<TenT> &shape, double offset) {
  return [shape, offset](const tci::elem_coors_t<TenT> &coors) {
    TCICT_ASSERT(coors.size() == shape.size());
    double position = 0.0;
    for (std::size_t b = 0; b < shape.size(); ++b) {
      TCICT_ASSERT(static_cast<std::size_t>(coors[b]) <
                   static_cast<std::size_t>(shape[b]));
      position = position * static_cast<double>(shape[b]) +
                 static_cast<double>(coors[b]);
    }
    return offset + position;
  };
}

/// `coors` with bond `bond`'s coordinate lowered by `by`: the coordinate an
/// element of a concatenation result has in the input it came from.
template <typename Coors>
Coors shifted(const Coors &coors, std::size_t bond, std::size_t by) {
  TCICT_ASSERT(bond < coors.size());
  TCICT_ASSERT(static_cast<std::size_t>(coors[bond]) >= by);
  Coors out = coors;
  out[bond] = static_cast<typename Coors::value_type>(
      static_cast<std::size_t>(coors[bond]) - by);
  return out;
}

/// `coors` without bond `bond`: the coordinate an element of a stack result
/// has in the input `coors[bond]` it came from.
template <typename Coors>
Coors without_bond(const Coors &coors, std::size_t bond) {
  TCICT_ASSERT(bond < coors.size());
  Coors out;
  for (std::size_t b = 0; b < coors.size(); ++b) {
    if (b != bond) {
      out.push_back(coors[b]);
    }
  }
  return out;
}

/// Sets the element at every coordinate `coors` of `tensor` to the value with
/// real part `v = value_at(coors)` and, for a complex element type, imaginary
/// part `-0.5 * v`, so that an operation losing or conjugating imaginary parts
/// is visible. Pairs with `expect_elems`, so a test can build an input and
/// check it with the same function; real-coefficient linear operations keep
/// the imaginary part at `-0.5` times the real part.
template <typename TenT, typename ValueAt>
void set_elems(tci_test_fixture<TenT> &fix, TenT &tensor, ValueAt value_at) {
  auto &ctx = fix.context();
  detail::for_each_coors<TenT>(
      tci::shape(ctx, tensor), [&](const tci::elem_coors_t<TenT> &coors) {
        const double v = static_cast<double>(value_at(coors));
        tci::set_elem(ctx, tensor, coors, make_elem<TenT>(v, -0.5 * v));
      });
}

/// Asserts that `tensor` has `shape`, then that the element at every
/// coordinate of that shape has real part `v = value_at(coors)` and, for a
/// complex element type, imaginary part `-0.5 * v`, as `set_elems` writes
/// them, each within `tolerance(fix, cat, n)`. The shape is asserted first, so
/// the sweep reads only coordinates the tensor has.
template <typename TenT, typename ValueAt>
void expect_elems(tci_test_fixture<TenT> &fix, const TenT &tensor,
                  const tci::shape_t<TenT> &shape, ValueAt value_at,
                  tol_category cat = tol_category::elementwise,
                  std::size_t n = 1) {
  auto &ctx = fix.context();
  auto tol = tolerance(fix, cat, n);
  TCICT_ASSERT(tci::shape(ctx, tensor) == shape);
  detail::for_each_coors<TenT>(
      shape, [&](const tci::elem_coors_t<TenT> &coors) {
        auto elem = tci::get_elem(ctx, tensor, coors);
        const double v = static_cast<double>(value_at(coors));
        TCICT_ASSERT_CLOSE(real_part<TenT>(elem), v, tol);
        if constexpr (is_complex_v<TenT>) {
          TCICT_ASSERT_CLOSE(imag_part<TenT>(elem), -0.5 * v, tol);
        }
      });
}

#endif

} // namespace tcict
