#pragma once

#include <tcict/assertion.h>
#include <tcict/elem_helper.h>
#include <tcict/fixture.h>
#include <tcict/skip.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <sstream>
#include <type_traits>

namespace tcict {

// Guarded by TCICT_HAS_ELEM_SWEEP (include/tcict/skip.h), the condition of the
// tests that use these helpers: the helpers call set_elem, shape and get_elem,
// and those tests also build their inputs with zeros.
#if defined(TCICT_HAS_ELEM_SWEEP)

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

/// The element value `set_elems` writes for the real value `v`: real part `v`
/// and, for a complex element type, imaginary part `-0.5 * v`.
template <typename TenT> std::complex<double> written(double v) {
  return {v, is_complex_v<TenT> ? -0.5 * v : 0.0};
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
        const auto w = written<TenT>(static_cast<double>(value_at(coors)));
        tci::set_elem(ctx, tensor, coors, make_elem<TenT>(w.real(), w.imag()));
      });
}

/// Asserts that `tensor` has `shape`, then that the element at every
/// coordinate of that shape equals `value_at(coors)` within
/// `tolerance(fix, cat, n)`, part by part; `tol_category::exact` compares for
/// equality instead. A real `value_at` result `v` stands
/// for `written<TenT>(v)`, the value `set_elems` writes; a
/// `std::complex<double>` result is compared as it is. The imaginary part is
/// compared only for a complex element type. The shape is asserted first, so
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
        const auto value = value_at(coors);
        std::complex<double> expected;
        if constexpr (std::is_arithmetic_v<std::decay_t<decltype(value)>>) {
          expected = written<TenT>(static_cast<double>(value));
        } else {
          expected = value;
        }
        // A zero tolerance (tol_category::exact) admits only equality.
        auto check = [&](double actual, double wanted, const char *part) {
          if (!(std::abs(actual - wanted) <= tol)) {
            std::ostringstream oss_;
            oss_ << __FILE__ << ":" << __LINE__ << ": " << part
                 << " part differs: " << actual << " != " << wanted
                 << " (tolerance " << tol << ")";
            throw ::tcict::assertion_error(oss_.str());
          }
        };
        check(real_part<TenT>(elem), expected.real(), "real");
        if constexpr (is_complex_v<TenT>) {
          check(imag_part<TenT>(elem), expected.imag(), "imaginary");
        }
      });
}

#endif

} // namespace tcict
