#pragma once

#include <tcict/assertion.h>
#include <tcict/elem_helper.h>
#include <tcict/fixture.h>
#include <tcict/skip.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <sstream>
#include <type_traits>

namespace tcict {

// Guarded by TCICT_HAS_ELEM_SWEEP (include/tcict/skip.h), the element-API
// condition every test using these helpers also requires: the helpers call
// set_elem, shape and get_elem, and those tests build their inputs with zeros.
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
  if constexpr (std::is_signed_v<typename Coors::value_type>) {
    TCICT_ASSERT(coors[bond] >= 0);
  }
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

/// The element value `TCICT_SET_ELEMS` writes for the real value `v`: real
/// part `v` and, for a complex element type, imaginary part `-0.5 * v`.
template <typename TenT> std::complex<double> written(double v) {
  return {v, is_complex_v<TenT> ? -0.5 * v : 0.0};
}

/// Sets the element at every coordinate `coors` of `tensor` to the value with
/// real part `v = value_at(coors)` and, for a complex element type, imaginary
/// part `-0.5 * v`, so that an operation losing or conjugating imaginary parts
/// is visible. Pairs with `TCICT_EXPECT_ELEMS`, so a test can build an input and
/// check it with the same function; real-coefficient linear operations keep
/// the imaginary part at `-0.5` times the real part. A std::exception from
/// shape, set_elem or `value_at`, an assertion failing in `value_at` included,
/// is reported at `file`:`line` as detail::nothrow_call reports it; call it
/// through TCICT_SET_ELEMS, which passes the caller's location.
template <typename TenT, typename ValueAt>
void set_elems_at(const char *file, int line, tci_test_fixture<TenT> &fix,
                  TenT &tensor, ValueAt value_at) {
  auto &ctx = fix.context();
  const auto shape = detail::nothrow_call(
      file, line, [&] { return tci::shape(ctx, tensor); });
  detail::for_each_coors<TenT>(
      shape, [&](const tci::elem_coors_t<TenT> &coors) {
        const auto w = written<TenT>(static_cast<double>(
            detail::nothrow_call(file, line, [&] { return value_at(coors); })));
        detail::nothrow_call(file, line, [&] {
          tci::set_elem(ctx, tensor, coors,
                        make_elem<TenT>(w.real(), w.imag()));
        });
      });
}

/// Asserts that `tensor` has `shape`, then that the element at every
/// coordinate of that shape equals `value_at(coors)` within
/// `tolerance(fix, cat, n)`, part by part; a zero tolerance, which
/// `tol_category::exact` gives, admits only equality, and any other tolerance
/// admits only a difference below it. A real
/// `value_at` result `v` stands for `written<TenT>(v)`, the value
/// `TCICT_SET_ELEMS` writes; a `std::complex<double>` result is compared as it
/// is. For a real element type, the expected imaginary part is compared
/// against zero. The
/// shape is asserted first, so the sweep reads only coordinates the tensor has.
/// A shape or element mismatch is reported at `file`:`line`. So is a
/// std::exception from shape, get_elem or `value_at`, an assertion failing in
/// `value_at` included, in the form detail::nothrow_call reports it. Call it
/// through TCICT_EXPECT_ELEMS, which passes the caller's location.
template <typename TenT, typename ValueAt>
void expect_elems_at(const char *file, int line, tci_test_fixture<TenT> &fix,
                     const TenT &tensor, const tci::shape_t<TenT> &shape,
                     ValueAt value_at,
                     tol_category cat = tol_category::elementwise,
                     std::size_t n = 1) {
  auto &ctx = fix.context();
  auto tol = tolerance(fix, cat, n);
  auto shape_text = [](const auto &dims) {
    std::ostringstream oss;
    oss << "{";
    for (std::size_t b = 0; b < dims.size(); ++b) {
      oss << (b == 0 ? "" : ", ") << dims[b];
    }
    oss << "}";
    return oss.str();
  };
  const auto actual_shape = detail::nothrow_call(
      file, line, [&] { return tci::shape(ctx, tensor); });
  if (!(actual_shape == shape)) {
    detail::fail_at(file, line,
                    "shape " + shape_text(actual_shape) + " != expected " +
                        shape_text(shape));
  }
  detail::for_each_coors<TenT>(
      shape, [&](const tci::elem_coors_t<TenT> &coors) {
        const auto elem = detail::nothrow_call(
            file, line, [&] { return tci::get_elem(ctx, tensor, coors); });
        const auto value =
            detail::nothrow_call(file, line, [&] { return value_at(coors); });
        std::complex<double> expected;
        if constexpr (std::is_arithmetic_v<std::decay_t<decltype(value)>>) {
          expected = written<TenT>(static_cast<double>(value));
        } else {
          expected = value;
        }
        // A zero tolerance (tol_category::exact) admits only equality; any
        // other admits only a difference below it.
        auto check = [&](double actual, double wanted, const char *part) {
          const double diff = std::abs(actual - wanted);
          const bool ok = tol == 0 ? actual == wanted : diff < tol;
          if (!ok) {
            std::ostringstream oss;
            oss << std::setprecision(std::numeric_limits<double>::max_digits10)
                << part << " part at " << shape_text(coors)
                << " differs: " << actual << " != " << wanted
                << " (|difference| " << diff << ", tolerance " << tol << ")";
            detail::fail_at(file, line, oss.str());
          }
        };
        check(real_part<TenT>(elem), expected.real(), "real");
        if constexpr (is_complex_v<TenT>) {
          check(imag_part<TenT>(elem), expected.imag(), "imaginary");
        } else {
          // A real element's imaginary part is zero.
          check(0.0, expected.imag(), "imaginary");
        }
      });
}

#endif

} // namespace tcict

/// expect_elems_at with the caller's file and line:
/// TCICT_EXPECT_ELEMS(fix, tensor, shape, value_at[, cat[, n]]). The line is
/// one of the lines the invocation spans; which one, for an invocation spanning
/// several, depends on the compiler.
#define TCICT_EXPECT_ELEMS(...) ::tcict::expect_elems_at(__FILE__, __LINE__, __VA_ARGS__)

/// set_elems_at with the caller's file and line:
/// TCICT_SET_ELEMS(fix, tensor, value_at), the line chosen as for
/// TCICT_EXPECT_ELEMS.
#define TCICT_SET_ELEMS(...) ::tcict::set_elems_at(__FILE__, __LINE__, __VA_ARGS__)
