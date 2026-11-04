#pragma once

// Check of TCAPI V1's diagonal tensor: a second-order tensor of shape {n, n}
// whose off-diagonal elements are zero. See "Diagonal tensors" in
// CONTRIBUTING.md.

#include <tcict/assertion.h>
#include <tcict/elem_helper.h>
#include <tcict/skip.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace tcict {

/// How expect_diagonal_at compares the diagonal with the expected values, each
/// comparison within its tolerance.
enum class diag_order {
  /// The element at {i, i} is compared with expected[i].
  as_listed,
  /// The real parts of the diagonal, sorted, are compared with the expected
  /// values, sorted, and each imaginary part with zero. For outputs whose
  /// order V1 leaves open, such as eig's on a matrix with real eigenvalues.
  any,
};

// Guarded by TCICT_HAS_DIAGONAL_CHECK (include/tcict/skip.h), the APIs it
// calls, which every test calling it also requires.
#if defined(TCICT_HAS_DIAGONAL_CHECK)

/// Asserts that `tensor`, of any tensor type, is a diagonal tensor holding
/// `expected` on its diagonal, with n = expected.size(): order 2, shape
/// {n, n} and size n * n, checked in that order before any element is read,
/// then every element. An off-diagonal element must be exactly zero in both
/// parts. A diagonal element's real part must differ from the expected value
/// by less than `tol`, or equal it when `tol` is zero, and its imaginary part,
/// for a complex element type, must likewise be zero. `order` says which
/// diagonal element each expected value is compared with. A mismatch, or a
/// std::exception from order, shape, size or get_elem, is reported at
/// `file`:`line`. Call it through TCICT_EXPECT_DIAGONAL, which passes the
/// caller's location.
template <typename Ctx, typename DiagT>
void expect_diagonal_at(const char *file, int line, Ctx &ctx,
                        const DiagT &tensor,
                        const std::vector<double> &expected, double tol,
                        diag_order order = diag_order::as_listed) {
  const std::size_t n = expected.size();
  if (n == 0) {
    detail::fail_at(file, line, "expect_diagonal_at: no expected values given");
  }
  auto text = [](double v) {
    std::ostringstream oss;
    oss << std::setprecision(std::numeric_limits<double>::max_digits10) << v;
    return oss.str();
  };
  auto coors_text = [](std::size_t i, std::size_t j) {
    return "{" + std::to_string(i) + ", " + std::to_string(j) + "}";
  };

  const auto actual_order =
      detail::nothrow_call(file, line, [&] { return tci::order(ctx, tensor); });
  if (actual_order != 2) {
    std::ostringstream oss;
    oss << "order " << actual_order << " != expected 2";
    detail::fail_at(file, line, oss.str());
  }
  const auto shape =
      detail::nothrow_call(file, line, [&] { return tci::shape(ctx, tensor); });
  if (shape.size() != 2 || static_cast<std::size_t>(shape[0]) != n ||
      static_cast<std::size_t>(shape[1]) != n) {
    std::ostringstream oss;
    oss << "shape {";
    for (std::size_t b = 0; b < shape.size(); ++b) {
      oss << (b == 0 ? "" : ", ") << shape[b];
    }
    oss << "} != expected {" << n << ", " << n << "}";
    detail::fail_at(file, line, oss.str());
  }
  const auto size =
      detail::nothrow_call(file, line, [&] { return tci::size(ctx, tensor); });
  if (static_cast<std::size_t>(size) != n * n) {
    std::ostringstream oss;
    oss << "size " << size << " != expected " << n * n;
    detail::fail_at(file, line, oss.str());
  }

  auto close = [tol](double actual, double wanted) {
    return tol == 0 ? actual == wanted : std::abs(actual - wanted) < tol;
  };
  auto fail_part = [&](const char *part, std::size_t i, std::size_t j,
                       double actual, double wanted) {
    detail::fail_at(file, line,
                    std::string(part) + " part at " + coors_text(i, j) +
                        " differs: " + text(actual) + " != " + text(wanted) +
                        " (|difference| " + text(std::abs(actual - wanted)) +
                        ", tolerance " + text(tol) + ")");
  };

  std::vector<double> diagonal(n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const auto elem = detail::nothrow_call(file, line, [&] {
        return tci::get_elem(ctx, tensor, coors_of<DiagT>(i, j));
      });
      const double re = real_part<DiagT>(elem);
      const double im = imag_part<DiagT>(elem);
      if (i != j) {
        // V1: a diagonal tensor's off-diagonal elements are zero, so no
        // tolerance applies.
        if (re != 0.0 || im != 0.0) {
          std::string value = text(re);
          if constexpr (is_complex_v<DiagT>) {
            value = "(" + value + ", " + text(im) + ")";
          }
          detail::fail_at(file, line,
                          "off-diagonal element at " + coors_text(i, j) +
                              " is " + value + ", not exactly zero");
        }
        continue;
      }
      if (!close(im, 0.0)) {
        fail_part("imaginary", i, i, im, 0.0);
      }
      if (order == diag_order::as_listed && !close(re, expected[i])) {
        fail_part("real", i, i, re, expected[i]);
      }
      diagonal[i] = re;
    }
  }

  if (order == diag_order::any) {
    // A NaN breaks the strict weak ordering std::sort requires, on either
    // side of the comparison.
    for (std::size_t i = 0; i < n; ++i) {
      if (std::isnan(diagonal[i])) {
        detail::fail_at(file, line,
                        "real part at " + coors_text(i, i) + " is NaN");
      }
      if (std::isnan(expected[i])) {
        detail::fail_at(file, line, "expected value " + std::to_string(i) +
                                        " is NaN");
      }
    }
    auto wanted = expected;
    std::sort(diagonal.begin(), diagonal.end());
    std::sort(wanted.begin(), wanted.end());
    for (std::size_t k = 0; k < n; ++k) {
      if (!close(diagonal[k], wanted[k])) {
        detail::fail_at(file, line,
                        "sorted diagonal real parts differ at position " +
                            std::to_string(k) + ": " + text(diagonal[k]) +
                            " != " + text(wanted[k]) + " (tolerance " +
                            text(tol) + ")");
      }
    }
  }
}

/// The real parts of the diagonal of `tensor`, of any tensor type, read at
/// {i, i}, after asserting that it is square and second-order; for comparing
/// two diagonal tensors with each other through expect_diagonal_at. A
/// std::exception from shape or get_elem is reported at `file`:`line`. Call it
/// through TCICT_DIAGONAL_OF, which passes the caller's location.
template <typename Ctx, typename DiagT>
std::vector<double> diagonal_of_at(const char *file, int line, Ctx &ctx,
                                   const DiagT &tensor) {
  const auto shape =
      detail::nothrow_call(file, line, [&] { return tci::shape(ctx, tensor); });
  if (shape.size() != 2 || shape[0] != shape[1]) {
    detail::fail_at(file, line,
                    "diagonal_of_at: not a square second-order tensor");
  }
  const auto n = static_cast<std::size_t>(shape[0]);
  if (n == 0) {
    detail::fail_at(file, line, "diagonal_of_at: the tensor has shape {0, 0}");
  }
  std::vector<double> diagonal(n);
  for (std::size_t i = 0; i < n; ++i) {
    diagonal[i] = real_part<DiagT>(detail::nothrow_call(file, line, [&] {
      return tci::get_elem(ctx, tensor, coors_of<DiagT>(i, i));
    }));
  }
  return diagonal;
}

/// expect_diagonal_at with the expected values in an array, as the diag tests
/// hold them; a braced list such as {3.0, 1.0} also binds here.
template <typename Ctx, typename DiagT, std::size_t N>
void expect_diagonal_at(const char *file, int line, Ctx &ctx,
                        const DiagT &tensor, const double (&expected)[N],
                        double tol, diag_order order = diag_order::as_listed) {
  expect_diagonal_at(file, line, ctx, tensor,
                     std::vector<double>(expected, expected + N), tol, order);
}

#endif

} // namespace tcict

/// expect_diagonal_at with the caller's file and line:
/// TCICT_EXPECT_DIAGONAL(ctx, tensor, expected, tol[, order]). Variadic so a
/// braced list of expected values passes through the preprocessor.
#define TCICT_EXPECT_DIAGONAL(...)                                             \
  ::tcict::expect_diagonal_at(__FILE__, __LINE__, __VA_ARGS__)

/// diagonal_of_at with the caller's file and line:
/// TCICT_DIAGONAL_OF(ctx, tensor).
#define TCICT_DIAGONAL_OF(...)                                                 \
  ::tcict::diagonal_of_at(__FILE__, __LINE__, __VA_ARGS__)
