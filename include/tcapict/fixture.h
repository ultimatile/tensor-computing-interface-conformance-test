#pragma once

#include <tcapi/tcapi.h>

#include <cstddef>
#include <limits>
#include <type_traits>

namespace tcapict {

/// Test fixture that manages a TCAPI context.
/// Backends may specialize this for custom context creation (e.g. GPU) or epsilon.
template <typename TenT>
struct tcapi_test_fixture {
  tcapi::context_handle_t<TenT> ctx;

  tcapi_test_fixture() { tcapi::create_context(ctx); }
  ~tcapi_test_fixture() { tcapi::destroy_context(ctx); }

  tcapi_test_fixture(const tcapi_test_fixture&) = delete;
  tcapi_test_fixture& operator=(const tcapi_test_fixture&) = delete;

  auto& context() { return ctx; }

  /// Comparison tolerance sized to the element type's real precision.
  /// The known-type branches match values empirically sufficient for the
  /// small test fixtures in this suite; the fallback scales machine epsilon
  /// to absorb O(N) accumulation errors for unfamiliar real_t types.
  auto epsilon() -> tcapi::real_t<TenT> {
    using real_type = tcapi::real_t<TenT>;
    if constexpr (std::is_same_v<real_type, float>) {
      return 1e-5F;
    } else if constexpr (std::is_same_v<real_type, double>) {
      return 1e-10;
    } else if constexpr (std::is_same_v<real_type, long double>) {
      return 1e-15L;
    } else {
      return std::numeric_limits<real_type>::epsilon() * static_cast<real_type>(100);
    }
  }
};

/// Backward-error categories for numerical comparisons. Tolerances scale with
/// the operation's expected error growth; see the LAPACK Users' Guide,
/// Chapter 4 "Accuracy and Stability" (https://www.netlib.org/lapack/lug/)
/// for the standard schedule.
enum class tol_category {
  /// No floating-point arithmetic (copy, reshape, fill, zeros, eye).
  exact,
  /// O(eps): single multiply / assignment (scale, set_elem / get_elem).
  elementwise,
  /// O(sqrt(N) * eps): pairwise / Kahan summation (norm, trace, linear_combine).
  reduction,
  /// O(N * eps * cond): BLAS / LAPACK backward stability (QR, LQ, SVD, eig, eigh).
  factorization,
  /// O(N^k * eps): algorithm-dependent composite (exp, inverse).
  iterative,
};

/// Comparison tolerance for an operation in a given backward-error category.
///
/// Currently returns category-specific constants that match the test
/// suite's pre-existing hardcoded multipliers; the asymptotic scalings
/// claimed in `tol_category`'s docstrings are the *target* error classes,
/// not yet realized in this implementation. The `N` parameter is reserved
/// for future N-scaling of categories whose asymptotic growth is
/// non-constant (reduction, factorization, iterative) and is currently
/// ignored. Backends that want a different tolerance schedule specialize
/// the fixture's `epsilon()`; the helper composes its result from
/// `fix.epsilon()`, so any backend-defined epsilon flows through.
template <typename TenT>
auto tolerance(tcapi_test_fixture<TenT>& fix, tol_category cat,
               std::size_t /*N*/ = 1) -> tcapi::real_t<TenT> {
  using real_type = tcapi::real_t<TenT>;
  const auto eps = fix.epsilon();
  switch (cat) {
    case tol_category::exact:
      return real_type{0};
    case tol_category::elementwise:
      return eps;
    case tol_category::reduction:
      return eps;
    case tol_category::factorization:
      return eps * real_type{100};
    case tol_category::iterative:
      return eps * real_type{100};
    default:
      // Defensive fallback for out-of-range `tol_category` values produced by
      // a `static_cast` from an integer that does not match a named case.
      return eps;
  }
}

}  // namespace tcapict
