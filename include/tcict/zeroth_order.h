#pragma once

// Helpers for the tests of TCAPI V1's zeroth-order tensor: order 0, shape {},
// size 1, its sole element read and written at the empty coordinate {}.

#include <tcict/assertion.h>
#include <tcict/elem_helper.h>
#include <tcict/elem_sweep.h>
#include <tcict/fixture.h>
#include <tcict/skip.h>

#include <cstddef>
#include <sstream>

namespace tcict {

namespace detail {

// Sends the empty coordinate to index 0. A non-empty coordinate sets
// `malformed` and also maps to index 0, which keeps the backend's access inside
// a one-element range; the wrappers below fail on `malformed`. See "Callbacks
// the backend invokes" in CONTRIBUTING.md.
template <typename TenT> auto map_0d(bool &malformed) {
  return [&malformed](const tci::elem_coors_t<TenT> &coors) -> std::ptrdiff_t {
    if (!coors.empty()) {
      malformed = true;
    }
    return 0;
  };
}

} // namespace detail

/// What a traversal of a zeroth-order tensor visits: record() runs in the
/// callback under "Callbacks the backend invokes" in CONTRIBUTING.md and
/// returns false for a non-empty coordinate, and check() runs after the call
/// returns.
struct visit_record_0d {
  std::size_t visits = 0;
  bool malformed_coordinate = false;

  template <typename Coors> bool record(const Coors &coors) {
    ++visits;
    if (!coors.empty()) {
      malformed_coordinate = true;
      return false;
    }
    return true;
  }

  // "Visits every element exactly once" over the one element, at {}.
  void check() const {
    TCICT_ASSERT(!malformed_coordinate);
    TCICT_ASSERT(visits == 1);
  }
};

// Each wrapper is guarded by the skip macro of the API it wraps, as in
// include/tcict/range_2d.h, and by TCICT_HAS_ELEM_SWEEP, which every test
// calling it also requires: the wrappers call tci::shape, which a backend
// defining TCICT_SKIP_SHAPE need not declare.

#if !defined(TCICT_SKIP_TO_RANGE) && defined(TCICT_HAS_ELEM_SWEEP)
/// `to_range` of a zeroth-order `a` into a one-element range at `first`.
/// Asserts that `a` has shape {} before the call, then that the call throws no
/// std::exception and that the backend passed only the empty coordinate.
template <typename TenT, typename RandomIt>
void to_range_0d(tci::context_handle_t<TenT> &ctx, const TenT &a,
                 RandomIt first) {
  TCICT_ASSERT(tci::shape(ctx, a).empty());
  bool malformed = false;
  TCICT_ASSERT_NOTHROW(
      tci::to_range(ctx, a, first, detail::map_0d<TenT>(malformed)));
  TCICT_ASSERT_MSG(!malformed, "to_range_0d: the backend passed a non-empty "
                               "coordinate for a zeroth-order tensor");
}
#endif

#if !defined(TCICT_SKIP_ASSIGN_FROM_RANGE) && defined(TCICT_HAS_ELEM_SWEEP)
/// `assign_from_range` at shape {} from the one-element range at `first`, into
/// `out`. Asserts that the call throws no std::exception, that the backend
/// passed only the empty coordinate, and that `out` has shape {}.
template <typename TenT, typename RandomIt>
void assign_from_range_0d(tci::context_handle_t<TenT> &ctx, RandomIt first,
                          TenT &out) {
  bool malformed = false;
  TCICT_ASSERT_NOTHROW(out = tci::template assign_from_range<TenT>(
                           ctx, tci::shape_t<TenT>{}, first,
                           detail::map_0d<TenT>(malformed)));
  TCICT_ASSERT_MSG(!malformed, "assign_from_range_0d: the backend passed a "
                               "non-empty coordinate for shape {}");
  TCICT_ASSERT(tci::shape(ctx, out).empty());
}
#endif

#if defined(TCICT_HAS_ELEM_SWEEP)

/// The element `TCICT_SET_ELEMS` writes for the real value `v`, as
/// `written<TenT>` gives it.
template <typename TenT> tci::elem_t<TenT> written_elem(double v) {
  const auto w = written<TenT>(v);
  return make_elem<TenT>(w.real(), w.imag());
}

#endif

#if defined(TCICT_HAS_ZEROTH_ORDER_SWEEP)

/// Asserts that `tensor` has order 0, shape {} and size 1, whatever its tensor
/// type, so that the results of real, imag and to_cplx can be checked with the
/// context their input came with. Call it through
/// TCICT_EXPECT_ZEROTH_ORDER_SHAPE, which passes the caller's location.
template <typename Ctx, typename Ten>
void expect_zeroth_order_shape_at(const char *file, int line, Ctx &ctx,
                                  const Ten &tensor) {
  const auto order =
      detail::nothrow_call(file, line, [&] { return tci::order(ctx, tensor); });
  if (order != 0) {
    std::ostringstream oss;
    oss << "order " << order << " != expected 0";
    detail::fail_at(file, line, oss.str());
  }
  const auto shape =
      detail::nothrow_call(file, line, [&] { return tci::shape(ctx, tensor); });
  if (!shape.empty()) {
    std::ostringstream oss;
    oss << "shape of " << shape.size() << " bonds != expected {}";
    detail::fail_at(file, line, oss.str());
  }
  const auto size =
      detail::nothrow_call(file, line, [&] { return tci::size(ctx, tensor); });
  if (size != 1) {
    std::ostringstream oss;
    oss << "size " << size << " != expected 1";
    detail::fail_at(file, line, oss.str());
  }
}

/// Asserts that `tensor` is zeroth-order, as expect_zeroth_order_shape_at
/// does, then, through expect_elems_at, that its sole element is `value`
/// within `tolerance(fix, cat, n)`. A real `value` stands for
/// `written<TenT>(value)`; a `std::complex<double>` one is compared as it is.
/// Call it through TCICT_EXPECT_ZEROTH_ORDER, which passes the caller's
/// location.
template <typename TenT, typename V>
void expect_zeroth_order_at(const char *file, int line,
                            tci_test_fixture<TenT> &fix, const TenT &tensor,
                            V value, tol_category cat = tol_category::exact,
                            std::size_t n = 1) {
  expect_zeroth_order_shape_at(file, line, fix.context(), tensor);
  expect_elems_at(
      file, line, fix, tensor, tci::shape_t<TenT>{},
      [value](const tci::elem_coors_t<TenT> &) { return value; }, cat, n);
}

#endif

} // namespace tcict

/// expect_zeroth_order_at with the caller's file and line:
/// TCICT_EXPECT_ZEROTH_ORDER(fix, tensor, value[, cat[, n]]), the line chosen as for
/// TCICT_EXPECT_ELEMS.
#define TCICT_EXPECT_ZEROTH_ORDER(...)                                         \
  ::tcict::expect_zeroth_order_at(__FILE__, __LINE__, __VA_ARGS__)

/// expect_zeroth_order_shape_at with the caller's file and line:
/// TCICT_EXPECT_ZEROTH_ORDER_SHAPE(ctx, tensor).
#define TCICT_EXPECT_ZEROTH_ORDER_SHAPE(...)                                   \
  ::tcict::expect_zeroth_order_shape_at(__FILE__, __LINE__, __VA_ARGS__)
