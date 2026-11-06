#pragma once

// Helpers for the tests of TCAPI V1's zeroth-order tensor: order 0, shape {},
// size 1, its sole element read and written at the empty coordinate {}.

#include <tcapict/assertion.h>
#include <tcapict/elem_helper.h>
#include <tcapict/elem_sweep.h>
#include <tcapict/fixture.h>
#include <tcapict/skip.h>

#include <cstddef>
#include <sstream>

namespace tcapict {

namespace detail {

// Sends the empty coordinate to index 0 and counts its invocations in `calls`.
// A non-empty coordinate sets `malformed` and also maps to index 0, which keeps
// the backend's access inside a one-element range; the wrappers below fail on
// `malformed` and on a count other than 1. See "Callbacks the backend invokes"
// in CONTRIBUTING.md.
template <typename TenT>
auto map_0d(bool &malformed, std::size_t &calls) {
  return [&malformed, &calls](
             const tcapi::elem_coors_t<TenT> &coors) -> std::ptrdiff_t {
    ++calls;
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

  // TCAPI V1's for_each_with_coors: "For a zeroth-order tensor, the callable is
  // invoked once with the empty coordinate sequence."
  void check() const {
    TCAPICT_ASSERT(!malformed_coordinate);
    TCAPICT_ASSERT(visits == 1);
  }
};

// Each wrapper is guarded by the skip macro of the API it wraps, as in
// include/tcapict/range_2d.h, and by TCAPICT_HAS_ELEM_SWEEP, which every test
// calling it also requires: the wrappers call tcapi::shape, which a backend
// defining TCAPICT_SKIP_SHAPE need not declare.

#if !defined(TCAPICT_SKIP_TO_RANGE) && defined(TCAPICT_HAS_ELEM_SWEEP)
/// `to_range` of a zeroth-order `a` into a one-element range at `first`.
/// Asserts that `a` has shape {} before the call, then that the call throws no
/// std::exception, that the backend passed only the empty coordinate, and that
/// it invoked the map once. TCAPI V1: "For a zeroth-order tensor, the output
/// range or list has length 1; `coors2idx` is invoked once with the empty
/// coordinate sequence and must return 0."
template <typename TenT, typename RandomIt>
void to_range_0d(tcapi::context_handle_t<TenT> &ctx, const TenT &a,
                 RandomIt first) {
  TCAPICT_ASSERT(tcapi::shape(ctx, a).empty());
  bool malformed = false;
  std::size_t calls = 0;
  TCAPICT_ASSERT_NOTHROW(
      tcapi::to_range(ctx, a, first, detail::map_0d<TenT>(malformed, calls)));
  TCAPICT_ASSERT_MSG(!malformed, "to_range_0d: the backend passed a non-empty "
                                 "coordinate for a zeroth-order tensor");
  TCAPICT_ASSERT_MSG(calls == 1, "to_range_0d: the backend did not invoke "
                                 "coors2idx exactly once");
}
#endif

#if !defined(TCAPICT_SKIP_ASSIGN_FROM_RANGE) && defined(TCAPICT_HAS_ELEM_SWEEP)
/// `assign_from_range` at shape {} from the one-element range at `first`, into
/// `out`. Asserts that the call throws no std::exception, that the backend
/// passed only the empty coordinate and invoked the map once, and that `out`
/// has shape {}. TCAPI V1: "The empty shape is valid; in that case, the
/// function returns a zeroth-order tensor and invokes the coordinate-to-index
/// map once with the empty coordinate sequence."
template <typename TenT, typename RandomIt>
void assign_from_range_0d(tcapi::context_handle_t<TenT> &ctx, RandomIt first,
                          TenT &out) {
  bool malformed = false;
  std::size_t calls = 0;
  TCAPICT_ASSERT_NOTHROW(out = tcapi::template assign_from_range<TenT>(
                           ctx, tcapi::shape_t<TenT>{}, first,
                           detail::map_0d<TenT>(malformed, calls)));
  TCAPICT_ASSERT_MSG(!malformed, "assign_from_range_0d: the backend passed a "
                                 "non-empty coordinate for shape {}");
  TCAPICT_ASSERT_MSG(calls == 1, "assign_from_range_0d: the backend did not "
                                 "invoke coors2idx exactly once");
  TCAPICT_ASSERT(tcapi::shape(ctx, out).empty());
}
#endif

#if defined(TCAPICT_HAS_ELEM_SWEEP)

/// Sets the sole element of the zeroth-order `tensor` to
/// `written_elem<TenT>(v)`, after asserting that its shape is {}. On an input
/// the caller just built at shape {}, a constructor that ignores the empty
/// shape is then reported here, as the input it built, rather than as a
/// failure of the operation the test is about. A
/// std::exception from shape or set_elem is reported at `file`:`line` as
/// detail::nothrow_call reports it; call it through TCAPICT_SET_ZEROTH_ORDER,
/// which passes the caller's location.
template <typename TenT>
void set_zeroth_order_at(const char *file, int line,
                         tcapi_test_fixture<TenT> &fix, TenT &tensor, double v) {
  auto &ctx = fix.context();
  const auto shape =
      detail::nothrow_call(file, line, [&] { return tcapi::shape(ctx, tensor); });
  if (!shape.empty()) {
    std::ostringstream oss;
    oss << "a tensor expected to be zeroth-order has " << shape.size() << " bonds";
    detail::fail_at(file, line, oss.str());
  }
  const auto elem = written_elem<TenT>(v);
  detail::nothrow_call(file, line,
                       [&] { tcapi::set_elem(ctx, tensor, {}, elem); });
}

#endif

#if defined(TCAPICT_HAS_ZEROTH_ORDER_SWEEP)

/// Asserts that `tensor` has order 0, shape {} and size 1, whatever its tensor
/// type, so that the results of real, imag and to_cplx can be checked with the
/// context their input came with. Call it through
/// TCAPICT_EXPECT_ZEROTH_ORDER_SHAPE, which passes the caller's location.
template <typename Ctx, typename Ten>
void expect_zeroth_order_shape_at(const char *file, int line, Ctx &ctx,
                                  const Ten &tensor) {
  const auto order =
      detail::nothrow_call(file, line, [&] { return tcapi::order(ctx, tensor); });
  if (order != 0) {
    std::ostringstream oss;
    oss << "order " << order << " != expected 0";
    detail::fail_at(file, line, oss.str());
  }
  const auto shape =
      detail::nothrow_call(file, line, [&] { return tcapi::shape(ctx, tensor); });
  if (!shape.empty()) {
    std::ostringstream oss;
    oss << "shape of " << shape.size() << " bonds != expected {}";
    detail::fail_at(file, line, oss.str());
  }
  const auto size =
      detail::nothrow_call(file, line, [&] { return tcapi::size(ctx, tensor); });
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
/// Call it through TCAPICT_EXPECT_ZEROTH_ORDER, which passes the caller's
/// location.
template <typename TenT, typename V>
void expect_zeroth_order_at(const char *file, int line,
                            tcapi_test_fixture<TenT> &fix, const TenT &tensor,
                            V value, tol_category cat = tol_category::exact,
                            std::size_t n = 1) {
  expect_zeroth_order_shape_at(file, line, fix.context(), tensor);
  expect_elems_at(
      file, line, fix, tensor, tcapi::shape_t<TenT>{},
      [value](const tcapi::elem_coors_t<TenT> &) { return value; }, cat, n);
}

#endif

} // namespace tcapict

/// expect_zeroth_order_at with the caller's file and line:
/// TCAPICT_EXPECT_ZEROTH_ORDER(fix, tensor, value[, cat[, n]]), the line chosen as for
/// TCAPICT_EXPECT_ELEMS.
#define TCAPICT_EXPECT_ZEROTH_ORDER(...)                                       \
  ::tcapict::expect_zeroth_order_at(__FILE__, __LINE__, __VA_ARGS__)

/// set_zeroth_order_at with the caller's file and line:
/// TCAPICT_SET_ZEROTH_ORDER(fix, tensor, v).
#define TCAPICT_SET_ZEROTH_ORDER(...)                                          \
  ::tcapict::set_zeroth_order_at(__FILE__, __LINE__, __VA_ARGS__)

/// expect_zeroth_order_shape_at with the caller's file and line:
/// TCAPICT_EXPECT_ZEROTH_ORDER_SHAPE(ctx, tensor).
#define TCAPICT_EXPECT_ZEROTH_ORDER_SHAPE(...)                                 \
  ::tcapict::expect_zeroth_order_shape_at(__FILE__, __LINE__, __VA_ARGS__)
