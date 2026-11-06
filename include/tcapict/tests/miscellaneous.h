#pragma once

#include <tcapict/assertion.h>
#include <tcapict/elem_helper.h>
#include <tcapict/elem_sweep.h>
#include <tcapict/fixture.h>
#include <tcapict/range_2d.h>
#include <tcapict/skip.h>
#include <tcapict/zeroth_order.h>

#include <complex>
#include <cstddef>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace tcapict {
namespace detail {

// Type helpers for tcapi::supported_ten_types_t. Each one takes any type and
// treats one that is not a std::tuple as holding no tensor types, so the tests
// using them compile against a catalog of the wrong kind, and
// test_supported_ten_types reports it through its assertion on
// is_std_tuple_v.

template <typename T> struct type_tag {
  using type = T;
};

template <typename T> struct is_std_tuple : std::false_type {};
template <typename... Ts> struct is_std_tuple<std::tuple<Ts...>> : std::true_type {};
template <typename T> inline constexpr bool is_std_tuple_v = is_std_tuple<T>::value;

template <typename T, typename Tuple> struct tuple_contains : std::false_type {};
template <typename T, typename... Ts>
struct tuple_contains<T, std::tuple<Ts...>>
    : std::bool_constant<(std::is_same_v<T, Ts> || ...)> {};
template <typename T, typename Tuple>
inline constexpr bool tuple_contains_v = tuple_contains<T, Tuple>::value;

template <typename Tuple> struct for_each_tuple_type_impl {
  template <typename F> static void run(F &&) {}
};
template <typename... Ts> struct for_each_tuple_type_impl<std::tuple<Ts...>> {
  template <typename F> static void run(F &&f) { (f(type_tag<Ts>{}), ...); }
};

// Calls `f(type_tag<U>{})` once for each element type `U` of `Tuple`.
template <typename Tuple, typename F> void for_each_tuple_type(F &&f) {
  for_each_tuple_type_impl<Tuple>::run(std::forward<F>(f));
}

} // namespace detail

namespace tests {

// --- close (eq) : identical tensors ---

template <typename TenT>
void test_close_identical(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_CLOSE
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  auto tensor1 = tcapi::eye<TenT>(ctx, 2);
  auto tensor2 = tcapi::eye<TenT>(ctx, 2);

  bool are_equal = tcapi::close(ctx, tensor1, tensor2, eps);
  TCAPICT_ASSERT(are_equal == true);
#else
  (void)fix;
#endif
}

// --- close (eq) : different tensors ---

template <typename TenT>
void test_close_different(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_CLOSE
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  auto tensor1 = tcapi::eye<TenT>(ctx, 2);
  auto tensor2 = tcapi::zeros<TenT>(ctx, {2, 2});

  bool are_equal = tcapi::close(ctx, tensor1, tensor2, eps);
  TCAPICT_ASSERT(are_equal == false);
#else
  (void)fix;
#endif
}

// --- close: zeroth-order ---

// TCAPI V1: close succeeds iff the shapes match and the largest elementwise
// |a_i - b_i| is at most epsilon. Against an equal tensor it is true; against
// another value, or for a complex element type a difference in the imaginary
// part alone, it is false; and against a shape-{1} tensor holding the same
// value it is false, since {} and {1} differ.
template <typename TenT>
void test_close_zeroth_order(tcapi_test_fixture<TenT> &fix) {
#if !defined(TCAPICT_SKIP_CLOSE) && defined(TCAPICT_HAS_ELEM_SWEEP)
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  auto value = [](const tcapi::elem_coors_t<TenT> &) { return 1.5; };
  auto a = tcapi::zeros<TenT>(ctx, {});
  TCAPICT_SET_ZEROTH_ORDER(fix, a, 1.5);

  auto same = tcapi::zeros<TenT>(ctx, {});
  TCAPICT_SET_ZEROTH_ORDER(fix, same, 1.5);
  TCAPICT_ASSERT(tcapi::close(ctx, a, same, eps));

  auto other = tcapi::zeros<TenT>(ctx, {});
  TCAPICT_SET_ZEROTH_ORDER(fix, other, 1.75);
  TCAPICT_ASSERT(!tcapi::close(ctx, a, other, eps));

  if constexpr (is_complex_v<TenT>) {
    const auto w = written<TenT>(1.5);
    auto imag_off = tcapi::zeros<TenT>(ctx, {});
    TCAPICT_ASSERT(tcapi::shape(ctx, imag_off).empty());
    tcapi::set_elem(ctx, imag_off, {}, make_elem<TenT>(w.real(), w.imag() + 0.25));
    TCAPICT_ASSERT(!tcapi::close(ctx, a, imag_off, eps));
  }

  auto one = tcapi::zeros<TenT>(ctx, {1});
  TCAPICT_SET_ELEMS(fix, one, value);
  TCAPICT_ASSERT(!tcapi::close(ctx, a, one, eps));
#else
  (void)fix;
#endif
}

// --- close: the maximum of the differences, not their norm ---

// TCAPI V1: close succeeds iff the shapes match and the largest elementwise
// |a_i - b_i| is at most epsilon. Two differences of 0.375 against epsilon
// 0.5 have maximum 0.375, so close is true, while their Frobenius norm,
// sqrt(2) * 0.375 ~ 0.53, exceeds 0.5. One difference of 0.625 makes it
// false. Every value is a binary fraction, so the differences are exact.
template <typename TenT>
void test_close_maximum_norm(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_CLOSE
  auto &ctx = fix.context();
  const tcapi::real_t<TenT> epsilon = 0.5;
  auto zero = tcapi::zeros<TenT>(ctx, {2});

  auto within = tcapi::zeros<TenT>(ctx, {2});
  tcapi::set_elem(ctx, within, {0}, make_elem<TenT>(0.375));
  tcapi::set_elem(ctx, within, {1}, make_elem<TenT>(0.375));
  TCAPICT_ASSERT(tcapi::close(ctx, zero, within, epsilon));

  auto beyond = tcapi::zeros<TenT>(ctx, {2});
  tcapi::set_elem(ctx, beyond, {0}, make_elem<TenT>(0.375));
  tcapi::set_elem(ctx, beyond, {1}, make_elem<TenT>(0.625));
  TCAPICT_ASSERT(!tcapi::close(ctx, zero, beyond, epsilon));
#else
  (void)fix;
#endif
}

// --- close: a difference equal to epsilon ---

// TCAPI V1's close requires the largest difference to be at most epsilon,
// which admits a difference equal to epsilon. The difference 0.5 and the
// tolerances 0.5 and 0.4375 are binary fractions, so the comparison is exact.
template <typename TenT>
void test_close_boundary_inclusive(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_CLOSE
  auto &ctx = fix.context();
  auto zero = tcapi::zeros<TenT>(ctx, {1});
  auto half = tcapi::zeros<TenT>(ctx, {1});
  tcapi::set_elem(ctx, half, {0}, make_elem<TenT>(0.5));
  TCAPICT_ASSERT(tcapi::close(ctx, zero, half, tcapi::real_t<TenT>(0.5)));
  TCAPICT_ASSERT(!tcapi::close(ctx, zero, half, tcapi::real_t<TenT>(0.4375)));
#else
  (void)fix;
#endif
}

// --- close: the modulus of a complex difference ---

// TCAPI V1 measures a complex difference by "absolute value, or modulus for
// complex values". 0.375 + 0.5i has modulus 0.625: false at epsilon 0.6, which
// the larger of its parts, 0.5, would pass, and true at 0.7, which the sum of
// its parts, 0.875, would fail.
template <typename TenT>
void test_close_complex_modulus(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_CLOSE
  if constexpr (is_complex_v<TenT>) {
    auto &ctx = fix.context();
    auto zero = tcapi::zeros<TenT>(ctx, {1});
    auto off = tcapi::zeros<TenT>(ctx, {1});
    tcapi::set_elem(ctx, off, {0}, make_elem<TenT>(0.375, 0.5));
    TCAPICT_ASSERT(!tcapi::close(ctx, zero, off, tcapi::real_t<TenT>(0.6)));
    TCAPICT_ASSERT(tcapi::close(ctx, zero, off, tcapi::real_t<TenT>(0.7)));
  } else {
    (void)fix;
  }
#else
  (void)fix;
#endif
}

// --- close: shapes of one size ---

// TCAPI V1 requires the shapes to match; {2, 3} and {3, 2} hold the same number
// of elements, all zero, and still differ.
template <typename TenT>
void test_close_shape_mismatch_same_size(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_CLOSE
  auto &ctx = fix.context();
  auto wide = tcapi::zeros<TenT>(ctx, {2, 3});
  auto tall = tcapi::zeros<TenT>(ctx, {3, 2});
  TCAPICT_ASSERT(!tcapi::close(ctx, wide, tall, fix.epsilon()));
#else
  (void)fix;
#endif
}

// --- to_range ---

template <typename TenT> void test_to_range(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_TO_RANGE
  auto &ctx = fix.context();
  auto eps = fix.epsilon();

  TenT a = tcapi::template zeros<TenT>(ctx, {2, 3});
  bool malformed_traversal = false;
  tcapi::for_each_with_coors(
      ctx, a,
      [&](tcapi::elem_t<TenT> &elem, const tcapi::elem_coors_t<TenT> &coors) {
        std::ptrdiff_t row = 0;
        std::ptrdiff_t col = 0;
        if (!coors_2d(coors, 2, 3, row, col)) {
          malformed_traversal = true;
          return;
        }
        elem = static_cast<tcapi::elem_t<TenT>>(row * 3 + col);
      });
  TCAPICT_ASSERT(!malformed_traversal);

  // Filled with a sentinel rather than zeros: element {0, 0} is 0, so a
  // zero-filled slot that to_range never wrote would pass for it.
  std::vector<tcapi::elem_t<TenT>> container(6, make_elem<TenT>(-999.0));

  to_range_2d(ctx, a, {2, 3}, container.begin(), layout_2d::row_major);

  for (int i = 0; i < 6; ++i) {
    TCAPICT_ASSERT_CLOSE(real_part<TenT>(container[i]), static_cast<double>(i),
                         eps);
    if constexpr (is_complex_v<TenT>) {
      TCAPICT_ASSERT_CLOSE(imag_part<TenT>(container[i]), 0.0, eps);
    }
  }
#else
  (void)fix;
#endif
}

// --- to_range: zeroth-order ---

// The sole element goes to `*(first + coors2idx({}))`, here the one slot of a
// range that starts out holding a sentinel.
template <typename TenT>
void test_to_range_zeroth_order(tcapi_test_fixture<TenT> &fix) {
#if !defined(TCAPICT_SKIP_TO_RANGE) && defined(TCAPICT_HAS_ELEM_SWEEP)
  auto &ctx = fix.context();
  auto tensor = tcapi::zeros<TenT>(ctx, {});
  TCAPICT_SET_ZEROTH_ORDER(fix, tensor, 5.5);

  std::vector<tcapi::elem_t<TenT>> out(1, make_elem<TenT>(-99.0));
  to_range_0d(ctx, tensor, out.begin());
  const auto w = written<TenT>(5.5);
  TCAPICT_ASSERT(real_part<TenT>(out[0]) == w.real());
  TCAPICT_ASSERT(imag_part<TenT>(out[0]) == w.imag());
#else
  (void)fix;
#endif
}

// --- show (does not throw) ---

template <typename TenT> void test_show(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_SHOW
  auto &ctx = fix.context();
  auto a = tcapi::eye<TenT>(ctx, 2);
  TCAPICT_ASSERT_NOTHROW(tcapi::show(ctx, a));
#else
  (void)fix;
#endif
}

// --- show: zeroth-order ---

template <typename TenT>
void test_show_zeroth_order(tcapi_test_fixture<TenT> &fix) {
#if !defined(TCAPICT_SKIP_SHOW) && defined(TCAPICT_HAS_ELEM_SWEEP)
  auto &ctx = fix.context();
  // The output format is implementation-defined; the call returning is what
  // is asserted.
  auto tensor = tcapi::zeros<TenT>(ctx, {});
  TCAPICT_SET_ZEROTH_ORDER(fix, tensor, 2.5);
  TCAPICT_ASSERT_NOTHROW(tcapi::show(ctx, tensor));
#else
  (void)fix;
#endif
}

// --- convert (same context) ---

template <typename TenT>
void test_convert_same_context(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_CONVERT
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  auto a = tcapi::eye<TenT>(ctx, 3);

  TenT b;
  TCAPICT_ASSERT_NOTHROW(tcapi::convert(ctx, a, ctx, b));
  TCAPICT_ASSERT(tcapi::close(ctx, a, b, eps));
  TCAPICT_ASSERT(tcapi::shape(ctx, a) == tcapi::shape(ctx, b));

  // Verify independence
  TenT original_b = tcapi::copy(ctx, b);
  tcapi::set_elem(ctx, a, {0, 0}, make_elem<TenT>(999.0));
  TCAPICT_ASSERT(tcapi::close(ctx, b, original_b, eps));
#else
  (void)fix;
#endif
}

// --- convert (different contexts) ---

template <typename TenT>
void test_convert_different_context(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_CONVERT
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  auto a = tcapi::eye<TenT>(ctx, 3);

  tcapi::context_handle_t<TenT> ctx2;
  tcapi::create_context(ctx2);

  TenT b;
  TCAPICT_ASSERT_NOTHROW(tcapi::convert(ctx, a, ctx2, b));
  TCAPICT_ASSERT(tcapi::close(ctx, a, b, eps));
  TCAPICT_ASSERT(tcapi::shape(ctx, a) == tcapi::shape(ctx, b));

  tcapi::destroy_context(ctx2);
#else
  (void)fix;
#endif
}

// --- convert (data integrity) ---

template <typename TenT>
void test_convert_data_integrity(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_CONVERT
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  auto a = tcapi::zeros<TenT>(ctx, {2, 3});
  tcapi::set_elem(ctx, a, {0, 0}, make_elem<TenT>(1.23, 4.56));
  tcapi::set_elem(ctx, a, {1, 2}, make_elem<TenT>(-7.89, 0.12));

  TenT b;
  tcapi::context_handle_t<TenT> ctx2;
  tcapi::create_context(ctx2);
  tcapi::convert(ctx, a, ctx2, b);

  auto val1 = tcapi::get_elem(ctx, b, {0, 0});
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(val1), 1.23, eps);

  auto val2 = tcapi::get_elem(ctx, b, {1, 2});
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(val2), -7.89, eps);

  if constexpr (is_complex_v<TenT>) {
    TCAPICT_ASSERT_CLOSE(imag_part<TenT>(val1), 4.56, eps);
    TCAPICT_ASSERT_CLOSE(imag_part<TenT>(val2), 0.12, eps);
  }

  tcapi::destroy_context(ctx2);
#else
  (void)fix;
#endif
}

// --- convert: zeroth-order ---

// TCAPI V1: with Ten1T == Ten2T, convert is "equivalent to deep copy".
template <typename TenT>
void test_convert_zeroth_order(tcapi_test_fixture<TenT> &fix) {
#if !defined(TCAPICT_SKIP_CONVERT) && defined(TCAPICT_HAS_ZEROTH_ORDER_SWEEP)
  auto &ctx = fix.context();
  auto a = tcapi::zeros<TenT>(ctx, {});
  TCAPICT_SET_ZEROTH_ORDER(fix, a, 2.5);

  TenT b;
  TCAPICT_ASSERT_NOTHROW(tcapi::convert(ctx, a, ctx, b));
  TCAPICT_EXPECT_ZEROTH_ORDER(fix, b, 2.5);

  TCAPICT_SET_ZEROTH_ORDER(fix, a, -1.0);
  TCAPICT_EXPECT_ZEROTH_ORDER(fix, b, 2.5);
#else
  (void)fix;
#endif
}

// --- convert: real to complex element type ---

// TCAPI V1: convert converts `t1` "into a destination tensor, possibly
// changing the element type, the underlying TCF/backend, or both", writing
// the result into `t2`, the tensor on the destination context, so the test
// reads it through that context. Converting a real tensor to
// tcapi::cplx_ten_t<RealTenT> keeps each value as its real part with a zero
// imaginary part, the one reading that preserves the values; every value here
// is a small integer, so the copy is exact. The input is read back as
// "Out-of-place inputs" in CONTRIBUTING.md describes.
template <typename RealTenT>
void test_convert_real_to_complex(tcapi_test_fixture<RealTenT> &fix) {
#if !defined(TCAPICT_SKIP_CONVERT) && defined(TCAPICT_HAS_ELEM_SWEEP)
  static_assert(!is_complex_v<RealTenT>,
                "test_convert_real_to_complex takes a real tensor type");
  using CplxTenT = tcapi::cplx_ten_t<RealTenT>;
  auto &ctx = fix.context();
  const tcapi::shape_t<RealTenT> shape = {2, 3};
  auto at = ramp_over<RealTenT>(shape, -2.0);
  auto a = tcapi::zeros<RealTenT>(ctx, shape);
  TCAPICT_SET_ELEMS(fix, a, at);

  // The destination context comes from the backend's fixture for CplxTenT,
  // which releases it on every exit; b, declared after it, is released first.
  tcapi_test_fixture<CplxTenT> fix2;
  CplxTenT b;
  TCAPICT_ASSERT_NOTHROW(tcapi::convert(ctx, a, fix2.context(), b));
  const tcapi::shape_t<CplxTenT> b_shape(shape.begin(), shape.end());
  TCAPICT_EXPECT_ELEMS(
      fix2, b, b_shape,
      [&](const tcapi::elem_coors_t<CplxTenT> &c) {
        const tcapi::elem_coors_t<RealTenT> coors(c.begin(), c.end());
        return std::complex<double>(at(coors), 0.0);
      },
      tol_category::exact);

  TCAPICT_EXPECT_ELEMS(fix, a, shape, at, tol_category::exact);
#else
  (void)fix;
#endif
}

// --- convert: precision change ---

// TCAPI V1: convert converts `t1` "into a destination tensor, possibly
// changing the element type, the underlying TCF/backend, or both. Conversion
// also supports precision changes within one TCF." The first sentence says
// what a conversion may change, not which pairs a backend supports; the second
// is what V1 says convert supports, so the destinations are the types of
// tcapi::supported_ten_types_t in TenT's TCF that hold TenT's kind of element,
// real or complex, at another precision. V1 ties a TCF to a context handle
// type: context_handle_t is the "Handle to the backend context maintained by
// the underlying TCF that provides `ten_t`", and of create_context, "the
// concrete type `ContextHandleT` selects the target TCF/backend". So a type
// whose context_handle_t is TenT's is in TenT's TCF; one with another handle
// type may or may not be, and is not converted to. Each result is read through
// the destination type's own fixture context. Every real and imaginary part
// written is a small integer or half of one, exact in single precision and
// wider, so the copy is exact. A catalog with no such type leaves nothing to
// convert. The input is read back as "Out-of-place inputs" in CONTRIBUTING.md
// describes.
template <typename TenT>
void test_convert_precision(tcapi_test_fixture<TenT> &fix) {
#if !defined(TCAPICT_SKIP_CONVERT) && !defined(TCAPICT_SKIP_SUPPORTED_TEN_TYPES) &&           \
    defined(TCAPICT_HAS_ELEM_SWEEP)
  auto &ctx = fix.context();
  const tcapi::shape_t<TenT> shape = {2, 3};
  auto at = ramp_over<TenT>(shape, -2.0);
  auto a = tcapi::zeros<TenT>(ctx, shape);
  TCAPICT_SET_ELEMS(fix, a, at);

  detail::for_each_tuple_type<tcapi::supported_ten_types_t>([&](auto tag) {
    using DestTenT = typename decltype(tag)::type;
    if constexpr (std::is_same_v<tcapi::context_handle_t<DestTenT>,
                                 tcapi::context_handle_t<TenT>> &&
                  is_complex_v<DestTenT> == is_complex_v<TenT> &&
                  !std::is_same_v<tcapi::real_t<DestTenT>, tcapi::real_t<TenT>>) {
      // b, declared after fix2, is released first.
      tcapi_test_fixture<DestTenT> fix2;
      DestTenT b;
      TCAPICT_ASSERT_NOTHROW(tcapi::convert(ctx, a, fix2.context(), b));
      const tcapi::shape_t<DestTenT> b_shape(shape.begin(), shape.end());
      // DestTenT holds TenT's kind of element, so a real value stands for the
      // same element in both.
      TCAPICT_EXPECT_ELEMS(
          fix2, b, b_shape,
          [&](const tcapi::elem_coors_t<DestTenT> &c) {
            const tcapi::elem_coors_t<TenT> coors(c.begin(), c.end());
            return at(coors);
          },
          tol_category::exact);
    }
  });

  TCAPICT_EXPECT_ELEMS(fix, a, shape, at, tol_category::exact);
#else
  (void)fix;
#endif
}

// --- supported_ten_types_t ---

// TCAPI V1: "Each C++ implementation provides a compile-time catalog of its
// tensor types through `tcapi::supported_ten_types_t` in `tcapi/tcapi.h`", a
// std::tuple of them. "The catalog order is implementation-defined, and
// context compatibility is backend-dependent. Standard tuple traits and
// `tcapi::elem_t` expose its size, types, and element types". The test asserts
// that the catalog is a std::tuple, and names tcapi::elem_t of each catalog
// type, which fails to compile for a type whose traits do not expose one. It
// also asserts a reading of "its tensor types": that the catalog holds TenT, a
// type the backend registers with the suite, and tcapi::real_ten_t<TenT> and
// tcapi::cplx_ten_t<TenT>, which V1 defines as tensor types and the backend's
// functions return. It asserts neither the order nor anything about contexts.
template <typename TenT>
void test_supported_ten_types(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_SUPPORTED_TEN_TYPES
  using Catalog = tcapi::supported_ten_types_t;
  TCAPICT_ASSERT((detail::is_std_tuple_v<Catalog>));
  TCAPICT_ASSERT((detail::tuple_contains_v<TenT, Catalog>));
  TCAPICT_ASSERT((detail::tuple_contains_v<tcapi::real_ten_t<TenT>, Catalog>));
  TCAPICT_ASSERT((detail::tuple_contains_v<tcapi::cplx_ten_t<TenT>, Catalog>));
  detail::for_each_tuple_type<Catalog>([](auto tag) {
    using ElemT = tcapi::elem_t<typename decltype(tag)::type>;
    [[maybe_unused]] ElemT *elem_t_named = nullptr;
  });
#endif
  (void)fix;
}

// --- version ---

template <typename TenT> void test_version(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_VERSION
  std::string ver = tcapi::template version<TenT>();
  TCAPICT_ASSERT(!ver.empty());
  // Version string should contain digits and a dot
  bool has_version_pattern =
      (ver.find_first_of("0123456789") != std::string::npos) &&
      (ver.find('.') != std::string::npos);
  TCAPICT_ASSERT(has_version_pattern);
#else
  (void)fix;
#endif
}

} // namespace tests
} // namespace tcapict

// Bulk registration helper: invokes X(..., "category", test_fn) once per test.
// See include/tcapict/adapters/doctest.h for usage.
#define TCAPICT_FOREACH_MISCELLANEOUS_TEST_ALL_TYPES(X, ...) \
  X(__VA_ARGS__, "miscellaneous", test_close_identical) \
  X(__VA_ARGS__, "miscellaneous", test_close_different) \
  X(__VA_ARGS__, "miscellaneous", test_close_zeroth_order) \
  X(__VA_ARGS__, "miscellaneous", test_close_maximum_norm) \
  X(__VA_ARGS__, "miscellaneous", test_close_boundary_inclusive) \
  X(__VA_ARGS__, "miscellaneous", test_close_shape_mismatch_same_size) \
  X(__VA_ARGS__, "miscellaneous", test_to_range) \
  X(__VA_ARGS__, "miscellaneous", test_to_range_zeroth_order) \
  X(__VA_ARGS__, "miscellaneous", test_show) \
  X(__VA_ARGS__, "miscellaneous", test_show_zeroth_order) \
  X(__VA_ARGS__, "miscellaneous", test_convert_same_context) \
  X(__VA_ARGS__, "miscellaneous", test_convert_different_context) \
  X(__VA_ARGS__, "miscellaneous", test_convert_data_integrity) \
  X(__VA_ARGS__, "miscellaneous", test_convert_zeroth_order) \
  X(__VA_ARGS__, "miscellaneous", test_convert_precision) \
  X(__VA_ARGS__, "miscellaneous", test_supported_ten_types) \
  X(__VA_ARGS__, "miscellaneous", test_version)

// REAL_ONLY: the test's input must be real, as its template parameter's name
//   says.
#define TCAPICT_FOREACH_MISCELLANEOUS_TEST_REAL_ONLY(X, ...) \
  X(__VA_ARGS__, "miscellaneous", test_convert_real_to_complex)

// CPLX_ONLY: body is wrapped in `if constexpr (is_complex_v<TenT>)`; running
//   for real TenT would be a no-op, so skip registration.
#define TCAPICT_FOREACH_MISCELLANEOUS_TEST_CPLX_ONLY(X, ...) \
  X(__VA_ARGS__, "miscellaneous", test_close_complex_modulus)
