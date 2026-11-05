#pragma once

#include <tcict/assertion.h>
#include <tcict/elem_helper.h>
#include <tcict/elem_sweep.h>
#include <tcict/fixture.h>
#include <tcict/range_2d.h>
#include <tcict/skip.h>
#include <tcict/zeroth_order.h>

#include <complex>
#include <cstddef>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace tcict {
namespace detail {

// Type helpers for tci::supported_ten_types_t. Each one takes any type and
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
void test_close_identical(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_CLOSE
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  auto tensor1 = tci::eye<TenT>(ctx, 2);
  auto tensor2 = tci::eye<TenT>(ctx, 2);

  bool are_equal = tci::close(ctx, tensor1, tensor2, eps);
  TCICT_ASSERT(are_equal == true);
#else
  (void)fix;
#endif
}

// --- close (eq) : different tensors ---

template <typename TenT>
void test_close_different(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_CLOSE
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  auto tensor1 = tci::eye<TenT>(ctx, 2);
  auto tensor2 = tci::zeros<TenT>(ctx, {2, 2});

  bool are_equal = tci::close(ctx, tensor1, tensor2, eps);
  TCICT_ASSERT(are_equal == false);
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
void test_close_zeroth_order(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_CLOSE) && defined(TCICT_HAS_ELEM_SWEEP)
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  auto value = [](const tci::elem_coors_t<TenT> &) { return 1.5; };
  auto a = tci::zeros<TenT>(ctx, {});
  TCICT_SET_ZEROTH_ORDER(fix, a, 1.5);

  auto same = tci::zeros<TenT>(ctx, {});
  TCICT_SET_ZEROTH_ORDER(fix, same, 1.5);
  TCICT_ASSERT(tci::close(ctx, a, same, eps));

  auto other = tci::zeros<TenT>(ctx, {});
  TCICT_SET_ZEROTH_ORDER(fix, other, 1.75);
  TCICT_ASSERT(!tci::close(ctx, a, other, eps));

  if constexpr (is_complex_v<TenT>) {
    const auto w = written<TenT>(1.5);
    auto imag_off = tci::zeros<TenT>(ctx, {});
    TCICT_ASSERT(tci::shape(ctx, imag_off).empty());
    tci::set_elem(ctx, imag_off, {}, make_elem<TenT>(w.real(), w.imag() + 0.25));
    TCICT_ASSERT(!tci::close(ctx, a, imag_off, eps));
  }

  auto one = tci::zeros<TenT>(ctx, {1});
  TCICT_SET_ELEMS(fix, one, value);
  TCICT_ASSERT(!tci::close(ctx, a, one, eps));
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
void test_close_maximum_norm(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_CLOSE
  auto &ctx = fix.context();
  const tci::real_t<TenT> epsilon = 0.5;
  auto zero = tci::zeros<TenT>(ctx, {2});

  auto within = tci::zeros<TenT>(ctx, {2});
  tci::set_elem(ctx, within, {0}, make_elem<TenT>(0.375));
  tci::set_elem(ctx, within, {1}, make_elem<TenT>(0.375));
  TCICT_ASSERT(tci::close(ctx, zero, within, epsilon));

  auto beyond = tci::zeros<TenT>(ctx, {2});
  tci::set_elem(ctx, beyond, {0}, make_elem<TenT>(0.375));
  tci::set_elem(ctx, beyond, {1}, make_elem<TenT>(0.625));
  TCICT_ASSERT(!tci::close(ctx, zero, beyond, epsilon));
#else
  (void)fix;
#endif
}

// --- close: a difference equal to epsilon ---

// TCAPI V1's close requires the largest difference to be at most epsilon,
// which admits a difference equal to epsilon. The difference 0.5 and the
// tolerances 0.5 and 0.4375 are binary fractions, so the comparison is exact.
template <typename TenT>
void test_close_boundary_inclusive(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_CLOSE
  auto &ctx = fix.context();
  auto zero = tci::zeros<TenT>(ctx, {1});
  auto half = tci::zeros<TenT>(ctx, {1});
  tci::set_elem(ctx, half, {0}, make_elem<TenT>(0.5));
  TCICT_ASSERT(tci::close(ctx, zero, half, tci::real_t<TenT>(0.5)));
  TCICT_ASSERT(!tci::close(ctx, zero, half, tci::real_t<TenT>(0.4375)));
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
void test_close_complex_modulus(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_CLOSE
  if constexpr (is_complex_v<TenT>) {
    auto &ctx = fix.context();
    auto zero = tci::zeros<TenT>(ctx, {1});
    auto off = tci::zeros<TenT>(ctx, {1});
    tci::set_elem(ctx, off, {0}, make_elem<TenT>(0.375, 0.5));
    TCICT_ASSERT(!tci::close(ctx, zero, off, tci::real_t<TenT>(0.6)));
    TCICT_ASSERT(tci::close(ctx, zero, off, tci::real_t<TenT>(0.7)));
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
void test_close_shape_mismatch_same_size(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_CLOSE
  auto &ctx = fix.context();
  auto wide = tci::zeros<TenT>(ctx, {2, 3});
  auto tall = tci::zeros<TenT>(ctx, {3, 2});
  TCICT_ASSERT(!tci::close(ctx, wide, tall, fix.epsilon()));
#else
  (void)fix;
#endif
}

// --- to_range ---

template <typename TenT> void test_to_range(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_TO_RANGE
  auto &ctx = fix.context();
  auto eps = fix.epsilon();

  TenT a = tci::template zeros<TenT>(ctx, {2, 3});
  bool malformed_traversal = false;
  tci::for_each_with_coors(
      ctx, a,
      [&](tci::elem_t<TenT> &elem, const tci::elem_coors_t<TenT> &coors) {
        std::ptrdiff_t row = 0;
        std::ptrdiff_t col = 0;
        if (!coors_2d(coors, 2, 3, row, col)) {
          malformed_traversal = true;
          return;
        }
        elem = static_cast<tci::elem_t<TenT>>(row * 3 + col);
      });
  TCICT_ASSERT(!malformed_traversal);

  // Filled with a sentinel rather than zeros: element {0, 0} is 0, so a
  // zero-filled slot that to_range never wrote would pass for it.
  std::vector<tci::elem_t<TenT>> container(6, make_elem<TenT>(-999.0));

  to_range_2d(ctx, a, {2, 3}, container.begin(), layout_2d::row_major);

  for (int i = 0; i < 6; ++i) {
    TCICT_ASSERT_CLOSE(real_part<TenT>(container[i]), static_cast<double>(i),
                       eps);
    if constexpr (is_complex_v<TenT>) {
      TCICT_ASSERT_CLOSE(imag_part<TenT>(container[i]), 0.0, eps);
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
void test_to_range_zeroth_order(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_TO_RANGE) && defined(TCICT_HAS_ELEM_SWEEP)
  auto &ctx = fix.context();
  auto tensor = tci::zeros<TenT>(ctx, {});
  TCICT_SET_ZEROTH_ORDER(fix, tensor, 5.5);

  std::vector<tci::elem_t<TenT>> out(1, make_elem<TenT>(-99.0));
  to_range_0d(ctx, tensor, out.begin());
  const auto w = written<TenT>(5.5);
  TCICT_ASSERT(real_part<TenT>(out[0]) == w.real());
  TCICT_ASSERT(imag_part<TenT>(out[0]) == w.imag());
#else
  (void)fix;
#endif
}

// --- show (does not throw) ---

template <typename TenT> void test_show(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_SHOW
  auto &ctx = fix.context();
  auto a = tci::eye<TenT>(ctx, 2);
  TCICT_ASSERT_NOTHROW(tci::show(ctx, a));
#else
  (void)fix;
#endif
}

// --- show: zeroth-order ---

template <typename TenT>
void test_show_zeroth_order(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_SHOW) && defined(TCICT_HAS_ELEM_SWEEP)
  auto &ctx = fix.context();
  // The output format is implementation-defined; the call returning is what
  // is asserted.
  auto tensor = tci::zeros<TenT>(ctx, {});
  TCICT_SET_ZEROTH_ORDER(fix, tensor, 2.5);
  TCICT_ASSERT_NOTHROW(tci::show(ctx, tensor));
#else
  (void)fix;
#endif
}

// --- convert (same context) ---

template <typename TenT>
void test_convert_same_context(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_CONVERT
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  auto a = tci::eye<TenT>(ctx, 3);

  TenT b;
  TCICT_ASSERT_NOTHROW(tci::convert(ctx, a, ctx, b));
  TCICT_ASSERT(tci::close(ctx, a, b, eps));
  TCICT_ASSERT(tci::shape(ctx, a) == tci::shape(ctx, b));

  // Verify independence
  TenT original_b = tci::copy(ctx, b);
  tci::set_elem(ctx, a, {0, 0}, make_elem<TenT>(999.0));
  TCICT_ASSERT(tci::close(ctx, b, original_b, eps));
#else
  (void)fix;
#endif
}

// --- convert (different contexts) ---

template <typename TenT>
void test_convert_different_context(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_CONVERT
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  auto a = tci::eye<TenT>(ctx, 3);

  tci::context_handle_t<TenT> ctx2;
  tci::create_context(ctx2);

  TenT b;
  TCICT_ASSERT_NOTHROW(tci::convert(ctx, a, ctx2, b));
  TCICT_ASSERT(tci::close(ctx, a, b, eps));
  TCICT_ASSERT(tci::shape(ctx, a) == tci::shape(ctx, b));

  tci::destroy_context(ctx2);
#else
  (void)fix;
#endif
}

// --- convert (data integrity) ---

template <typename TenT>
void test_convert_data_integrity(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_CONVERT
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  auto a = tci::zeros<TenT>(ctx, {2, 3});
  tci::set_elem(ctx, a, {0, 0}, make_elem<TenT>(1.23, 4.56));
  tci::set_elem(ctx, a, {1, 2}, make_elem<TenT>(-7.89, 0.12));

  TenT b;
  tci::context_handle_t<TenT> ctx2;
  tci::create_context(ctx2);
  tci::convert(ctx, a, ctx2, b);

  auto val1 = tci::get_elem(ctx, b, {0, 0});
  TCICT_ASSERT_CLOSE(real_part<TenT>(val1), 1.23, eps);

  auto val2 = tci::get_elem(ctx, b, {1, 2});
  TCICT_ASSERT_CLOSE(real_part<TenT>(val2), -7.89, eps);

  if constexpr (is_complex_v<TenT>) {
    TCICT_ASSERT_CLOSE(imag_part<TenT>(val1), 4.56, eps);
    TCICT_ASSERT_CLOSE(imag_part<TenT>(val2), 0.12, eps);
  }

  tci::destroy_context(ctx2);
#else
  (void)fix;
#endif
}

// --- convert: zeroth-order ---

// TCAPI V1: with Ten1T == Ten2T, convert is "equivalent to deep copy".
template <typename TenT>
void test_convert_zeroth_order(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_CONVERT) && defined(TCICT_HAS_ZEROTH_ORDER_SWEEP)
  auto &ctx = fix.context();
  auto a = tci::zeros<TenT>(ctx, {});
  TCICT_SET_ZEROTH_ORDER(fix, a, 2.5);

  TenT b;
  TCICT_ASSERT_NOTHROW(tci::convert(ctx, a, ctx, b));
  TCICT_EXPECT_ZEROTH_ORDER(fix, b, 2.5);

  TCICT_SET_ZEROTH_ORDER(fix, a, -1.0);
  TCICT_EXPECT_ZEROTH_ORDER(fix, b, 2.5);
#else
  (void)fix;
#endif
}

// --- convert: real to complex element type ---

// TCAPI V1: convert converts `t1` "into a destination tensor, possibly
// changing the element type, the underlying TCF/backend, or both", writing
// the result into `t2`, the tensor on the destination context, so the test
// reads it through that context. Converting a real tensor to
// tci::cplx_ten_t<RealTenT> keeps each value as its real part with a zero
// imaginary part, the one reading that preserves the values; every value here
// is a small integer, so the copy is exact. The input is read back as
// "Out-of-place inputs" in CONTRIBUTING.md describes.
template <typename RealTenT>
void test_convert_real_to_complex(tci_test_fixture<RealTenT> &fix) {
#if !defined(TCICT_SKIP_CONVERT) && defined(TCICT_HAS_ELEM_SWEEP)
  static_assert(!is_complex_v<RealTenT>,
                "test_convert_real_to_complex takes a real tensor type");
  using CplxTenT = tci::cplx_ten_t<RealTenT>;
  auto &ctx = fix.context();
  const tci::shape_t<RealTenT> shape = {2, 3};
  auto at = ramp_over<RealTenT>(shape, -2.0);
  auto a = tci::zeros<RealTenT>(ctx, shape);
  TCICT_SET_ELEMS(fix, a, at);

  // The destination context comes from the backend's fixture for CplxTenT,
  // which releases it on every exit; b, declared after it, is released first.
  tci_test_fixture<CplxTenT> fix2;
  CplxTenT b;
  TCICT_ASSERT_NOTHROW(tci::convert(ctx, a, fix2.context(), b));
  const tci::shape_t<CplxTenT> b_shape(shape.begin(), shape.end());
  TCICT_EXPECT_ELEMS(
      fix2, b, b_shape,
      [&](const tci::elem_coors_t<CplxTenT> &c) {
        const tci::elem_coors_t<RealTenT> coors(c.begin(), c.end());
        return std::complex<double>(at(coors), 0.0);
      },
      tol_category::exact);

  TCICT_EXPECT_ELEMS(fix, a, shape, at, tol_category::exact);
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
// tci::supported_ten_types_t in TenT's TCF that hold TenT's kind of element,
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
void test_convert_precision(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_CONVERT) && !defined(TCICT_SKIP_SUPPORTED_TEN_TYPES) &&               \
    defined(TCICT_HAS_ELEM_SWEEP)
  auto &ctx = fix.context();
  const tci::shape_t<TenT> shape = {2, 3};
  auto at = ramp_over<TenT>(shape, -2.0);
  auto a = tci::zeros<TenT>(ctx, shape);
  TCICT_SET_ELEMS(fix, a, at);

  detail::for_each_tuple_type<tci::supported_ten_types_t>([&](auto tag) {
    using DestTenT = typename decltype(tag)::type;
    if constexpr (std::is_same_v<tci::context_handle_t<DestTenT>,
                                 tci::context_handle_t<TenT>> &&
                  is_complex_v<DestTenT> == is_complex_v<TenT> &&
                  !std::is_same_v<tci::real_t<DestTenT>, tci::real_t<TenT>>) {
      // b, declared after fix2, is released first.
      tci_test_fixture<DestTenT> fix2;
      DestTenT b;
      TCICT_ASSERT_NOTHROW(tci::convert(ctx, a, fix2.context(), b));
      const tci::shape_t<DestTenT> b_shape(shape.begin(), shape.end());
      // DestTenT holds TenT's kind of element, so a real value stands for the
      // same element in both.
      TCICT_EXPECT_ELEMS(
          fix2, b, b_shape,
          [&](const tci::elem_coors_t<DestTenT> &c) {
            const tci::elem_coors_t<TenT> coors(c.begin(), c.end());
            return at(coors);
          },
          tol_category::exact);
    }
  });

  TCICT_EXPECT_ELEMS(fix, a, shape, at, tol_category::exact);
#else
  (void)fix;
#endif
}

// --- supported_ten_types_t ---

// TCAPI V1: "Each C++ implementation provides a compile-time catalog of its
// tensor types through `tcapi::supported_ten_types_t` in `tcapi/tcapi.h`", a
// std::tuple of them. "The catalog order is implementation-defined, and
// context compatibility is backend-dependent. Standard tuple traits and
// `tcapi::elem_t` expose its size, types, and element types". This suite spells
// V1's tcapi namespace as tci, so it reads the catalog as
// tci::supported_ten_types_t. The test asserts that the catalog is a
// std::tuple, and names tci::elem_t of each catalog
// type, which fails to compile for a type whose traits do not expose one. It
// also asserts a reading of "its tensor types": that the catalog holds TenT, a
// type the backend registers with the suite, and tci::real_ten_t<TenT> and
// tci::cplx_ten_t<TenT>, which V1 defines as tensor types and the backend's
// functions return. It asserts neither the order nor anything about contexts.
template <typename TenT>
void test_supported_ten_types(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_SUPPORTED_TEN_TYPES
  using Catalog = tci::supported_ten_types_t;
  TCICT_ASSERT((detail::is_std_tuple_v<Catalog>));
  TCICT_ASSERT((detail::tuple_contains_v<TenT, Catalog>));
  TCICT_ASSERT((detail::tuple_contains_v<tci::real_ten_t<TenT>, Catalog>));
  TCICT_ASSERT((detail::tuple_contains_v<tci::cplx_ten_t<TenT>, Catalog>));
  detail::for_each_tuple_type<Catalog>([](auto tag) {
    using ElemT = tci::elem_t<typename decltype(tag)::type>;
    [[maybe_unused]] ElemT *elem_t_named = nullptr;
  });
#endif
  (void)fix;
}

// --- version ---

template <typename TenT> void test_version(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_VERSION
  std::string ver = tci::template version<TenT>();
  TCICT_ASSERT(!ver.empty());
  // Version string should contain digits and a dot
  bool has_version_pattern =
      (ver.find_first_of("0123456789") != std::string::npos) &&
      (ver.find('.') != std::string::npos);
  TCICT_ASSERT(has_version_pattern);
#else
  (void)fix;
#endif
}

} // namespace tests
} // namespace tcict

// Bulk registration helper: invokes X(..., "category", test_fn) once per test.
// See include/tcict/adapters/doctest.h for usage.
#define TCICT_FOREACH_MISCELLANEOUS_TEST_ALL_TYPES(X, ...) \
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
#define TCICT_FOREACH_MISCELLANEOUS_TEST_REAL_ONLY(X, ...) \
  X(__VA_ARGS__, "miscellaneous", test_convert_real_to_complex)

// CPLX_ONLY: body is wrapped in `if constexpr (is_complex_v<TenT>)`; running
//   for real TenT would be a no-op, so skip registration.
#define TCICT_FOREACH_MISCELLANEOUS_TEST_CPLX_ONLY(X, ...) \
  X(__VA_ARGS__, "miscellaneous", test_close_complex_modulus)
