#pragma once

#include <tcapict/assertion.h>
#include <tcapict/diagonal.h>
#include <tcapict/elem_helper.h>
#include <tcapict/elem_sweep.h>
#include <tcapict/fixture.h>
#include <tcapict/skip.h>
#include <tcapict/zeroth_order.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
#include <vector>

namespace tcapict {
namespace tests {

// --- shrink (in-place) ---

template <typename TenT> void test_shrink_inplace(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_SHRINK
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto tensor = tcapi::zeros<TenT>(ctx, {3, 3});

  // Fill with values 1-9
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      tcapi::set_elem(ctx, tensor,
                      {static_cast<tcapi::elem_coor_t<TenT>>(i),
                       static_cast<tcapi::elem_coor_t<TenT>>(j)},
                      make_elem<TenT>(i * 3 + j + 1));
    }
  }

  // Shrink to top-left 2x2
  tcapi::bond_idx_elem_coor_pair_map<TenT> shrink_map;
  shrink_map[0] = std::make_pair(static_cast<tcapi::elem_coor_t<TenT>>(0),
                                 static_cast<tcapi::elem_coor_t<TenT>>(2));
  shrink_map[1] = std::make_pair(static_cast<tcapi::elem_coor_t<TenT>>(0),
                                 static_cast<tcapi::elem_coor_t<TenT>>(2));
  tcapi::shrink(ctx, tensor, shrink_map);

  auto result_shape = tcapi::shape(ctx, tensor);
  TCAPICT_ASSERT(result_shape[0] == 2);
  TCAPICT_ASSERT(result_shape[1] == 2);
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(tcapi::get_elem(ctx, tensor, {0, 0})), 1.0,
                       tol);
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(tcapi::get_elem(ctx, tensor, {0, 1})), 2.0,
                       tol);
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(tcapi::get_elem(ctx, tensor, {1, 0})), 4.0,
                       tol);
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(tcapi::get_elem(ctx, tensor, {1, 1})), 5.0,
                       tol);
#else
  (void)fix;
#endif
}

// --- shrink (out-of-place) ---

template <typename TenT>
void test_shrink_outofplace(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_SHRINK
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto input = tcapi::zeros<TenT>(ctx, {4, 4});

  // Set values in center 2x2 region [1:3, 1:3]
  tcapi::set_elem(ctx, input, {1, 1}, make_elem<TenT>(11.0));
  tcapi::set_elem(ctx, input, {1, 2}, make_elem<TenT>(12.0));
  tcapi::set_elem(ctx, input, {2, 1}, make_elem<TenT>(21.0));
  tcapi::set_elem(ctx, input, {2, 2}, make_elem<TenT>(22.0));

  tcapi::bond_idx_elem_coor_pair_map<TenT> shrink_map;
  shrink_map[0] = std::make_pair(static_cast<tcapi::elem_coor_t<TenT>>(1),
                                 static_cast<tcapi::elem_coor_t<TenT>>(3));
  shrink_map[1] = std::make_pair(static_cast<tcapi::elem_coor_t<TenT>>(1),
                                 static_cast<tcapi::elem_coor_t<TenT>>(3));

  TenT output;
  tcapi::shrink(ctx, input, shrink_map, output);

  auto result_shape = tcapi::shape(ctx, output);
  TCAPICT_ASSERT(result_shape[0] == 2);
  TCAPICT_ASSERT(result_shape[1] == 2);
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(tcapi::get_elem(ctx, output, {0, 0})), 11.0,
                       tol);
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(tcapi::get_elem(ctx, output, {0, 1})), 12.0,
                       tol);
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(tcapi::get_elem(ctx, output, {1, 0})), 21.0,
                       tol);
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(tcapi::get_elem(ctx, output, {1, 1})), 22.0,
                       tol);
#else
  (void)fix;
#endif
}

// --- shrink preserves complex values ---

template <typename TenT>
void test_shrink_complex_values(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_SHRINK
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto tensor = tcapi::zeros<TenT>(ctx, {3, 3});

  tcapi::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(1.5, 2.5));
  tcapi::set_elem(ctx, tensor, {0, 1}, make_elem<TenT>(3.5, 4.5));
  tcapi::set_elem(ctx, tensor, {1, 0}, make_elem<TenT>(5.5, 6.5));
  tcapi::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(7.5, 8.5));

  tcapi::bond_idx_elem_coor_pair_map<TenT> shrink_map;
  shrink_map[0] = std::make_pair(static_cast<tcapi::elem_coor_t<TenT>>(0),
                                 static_cast<tcapi::elem_coor_t<TenT>>(2));
  shrink_map[1] = std::make_pair(static_cast<tcapi::elem_coor_t<TenT>>(0),
                                 static_cast<tcapi::elem_coor_t<TenT>>(2));

  TenT output;
  tcapi::shrink(ctx, tensor, shrink_map, output);

  auto e00 = tcapi::get_elem(ctx, output, {0, 0});
  auto e01 = tcapi::get_elem(ctx, output, {0, 1});
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(e00), 1.5, tol);
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(e01), 3.5, tol);
  if constexpr (is_complex_v<TenT>) {
    TCAPICT_ASSERT_CLOSE(imag_part<TenT>(e00), 2.5, tol);
    TCAPICT_ASSERT_CLOSE(imag_part<TenT>(e01), 4.5, tol);
  }
#else
  (void)fix;
#endif
}

// --- real extraction (out-of-place) ---

template <typename TenT>
void test_real_extraction(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_REAL
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto tensor = tcapi::zeros<TenT>(ctx, {2, 2});
  tcapi::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(3.14, 2.71));
  tcapi::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(-1.59, 0.58));

  auto real_tensor = tcapi::real(ctx, tensor);

  using RealTenT = tcapi::real_ten_t<TenT>;
  auto elem00 = tcapi::get_elem(ctx, real_tensor, {0, 0});
  auto elem11 = tcapi::get_elem(ctx, real_tensor, {1, 1});
  TCAPICT_ASSERT_CLOSE(real_part<RealTenT>(elem00), 3.14, tol);
  TCAPICT_ASSERT_CLOSE(real_part<RealTenT>(elem11), -1.59, tol);
#else
  (void)fix;
#endif
}

// --- imag extraction (out-of-place) ---

template <typename TenT>
void test_imag_extraction(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_IMAG
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto tensor = tcapi::zeros<TenT>(ctx, {2, 2});
  tcapi::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(3.14, 2.71));
  tcapi::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(-1.59, 0.58));

  auto imag_tensor = tcapi::imag(ctx, tensor);

  using RealTenT = tcapi::real_ten_t<TenT>;
  auto elem00 = tcapi::get_elem(ctx, imag_tensor, {0, 0});
  auto elem11 = tcapi::get_elem(ctx, imag_tensor, {1, 1});
  if constexpr (is_complex_v<TenT>) {
    // Complex input: imag(tensor) extracts the imaginary parts set above.
    TCAPICT_ASSERT_CLOSE(real_part<RealTenT>(elem00), 2.71, tol);
    TCAPICT_ASSERT_CLOSE(real_part<RealTenT>(elem11), 0.58, tol);
  } else {
    // Real input. V1: "If `tcapi::elem_t<TenT>` is already real, the function
    // returns a real-valued tensor of the same shape whose elements are all
    // zero."
    TCAPICT_ASSERT_CLOSE(real_part<RealTenT>(elem00), 0.0, tol);
    TCAPICT_ASSERT_CLOSE(real_part<RealTenT>(elem11), 0.0, tol);
  }
#else
  (void)fix;
#endif
}

// --- real / imag extraction: zeroth-order ---

// Their results have type real_ten_t<TenT>, which is not TenT when TenT is
// complex, so TCAPICT_EXPECT_ZEROTH_ORDER cannot take them with this test's
// fixture; they are checked through TCAPICT_EXPECT_ZEROTH_ORDER_SHAPE and
// get_elem.
template <typename TenT>
void test_real_zeroth_order(tcapi_test_fixture<TenT> &fix) {
#if !defined(TCAPICT_SKIP_REAL) && defined(TCAPICT_HAS_ZEROTH_ORDER_SWEEP)
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto tensor = tcapi::zeros<TenT>(ctx, {});
  TCAPICT_SET_ZEROTH_ORDER(fix, tensor, 4.0);

  auto real_tensor = tcapi::real(ctx, tensor);

  using RealTenT = tcapi::real_ten_t<TenT>;
  TCAPICT_EXPECT_ZEROTH_ORDER_SHAPE(ctx, real_tensor);
  TCAPICT_ASSERT_CLOSE(real_part<RealTenT>(tcapi::get_elem(ctx, real_tensor, {})),
                       4.0, tol);

  // V1: "If `tcapi::elem_t<TenT>` is already real, this function returns a deep
  // copy of `in`", so for a real TenT a write to the input afterwards leaves
  // the result as it was.
  if constexpr (!is_complex_v<TenT>) {
    TCAPICT_SET_ZEROTH_ORDER(fix, tensor, -7.0);
    TCAPICT_ASSERT_CLOSE(
        real_part<RealTenT>(tcapi::get_elem(ctx, real_tensor, {})), 4.0, tol);
  }
#else
  (void)fix;
#endif
}

template <typename TenT>
void test_imag_zeroth_order(tcapi_test_fixture<TenT> &fix) {
#if !defined(TCAPICT_SKIP_IMAG) && defined(TCAPICT_HAS_ZEROTH_ORDER_SWEEP)
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto tensor = tcapi::zeros<TenT>(ctx, {});
  TCAPICT_SET_ZEROTH_ORDER(fix, tensor, 4.0);

  auto imag_tensor = tcapi::imag(ctx, tensor);

  // written<TenT>(4.0) has imaginary part -2.0 for a complex TenT and 0 for a
  // real one, whose imag is a zero tensor.
  using RealTenT = tcapi::real_ten_t<TenT>;
  TCAPICT_EXPECT_ZEROTH_ORDER_SHAPE(ctx, imag_tensor);
  TCAPICT_ASSERT_CLOSE(real_part<RealTenT>(tcapi::get_elem(ctx, imag_tensor, {})),
                       written<TenT>(4.0).imag(), tol);
#else
  (void)fix;
#endif
}

// --- cplx_conj (in-place) ---

template <typename TenT>
void test_cplx_conj_inplace(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_CPLX_CONJ
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto tensor = tcapi::zeros<TenT>(ctx, {2, 2});
  tcapi::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(1.0, 2.0));
  tcapi::set_elem(ctx, tensor, {0, 1}, make_elem<TenT>(-3.0, 4.0));
  tcapi::set_elem(ctx, tensor, {1, 0}, make_elem<TenT>(5.0, -6.0));
  tcapi::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(-7.0, -8.0));

  tcapi::cplx_conj(ctx, tensor);

  // Real parts unchanged for both real and complex: of a real TenT V1 says
  // "(1) performs no change". Imaginary parts only exist for complex.
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(tcapi::get_elem(ctx, tensor, {0, 0})), 1.0,
                       tol);
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(tcapi::get_elem(ctx, tensor, {0, 1})), -3.0,
                       tol);
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(tcapi::get_elem(ctx, tensor, {1, 0})), 5.0,
                       tol);
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(tcapi::get_elem(ctx, tensor, {1, 1})), -7.0,
                       tol);
  if constexpr (is_complex_v<TenT>) {
    TCAPICT_ASSERT_CLOSE(imag_part<TenT>(tcapi::get_elem(ctx, tensor, {0, 0})), -2.0,
                         tol);
    TCAPICT_ASSERT_CLOSE(imag_part<TenT>(tcapi::get_elem(ctx, tensor, {0, 1})), -4.0,
                         tol);
    TCAPICT_ASSERT_CLOSE(imag_part<TenT>(tcapi::get_elem(ctx, tensor, {1, 0})), 6.0,
                         tol);
    TCAPICT_ASSERT_CLOSE(imag_part<TenT>(tcapi::get_elem(ctx, tensor, {1, 1})), 8.0,
                         tol);
  }
#else
  (void)fix;
#endif
}

// --- cplx_conj (out-of-place) ---

template <typename TenT>
void test_cplx_conj_outofplace(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_CPLX_CONJ
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto input = tcapi::zeros<TenT>(ctx, {2, 2});
  tcapi::set_elem(ctx, input, {0, 0}, make_elem<TenT>(3.14, 2.71));
  tcapi::set_elem(ctx, input, {1, 1}, make_elem<TenT>(-1.41, -1.73));

  TenT output;
  tcapi::cplx_conj(ctx, input, output);

  // Real parts hold for both real and complex: of a real TenT V1 says "(2)
  // produces a deep copy". Imaginary parts only exist for complex. `input` is
  // read back at {0, 0}; see "Out-of-place inputs" in CONTRIBUTING.md.
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(tcapi::get_elem(ctx, input, {0, 0})), 3.14,
                       tol);
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(tcapi::get_elem(ctx, output, {0, 0})), 3.14,
                       tol);
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(tcapi::get_elem(ctx, output, {1, 1})), -1.41,
                       tol);
  if constexpr (is_complex_v<TenT>) {
    TCAPICT_ASSERT_CLOSE(imag_part<TenT>(tcapi::get_elem(ctx, input, {0, 0})), 2.71,
                         tol);
    TCAPICT_ASSERT_CLOSE(imag_part<TenT>(tcapi::get_elem(ctx, output, {0, 0})), -2.71,
                         tol);
    TCAPICT_ASSERT_CLOSE(imag_part<TenT>(tcapi::get_elem(ctx, output, {1, 1})), 1.73,
                         tol);
  }
#else
  (void)fix;
#endif
}

// --- cplx_conj: zeroth-order ---

template <typename TenT>
void test_cplx_conj_zeroth_order(tcapi_test_fixture<TenT> &fix) {
#if !defined(TCAPICT_SKIP_CPLX_CONJ) && defined(TCAPICT_HAS_ZEROTH_ORDER_SWEEP)
  auto &ctx = fix.context();
  auto conj_of = [](double v) { return std::conj(written<TenT>(v)); };

  auto tensor = tcapi::zeros<TenT>(ctx, {});
  TCAPICT_SET_ZEROTH_ORDER(fix, tensor, 2.0);
  tcapi::cplx_conj(ctx, tensor);
  TCAPICT_EXPECT_ZEROTH_ORDER(fix, tensor, conj_of(2.0), tol_category::elementwise);

  // Out of place; the input is read back, see "Out-of-place inputs" in
  // CONTRIBUTING.md.
  auto input = tcapi::zeros<TenT>(ctx, {});
  TCAPICT_SET_ZEROTH_ORDER(fix, input, -3.0);
  TenT output;
  tcapi::cplx_conj(ctx, input, output);
  TCAPICT_EXPECT_ZEROTH_ORDER(fix, output, conj_of(-3.0), tol_category::elementwise);
  TCAPICT_EXPECT_ZEROTH_ORDER(fix, input, -3.0);

  // V1: "If `tcapi::elem_t<TenT>` is real, (1) performs no change and (2)
  // produces a deep copy", so for a real TenT a write to the input afterwards
  // leaves the output of the out-of-place overload (2) as it was.
  if constexpr (!is_complex_v<TenT>) {
    TCAPICT_SET_ZEROTH_ORDER(fix, input, 5.0);
    TCAPICT_EXPECT_ZEROTH_ORDER(fix, output, conj_of(-3.0), tol_category::elementwise);
  }
#else
  (void)fix;
#endif
}

// --- to_cplx (out-of-place, from real type) ---
// NOTE: TenT here should be a real tensor type (e.g., CytnxTensor<double>)

template <typename RealTenT>
void test_to_cplx_outofplace(tcapi_test_fixture<RealTenT> &fix) {
#ifndef TCAPICT_SKIP_TO_CPLX
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  RealTenT real_tensor;
  real_tensor = tcapi::zeros<RealTenT>(ctx, {2, 2});

  tcapi::set_elem(ctx, real_tensor, {0, 0},
                  static_cast<tcapi::elem_t<RealTenT>>(1.5));
  tcapi::set_elem(ctx, real_tensor, {0, 1},
                  static_cast<tcapi::elem_t<RealTenT>>(2.5));
  tcapi::set_elem(ctx, real_tensor, {1, 0},
                  static_cast<tcapi::elem_t<RealTenT>>(3.5));
  tcapi::set_elem(ctx, real_tensor, {1, 1},
                  static_cast<tcapi::elem_t<RealTenT>>(4.5));

  auto complex_tensor = tcapi::to_cplx(ctx, real_tensor);

  using CplxTenT = tcapi::cplx_ten_t<RealTenT>;
  auto elem00 = tcapi::get_elem(ctx, complex_tensor, {0, 0});
  auto elem11 = tcapi::get_elem(ctx, complex_tensor, {1, 1});
  TCAPICT_ASSERT_CLOSE(real_part<CplxTenT>(elem00), 1.5, tol);
  TCAPICT_ASSERT_CLOSE(imag_part<CplxTenT>(elem00), 0.0, tol);
  TCAPICT_ASSERT_CLOSE(real_part<CplxTenT>(elem11), 4.5, tol);
  TCAPICT_ASSERT_CLOSE(imag_part<CplxTenT>(elem11), 0.0, tol);
#else
  (void)fix;
#endif
}

// --- to_cplx (complex to complex) ---

template <typename TenT>
void test_to_cplx_complex_to_complex(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_TO_CPLX
  // For real TenT, tcapi::to_cplx returns cplx_ten_t<TenT> whose elements are
  // cplx_t<TenT>; calling imag_part<TenT>(cplx_elem) would be a type error.
  // This test therefore only runs when TenT is already complex.
  if constexpr (is_complex_v<TenT>) {
    auto &ctx = fix.context();
    auto tol = tolerance(fix, tol_category::elementwise);
    auto tensor = tcapi::zeros<TenT>(ctx, {2, 2});
    tcapi::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(3.14, 2.71));
    tcapi::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(-1.41, 1.73));

    auto result = tcapi::to_cplx(ctx, tensor);

    auto elem00 = tcapi::get_elem(ctx, result, {0, 0});
    auto elem11 = tcapi::get_elem(ctx, result, {1, 1});
    TCAPICT_ASSERT_CLOSE(real_part<TenT>(elem00), 3.14, tol);
    TCAPICT_ASSERT_CLOSE(imag_part<TenT>(elem00), 2.71, tol);
    TCAPICT_ASSERT_CLOSE(real_part<TenT>(elem11), -1.41, tol);
    TCAPICT_ASSERT_CLOSE(imag_part<TenT>(elem11), 1.73, tol);
  }
#else
  (void)fix;
#endif
}

// --- to_cplx: zeroth-order ---

// Registered for every TenT: V1 embeds a real input as real parts and gives a
// deep copy of a complex one. The result has type cplx_ten_t<TenT>, which is
// not TenT when TenT is real, so it is checked through
// TCAPICT_EXPECT_ZEROTH_ORDER_SHAPE and get_elem.
template <typename TenT>
void test_to_cplx_zeroth_order(tcapi_test_fixture<TenT> &fix) {
#if !defined(TCAPICT_SKIP_TO_CPLX) && defined(TCAPICT_HAS_ZEROTH_ORDER_SWEEP)
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto tensor = tcapi::zeros<TenT>(ctx, {});
  TCAPICT_SET_ZEROTH_ORDER(fix, tensor, -2.5);

  auto complex_tensor = tcapi::to_cplx(ctx, tensor);

  using CplxTenT = tcapi::cplx_ten_t<TenT>;
  const auto expected = written<TenT>(-2.5);
  TCAPICT_EXPECT_ZEROTH_ORDER_SHAPE(ctx, complex_tensor);
  const auto elem = tcapi::get_elem(ctx, complex_tensor, {});
  TCAPICT_ASSERT_CLOSE(real_part<CplxTenT>(elem), expected.real(), tol);
  TCAPICT_ASSERT_CLOSE(imag_part<CplxTenT>(elem), expected.imag(), tol);

  // V1: "If `tcapi::elem_t<TenT>` is already complex, the function returns a
  // deep copy of `in`", so for a complex TenT a write to the input afterwards
  // leaves the result as it was.
  if constexpr (is_complex_v<TenT>) {
    TCAPICT_SET_ZEROTH_ORDER(fix, tensor, 9.0);
    const auto after = tcapi::get_elem(ctx, complex_tensor, {});
    TCAPICT_ASSERT_CLOSE(real_part<CplxTenT>(after), expected.real(), tol);
    TCAPICT_ASSERT_CLOSE(imag_part<CplxTenT>(after), expected.imag(), tol);
  }
#else
  (void)fix;
#endif
}

// --- for_each: element doubling ---

template <typename TenT>
void test_for_each_doubling(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_FOR_EACH
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  using Elem = tcapi::elem_t<TenT>;

  auto tensor = tcapi::zeros<TenT>(ctx, {2, 3});
  tcapi::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(1.0));
  tcapi::set_elem(ctx, tensor, {0, 1}, make_elem<TenT>(2.0));
  tcapi::set_elem(ctx, tensor, {0, 2}, make_elem<TenT>(3.0));
  tcapi::set_elem(ctx, tensor, {1, 0}, make_elem<TenT>(4.0));
  tcapi::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(5.0));
  tcapi::set_elem(ctx, tensor, {1, 2}, make_elem<TenT>(6.0));

  tcapi::for_each(ctx, tensor,
                  [](Elem &elem) { elem = elem * make_elem<TenT>(2.0); });

  TCAPICT_ASSERT_CLOSE(real_part<TenT>(tcapi::get_elem(ctx, tensor, {0, 0})), 2.0,
                       tol);
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(tcapi::get_elem(ctx, tensor, {0, 2})), 6.0,
                       tol);
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(tcapi::get_elem(ctx, tensor, {1, 2})), 12.0,
                       tol);
#else
  (void)fix;
#endif
}

// --- for_each: iteration and summation ---

template <typename TenT>
void test_for_each_summation(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_FOR_EACH
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::reduction, 4);
  using Elem = tcapi::elem_t<TenT>;

  auto tensor = tcapi::zeros<TenT>(ctx, {2, 2});
  tcapi::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(1.0));
  tcapi::set_elem(ctx, tensor, {0, 1}, make_elem<TenT>(2.0));
  tcapi::set_elem(ctx, tensor, {1, 0}, make_elem<TenT>(3.0));
  tcapi::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(4.0));

  int count = 0;
  Elem sum = make_elem<TenT>(0.0);
  tcapi::for_each(ctx, tensor, [&count, &sum](Elem &elem) {
    count++;
    sum = sum + elem;
  });

  TCAPICT_ASSERT(count == 4);
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(sum), 10.0, tol);
#else
  (void)fix;
#endif
}

// --- for_each: scalar multiplication with capture ---

template <typename TenT>
void test_for_each_capture(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_FOR_EACH
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  using Elem = tcapi::elem_t<TenT>;

  auto tensor = tcapi::fill<TenT>(ctx, {2, 2}, make_elem<TenT>(3.0, 1.0));

  auto multiplier = make_elem<TenT>(0.5);
  tcapi::for_each(ctx, tensor,
                  [multiplier](Elem &elem) { elem = elem * multiplier; });

  auto result = tcapi::get_elem(ctx, tensor, {0, 0});
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(result), 1.5, tol);
  if constexpr (is_complex_v<TenT>) {
    TCAPICT_ASSERT_CLOSE(imag_part<TenT>(result), 0.5, tol);
  }
#else
  (void)fix;
#endif
}

// --- for_each: const version ---

template <typename TenT> void test_for_each_const(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_FOR_EACH
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::reduction, 3);
  using Elem = tcapi::elem_t<TenT>;

  auto tensor = tcapi::fill<TenT>(ctx, {3}, make_elem<TenT>(2.0, 3.0));

  Elem sum = make_elem<TenT>(0.0);
  tcapi::for_each(ctx, static_cast<const TenT &>(tensor),
                  [&sum](const Elem &elem) { sum = sum + elem; });

  TCAPICT_ASSERT_CLOSE(real_part<TenT>(sum), 6.0, tol);
  if constexpr (is_complex_v<TenT>) {
    TCAPICT_ASSERT_CLOSE(imag_part<TenT>(sum), 9.0, tol);
  }
#else
  (void)fix;
#endif
}

// --- for_each: element-wise inversion ---

template <typename TenT>
void test_for_each_inversion(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_FOR_EACH
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  using Elem = tcapi::elem_t<TenT>;

  auto tensor = tcapi::fill<TenT>(ctx, {2, 2}, make_elem<TenT>(0.5));

  tcapi::for_each(ctx, tensor, [](Elem &elem) {
    if (std::abs(elem) > 1e-12) {
      elem = make_elem<TenT>(1.0, 0.0) / elem;
    }
  });

  auto result = tcapi::get_elem(ctx, tensor, {0, 0});
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(result), 2.0, tol);
#else
  (void)fix;
#endif
}

// --- for_each: zeroth-order ---

// V1's for_each: "Visit every element exactly once", and a zeroth-order tensor
// has one.
template <typename TenT>
void test_for_each_zeroth_order(tcapi_test_fixture<TenT> &fix) {
#if !defined(TCAPICT_SKIP_FOR_EACH) && defined(TCAPICT_HAS_ZEROTH_ORDER_SWEEP)
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  using Elem = tcapi::elem_t<TenT>;
  auto tensor = tcapi::zeros<TenT>(ctx, {});
  TCAPICT_SET_ZEROTH_ORDER(fix, tensor, 1.5);

  std::size_t visits = 0;
  Elem passed = make_elem<TenT>(0.0);
  tcapi::for_each(ctx, static_cast<const TenT &>(tensor),
                  [&visits, &passed](const Elem elem) {
                    ++visits;
                    passed = elem;
                  });
  TCAPICT_ASSERT(visits == 1);
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(passed), written<TenT>(1.5).real(), tol);
  TCAPICT_ASSERT_CLOSE(imag_part<TenT>(passed), written<TenT>(1.5).imag(), tol);

  visits = 0;
  tcapi::for_each(ctx, tensor, [&visits](Elem &elem) {
    ++visits;
    elem = elem * make_elem<TenT>(2.0);
  });
  TCAPICT_ASSERT(visits == 1);
  TCAPICT_EXPECT_ZEROTH_ORDER(fix, tensor, 3.0, tol_category::elementwise);
#else
  (void)fix;
#endif
}

// --- for_each_with_coors ---

// The coordinates a traversal of a 2x2 tensor visits: record() runs in the
// callback under "Callbacks the backend invokes" in CONTRIBUTING.md, and
// check() runs after the call returns.
struct visit_record_2x2 {
  bool seen[2][2] = {{false, false}, {false, false}};
  bool malformed_coordinate = false;
  bool revisited = false;

  // Returns false for a coordinate coors_2d rejects; the callback then leaves
  // the element alone.
  template <typename Coors>
  bool record(const Coors &coors, std::ptrdiff_t &row, std::ptrdiff_t &col) {
    if (!coors_2d(coors, 2, 2, row, col)) {
      malformed_coordinate = true;
      return false;
    }
    revisited = revisited || seen[row][col];
    seen[row][col] = true;
    return true;
  }

  // V1 gives `for_each_with_coors` as "similar to `for_each`, but invokes `f`
  // with both the element and its coordinates", and the suite reads
  // `for_each`'s "Visit every element exactly once" as carrying over. No
  // malformed coordinate, no revisit and every one of the 4 coordinates seen
  // is that, and it excludes a backend walking only the 2 entries a diagonal
  // tensor stores.
  void check() const {
    TCAPICT_ASSERT(!malformed_coordinate);
    TCAPICT_ASSERT(!revisited);
    for (std::size_t i = 0; i < 2; ++i) {
      for (std::size_t j = 0; j < 2; ++j) {
        TCAPICT_ASSERT(seen[i][j]);
      }
    }
  }
};

// --- for_each_with_coors: mutable ---

template <typename TenT>
void test_for_each_with_coors(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_FOR_EACH_WITH_COORS
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  using Elem = tcapi::elem_t<TenT>;

  TenT a = tcapi::template eye<TenT>(ctx, 2);

  visit_record_2x2 visited;
  tcapi::for_each_with_coors(
      ctx, a, [&](Elem &elem, const tcapi::elem_coors_t<TenT> &coors) {
        std::ptrdiff_t row = 0;
        std::ptrdiff_t col = 0;
        if (visited.record(coors, row, col) && row == col) {
          elem = static_cast<Elem>(2.0);
        }
      });
  visited.check();

  for (std::size_t i = 0; i < 2; ++i) {
    const auto elem = tcapi::get_elem(ctx, a, coors_of<TenT>(i, i));
    TCAPICT_ASSERT_CLOSE(real_part<TenT>(elem), 2.0, tol);
    if constexpr (is_complex_v<TenT>) {
      TCAPICT_ASSERT_CLOSE(imag_part<TenT>(elem), 0.0, tol);
    }
  }
  TCAPICT_ASSERT_CLOSE(std::abs(tcapi::get_elem(ctx, a, {0, 1})), 0.0, tol);
  TCAPICT_ASSERT_CLOSE(std::abs(tcapi::get_elem(ctx, a, {1, 0})), 0.0, tol);
#else
  (void)fix;
#endif
}

// --- for_each_with_coors: const version ---

template <typename TenT>
void test_for_each_with_coors_const(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_FOR_EACH_WITH_COORS
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  using Elem = tcapi::elem_t<TenT>;

  TenT a = tcapi::template eye<TenT>(ctx, 2);
  const TenT &const_a = a;

  // Each element as passed to the callback, checked one by one below.
  Elem passed[2][2] = {};
  visit_record_2x2 visited;
  tcapi::for_each_with_coors(
      ctx, const_a, [&](const Elem &elem, const tcapi::elem_coors_t<TenT> &coors) {
        std::ptrdiff_t row = 0;
        std::ptrdiff_t col = 0;
        if (!visited.record(coors, row, col)) {
          return;
        }
        passed[row][col] = elem;
      });
  visited.check();

  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 2; ++j) {
      if (i == j) {
        TCAPICT_ASSERT_CLOSE(real_part<TenT>(passed[i][j]), 1.0, tol);
        if constexpr (is_complex_v<TenT>) {
          TCAPICT_ASSERT_CLOSE(imag_part<TenT>(passed[i][j]), 0.0, tol);
        }
      } else {
        TCAPICT_ASSERT_CLOSE(std::abs(passed[i][j]), 0.0, tol);
      }
    }
  }
#else
  (void)fix;
#endif
}

// --- for_each_with_coors: zeroth-order ---

// One visit, whose coordinate is the empty sequence.
template <typename TenT>
void test_for_each_with_coors_zeroth_order(tcapi_test_fixture<TenT> &fix) {
#if !defined(TCAPICT_SKIP_FOR_EACH_WITH_COORS) && defined(TCAPICT_HAS_ZEROTH_ORDER_SWEEP)
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  using Elem = tcapi::elem_t<TenT>;
  auto tensor = tcapi::zeros<TenT>(ctx, {});
  TCAPICT_SET_ZEROTH_ORDER(fix, tensor, 1.5);

  Elem passed = make_elem<TenT>(0.0);
  visit_record_0d read_only;
  tcapi::for_each_with_coors(
      ctx, static_cast<const TenT &>(tensor),
      [&](const Elem &elem, const tcapi::elem_coors_t<TenT> &coors) {
        if (read_only.record(coors)) {
          passed = elem;
        }
      });
  read_only.check();
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(passed), written<TenT>(1.5).real(), tol);
  TCAPICT_ASSERT_CLOSE(imag_part<TenT>(passed), written<TenT>(1.5).imag(), tol);

  visit_record_0d mutating;
  tcapi::for_each_with_coors(
      ctx, tensor, [&](Elem &elem, const tcapi::elem_coors_t<TenT> &coors) {
        if (mutating.record(coors)) {
          elem = elem * make_elem<TenT>(2.0);
        }
      });
  mutating.check();
  TCAPICT_EXPECT_ZEROTH_ORDER(fix, tensor, 3.0, tol_category::elementwise);
#else
  (void)fix;
#endif
}

// --- for_each: traversal over a diagonal tensor ---

// V1's rule for `for_each` is "Visit every element exactly once". `eye(3)` has
// 9 logical elements — a diagonal tensor's off-diagonal zeros are elements too
// — so a backend traversing the 3 entries it stores is excluded by the visit
// count, which is what this test adds over reading values.
template <typename TenT>
void test_for_each_eye_traversal(tcapi_test_fixture<TenT> &fix) {
#if !defined(TCAPICT_SKIP_FOR_EACH) && !defined(TCAPICT_SKIP_EYE)
  auto &ctx = fix.context();
  auto sum_tol = tolerance(fix, tol_category::reduction, 9);
  using Elem = tcapi::elem_t<TenT>;

  auto identity = tcapi::template eye<TenT>(ctx, 3);

  const auto one = make_elem<TenT>(1.0);
  const auto zero = make_elem<TenT>(0.0);
  std::size_t visits = 0;
  std::size_t ones = 0;
  std::size_t zeros = 0;
  Elem sum = make_elem<TenT>(0.0);

  // Classifying on the whole element rather than its real part keeps the
  // counts meaningful for the complex instantiations: a backend returning a
  // stray imaginary component would otherwise still be counted as a one. The
  // comparison is exact: eye's elements are exact ones and zeros (see
  // test_eye).
  tcapi::for_each(ctx, static_cast<const TenT &>(identity),
                  [&](const Elem &elem) {
                    ++visits;
                    sum = sum + elem;
                    if (elem == one) {
                      ++ones;
                    } else if (elem == zero) {
                      ++zeros;
                    }
                  });

  TCAPICT_ASSERT(visits == 9);
  TCAPICT_ASSERT(ones == 3);
  TCAPICT_ASSERT(zeros == 6);
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(sum), 3.0, sum_tol);
  if constexpr (is_complex_v<TenT>) {
    TCAPICT_ASSERT_CLOSE(imag_part<TenT>(sum), 0.0, sum_tol);
  }
#else
  (void)fix;
#endif
}

// --- distinct-value fixture shared by reshape and transpose ---

// Both tests need per-element values a constant fill cannot distinguish: with
// one value repeated, a dropped, duplicated, or misplaced element leaves the
// tensor identical to a correct one. The offset keeps every value
// non-zero, so a slot left untouched by an incomplete write is distinguishable
// too. fill_ramp_2x3x4 writes -0.5 times each value as the element's imaginary
// part, which lets a comparison on real parts extend to whole elements.
inline double ramp_2x3x4(std::size_t i, std::size_t j, std::size_t k) {
  return static_cast<double>(i * 12 + j * 4 + k) + 1.0;
}

// Each helper below that calls a tcapi:: API is compiled exactly when at least
// one of its callers is, so skipping all of its callers also removes its
// references to those APIs.

#if !defined(TCAPICT_SKIP_RESHAPE) || !defined(TCAPICT_SKIP_TRANSPOSE)
// Writes the ramp into a {2, 3, 4} tensor, after asserting it has that shape.
// Called by test_reshape and test_transpose.
template <typename TenT>
void fill_ramp_2x3x4(tcapi_test_fixture<TenT> &fix, TenT &tensor) {
  auto &ctx = fix.context();
  const tcapi::shape_t<TenT> ramp_shape = {2, 3, 4};
  TCAPICT_ASSERT(tcapi::shape(ctx, tensor) == ramp_shape);
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      for (std::size_t k = 0; k < 4; ++k) {
        const double value = ramp_2x3x4(i, j, k);
        tcapi::set_elem(ctx, tensor, coors_of<TenT>(i, j, k),
                        make_elem<TenT>(value, -0.5 * value));
      }
    }
  }
}
#endif

#ifndef TCAPICT_SKIP_TRANSPOSE
// Called by test_transpose. Asserts that `tensor` has `shape`, then that every
// ramp value is readable at the coordinate `coors_at` maps it to. Callers
// supply the shape and the mapping, so the same sweep serves an output whose
// axes were permuted and an input that must have stayed put.
template <typename TenT, typename CoorsAt>
void expect_ramp_2x3x4(tcapi_test_fixture<TenT> &fix, const TenT &tensor,
                       const tcapi::shape_t<TenT> &shape, CoorsAt coors_at) {
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  TCAPICT_ASSERT(tcapi::shape(ctx, tensor) == shape);
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      for (std::size_t k = 0; k < 4; ++k) {
        const double value = ramp_2x3x4(i, j, k);
        auto elem = tcapi::get_elem(ctx, tensor, coors_at(i, j, k));
        TCAPICT_ASSERT_CLOSE(real_part<TenT>(elem), value, tol);
        if constexpr (is_complex_v<TenT>) {
          TCAPICT_ASSERT_CLOSE(imag_part<TenT>(elem), -0.5 * value, tol);
        }
      }
    }
  }
}
#endif

// --- reshape (in-place) ---

template <typename TenT> void test_reshape(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_RESHAPE
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto tensor = tcapi::zeros<TenT>(ctx, {2, 3, 4});
  fill_ramp_2x3x4(fix, tensor);

  tcapi::shape_t<TenT> new_shape = {6, 4};
  TCAPICT_ASSERT_NOTHROW(tcapi::reshape(ctx, tensor, new_shape));
  TCAPICT_ASSERT(tcapi::shape(ctx, tensor) == new_shape);
  TCAPICT_ASSERT(tcapi::size(ctx, tensor) == 24);

  // V1 says of reshape: "No element reordering, transposition, or value
  // modification occurs; the linear order of logical tensor elements is
  // preserved." But it defines that linear order for no shape, so which
  // coordinate a given element lands on is not portably assertable. The
  // multiset of values is: it holds under any ordering, and still catches a
  // dropped, duplicated, or corrupted element.
  std::vector<double> observed;
  observed.reserve(24);
  for (std::size_t i = 0; i < 6; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      auto elem = tcapi::get_elem(ctx, tensor, coors_of<TenT>(i, j));
      const double value = real_part<TenT>(elem);
      // A NaN breaks the strict weak ordering std::sort requires; fail here
      // instead of sorting with one.
      TCAPICT_ASSERT(!std::isnan(value));
      observed.push_back(value);
      if constexpr (is_complex_v<TenT>) {
        // Pins the imaginary part to its own real part, which extends the
        // multiset comparison below from real parts to whole elements.
        TCAPICT_ASSERT_CLOSE(imag_part<TenT>(elem), -0.5 * value, tol);
      }
    }
  }

  // The ramp yields 1, 2, ..., 24, so the sorted observed values must be
  // exactly that sequence.
  std::sort(observed.begin(), observed.end());
  for (std::size_t n = 0; n < observed.size(); ++n) {
    TCAPICT_ASSERT_CLOSE(observed[n], static_cast<double>(n + 1), tol);
  }
#else
  (void)fix;
#endif
}

// --- reshape: zeroth-order ---

// TCAPI V1: reshape "also converts between the empty shape and any shape with
// total size 1, such as {1} or {1,1}", with "No element reordering,
// transposition, or value modification". One element has one position, so its
// value is assertable at every shape.
template <typename TenT>
void test_reshape_zeroth_order(tcapi_test_fixture<TenT> &fix) {
#if !defined(TCAPICT_SKIP_RESHAPE) && defined(TCAPICT_HAS_ZEROTH_ORDER_SWEEP)
  auto &ctx = fix.context();
  auto value = [](const tcapi::elem_coors_t<TenT> &) { return 2.75; };

  auto tensor = tcapi::zeros<TenT>(ctx, {});
  TCAPICT_SET_ZEROTH_ORDER(fix, tensor, 2.75);
  TCAPICT_ASSERT_NOTHROW(tcapi::reshape(ctx, tensor, {1}));
  TCAPICT_EXPECT_ELEMS(fix, tensor, {1}, value, tol_category::exact);
  TCAPICT_ASSERT_NOTHROW(tcapi::reshape(ctx, tensor, {1, 1}));
  TCAPICT_EXPECT_ELEMS(fix, tensor, {1, 1}, value, tol_category::exact);
  TCAPICT_ASSERT_NOTHROW(tcapi::reshape(ctx, tensor, {}));
  TCAPICT_EXPECT_ZEROTH_ORDER(fix, tensor, 2.75);

  // From a tensor built at shape {1}.
  auto one = tcapi::zeros<TenT>(ctx, {1});
  TCAPICT_SET_ELEMS(fix, one, [](const tcapi::elem_coors_t<TenT> &) { return 0.5; });
  TCAPICT_ASSERT_NOTHROW(tcapi::reshape(ctx, one, {}));
  TCAPICT_EXPECT_ZEROTH_ORDER(fix, one, 0.5);

  // Out of place; the input is read back, see "Out-of-place inputs" in
  // CONTRIBUTING.md.
  auto input = tcapi::zeros<TenT>(ctx, {});
  TCAPICT_SET_ZEROTH_ORDER(fix, input, -1.5);
  TenT output;
  TCAPICT_ASSERT_NOTHROW(tcapi::reshape(ctx, input, {1}, output));
  TCAPICT_EXPECT_ELEMS(
      fix, output, {1}, [](const tcapi::elem_coors_t<TenT> &) { return -1.5; },
      tol_category::exact);
  TCAPICT_EXPECT_ZEROTH_ORDER(fix, input, -1.5);
#else
  (void)fix;
#endif
}

// --- transpose (out-of-place) ---

template <typename TenT> void test_transpose(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_TRANSPOSE
  auto &ctx = fix.context();
  tcapi::shape_t<TenT> original_shape = {2, 3, 4};
  auto tensor = tcapi::zeros<TenT>(ctx, original_shape);
  fill_ramp_2x3x4(fix, tensor);

  TenT transposed;
  tcapi::List<tcapi::bond_idx_t<TenT>> new_order = {2, 0, 1};
  TCAPICT_ASSERT_NOTHROW(tcapi::transpose(ctx, tensor, new_order, transposed));

  // V1 says only "Permute tensor bonds according to the permutation
  // `new_order`. Element values are preserved; only their coordinates are
  // permuted (the memory layout may also change).", and both of its examples
  // use orders that are their own inverses, so which way the permutation runs
  // is a reading. The suite reads new bond p as old bond new_order[p]: the
  // {2, 3, 4} input becomes {4, 2, 3}, and out[k, i, j] == a[i, j, k]. The
  // coordinate map follows from that reading, so no convention for the linear
  // order of elements enters into it.
  const tcapi::shape_t<TenT> expected_shape = {4, 2, 3};
  expect_ramp_2x3x4(fix, transposed, expected_shape,
                    [](std::size_t i, std::size_t j, std::size_t k) {
                      return coors_of<TenT>(k, i, j);
                    });

  // Input unchanged; see "Out-of-place inputs" in CONTRIBUTING.md.
  expect_ramp_2x3x4(fix, tensor, original_shape,
                    [](std::size_t i, std::size_t j, std::size_t k) {
                      return coors_of<TenT>(i, j, k);
                    });
#else
  (void)fix;
#endif
}

// --- concatenate: basic 2D ---

template <typename TenT>
void test_concatenate_basic(tcapi_test_fixture<TenT> &fix) {
#if !defined(TCAPICT_SKIP_CONCATENATE) && defined(TCAPICT_HAS_ELEM_SWEEP)
  auto &ctx = fix.context();
  // Distinct values across both inputs; the expected results are computed
  // from the same functions. See "Inputs to multi-input operations" in
  // CONTRIBUTING.md.
  auto t1_at = ramp_over<TenT>({2, 3}, 10.0);
  auto t2_at = ramp_over<TenT>({2, 3}, 20.0);
  auto t1 = tcapi::zeros<TenT>(ctx, {2, 3});
  auto t2 = tcapi::zeros<TenT>(ctx, {2, 3});
  TCAPICT_SET_ELEMS(fix, t1, t1_at);
  TCAPICT_SET_ELEMS(fix, t2, t2_at);

  // A named list; see "Inputs to multi-input operations" in CONTRIBUTING.md.
  tcapi::List<tcapi::CRef<TenT>> ins = {std::cref(t1), std::cref(t2)};

  // Vertical concatenation: rows 0-1 from t1, rows 2-3 from t2
  auto vertical = TCAPICT_NOTHROW_VALUE(tcapi::concatenate<TenT>(ctx, ins, 0));
  TCAPICT_EXPECT_ELEMS(fix, vertical, {4, 3}, [&](const auto &x) {
    return x[0] < 2 ? t1_at(x) : t2_at(shifted(x, 0, 2));
  }, tol_category::exact);

  // Horizontal concatenation: columns 0-2 from t1, columns 3-5 from t2
  auto horizontal = TCAPICT_NOTHROW_VALUE(tcapi::concatenate<TenT>(ctx, ins, 1));
  TCAPICT_EXPECT_ELEMS(fix, horizontal, {2, 6}, [&](const auto &x) {
    return x[1] < 3 ? t1_at(x) : t2_at(shifted(x, 1, 3));
  }, tol_category::exact);

  // Inputs unchanged; see "Inputs to multi-input operations" in
  // CONTRIBUTING.md.
  TCAPICT_EXPECT_ELEMS(fix, t1, {2, 3}, t1_at, tol_category::exact);
  TCAPICT_EXPECT_ELEMS(fix, t2, {2, 3}, t2_at, tol_category::exact);
#else
  (void)fix;
#endif
}

// --- concatenate: multi-tensor with value verification ---

template <typename TenT>
void test_concatenate_values(tcapi_test_fixture<TenT> &fix) {
#if !defined(TCAPICT_SKIP_CONCATENATE) && defined(TCAPICT_HAS_ELEM_SWEEP)
  auto &ctx = fix.context();
  // Distinct values across all three inputs; the expected result is computed
  // from the same functions. See "Inputs to multi-input operations" in
  // CONTRIBUTING.md.
  // a holds 1..24, b 25..32 and c 33..48: small integers, so a float
  // represents each value, and its imaginary part, exactly.
  auto a_at = ramp_over<TenT>({2, 3, 4}, 1.0);
  auto b_at = ramp_over<TenT>({2, 1, 4}, 25.0);
  auto c_at = ramp_over<TenT>({2, 2, 4}, 33.0);
  auto a = tcapi::zeros<TenT>(ctx, {2, 3, 4});
  auto b = tcapi::zeros<TenT>(ctx, {2, 1, 4});
  auto c = tcapi::zeros<TenT>(ctx, {2, 2, 4});
  TCAPICT_SET_ELEMS(fix, a, a_at);
  TCAPICT_SET_ELEMS(fix, b, b_at);
  TCAPICT_SET_ELEMS(fix, c, c_at);

  auto d = TCAPICT_NOTHROW_VALUE(tcapi::concatenate<TenT>(
      ctx, {std::cref(a), std::cref(b), std::cref(c)}, 1));

  // Bond 1 of d: 0-2 from a, 3 from b, 4-5 from c
  TCAPICT_EXPECT_ELEMS(fix, d, {2, 6, 4}, [&](const auto &x) {
    if (x[1] < 3) {
      return a_at(x);
    }
    if (x[1] < 4) {
      return b_at(shifted(x, 1, 3));
    }
    return c_at(shifted(x, 1, 4));
  }, tol_category::exact);

  // Inputs unchanged; see "Inputs to multi-input operations" in
  // CONTRIBUTING.md.
  TCAPICT_EXPECT_ELEMS(fix, a, {2, 3, 4}, a_at, tol_category::exact);
  TCAPICT_EXPECT_ELEMS(fix, b, {2, 1, 4}, b_at, tol_category::exact);
  TCAPICT_EXPECT_ELEMS(fix, c, {2, 2, 4}, c_at, tol_category::exact);
#else
  (void)fix;
#endif
}

// --- extract_sub (out-of-place) ---

template <typename TenT> void test_extract_sub(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_EXTRACT_SUB
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto a = tcapi::zeros<TenT>(ctx, {3, 4, 2});
  tcapi::set_elem(ctx, a, {1, 0, 0}, make_elem<TenT>(42.0));
  tcapi::set_elem(ctx, a, {2, 1, 1}, make_elem<TenT>(13.0));

  TenT sub;
  tcapi::List<tcapi::Pair<tcapi::elem_coor_t<TenT>, tcapi::elem_coor_t<TenT>>>
      coor_pairs = {{1, 3}, {0, 2}, {0, 2}};
  TCAPICT_ASSERT_NOTHROW(tcapi::extract_sub(ctx, a, coor_pairs, sub));

  tcapi::shape_t<TenT> expected = {2, 2, 2};
  TCAPICT_ASSERT(tcapi::shape(ctx, sub) == expected);

  // (1,0,0) in original maps to (0,0,0) in sub
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(tcapi::get_elem(ctx, sub, {0, 0, 0})), 42.0,
                       tol);
  // (2,1,1) in original maps to (1,1,1) in sub
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(tcapi::get_elem(ctx, sub, {1, 1, 1})), 13.0,
                       tol);
#else
  (void)fix;
#endif
}

// --- replace_sub (in-place) ---

template <typename TenT>
void test_replace_sub_inplace(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_REPLACE_SUB
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto a = tcapi::zeros<TenT>(ctx, {3, 4, 2});
  auto sub = tcapi::zeros<TenT>(ctx, {2, 2, 2});
  tcapi::set_elem(ctx, sub, {0, 0, 0}, make_elem<TenT>(42.0));
  tcapi::set_elem(ctx, sub, {1, 1, 1}, make_elem<TenT>(13.0));

  tcapi::elem_coors_t<TenT> begin_pt = {1, 2, 0};
  TCAPICT_ASSERT_NOTHROW(tcapi::replace_sub(ctx, a, sub, begin_pt));

  TCAPICT_ASSERT_CLOSE(real_part<TenT>(tcapi::get_elem(ctx, a, {1, 2, 0})), 42.0,
                       tol);
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(tcapi::get_elem(ctx, a, {2, 3, 1})), 13.0,
                       tol);
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(tcapi::get_elem(ctx, a, {0, 0, 0})), 0.0,
                       tol);
#else
  (void)fix;
#endif
}

// --- replace_sub (out-of-place) ---

template <typename TenT>
void test_replace_sub_outofplace(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_REPLACE_SUB
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto a = tcapi::zeros<TenT>(ctx, {4, 4});
  tcapi::set_elem(ctx, a, {0, 0}, make_elem<TenT>(99.0));

  auto sub = tcapi::fill<TenT>(ctx, {2, 2}, make_elem<TenT>(1.0));

  TenT result;
  tcapi::elem_coors_t<TenT> begin_pt = {1, 1};
  TCAPICT_ASSERT_NOTHROW(tcapi::replace_sub(ctx, a, sub, begin_pt, result));

  TCAPICT_ASSERT_CLOSE(real_part<TenT>(tcapi::get_elem(ctx, result, {1, 1})), 1.0,
                       tol);
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(tcapi::get_elem(ctx, result, {0, 0})), 99.0,
                       tol);
  // The input is read back at {1, 1}; see "Out-of-place inputs" in
  // CONTRIBUTING.md.
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(tcapi::get_elem(ctx, a, {1, 1})), 0.0, tol);
#else
  (void)fix;
#endif
}

// --- expand (in-place) ---

template <typename TenT> void test_expand_inplace(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_EXPAND
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto a = tcapi::zeros<TenT>(ctx, {2, 2, 2});

  tcapi::Map<tcapi::bond_idx_t<TenT>, tcapi::bond_dim_t<TenT>> bond_map = {{1, 2},
                                                                           {0, 1}};
  TCAPICT_ASSERT_NOTHROW(tcapi::expand(ctx, a, bond_map));

  tcapi::shape_t<TenT> expected = {3, 4, 2};
  TCAPICT_ASSERT(tcapi::shape(ctx, a) == expected);

  TCAPICT_ASSERT_CLOSE(real_part<TenT>(tcapi::get_elem(ctx, a, {2, 3, 0})), 0.0,
                       tol);
#else
  (void)fix;
#endif
}

// --- expand (out-of-place) ---

template <typename TenT>
void test_expand_outofplace(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_EXPAND
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto a = tcapi::zeros<TenT>(ctx, {2, 2, 2});
  tcapi::set_elem(ctx, a, {1, 1, 1}, make_elem<TenT>(5.0));

  tcapi::Map<tcapi::bond_idx_t<TenT>, tcapi::bond_dim_t<TenT>> bond_map = {{1, 2},
                                                                           {0, 1}};
  TenT expanded;
  TCAPICT_ASSERT_NOTHROW(tcapi::expand(ctx, a, bond_map, expanded));

  tcapi::shape_t<TenT> expected = {3, 4, 2};
  TCAPICT_ASSERT(tcapi::shape(ctx, expanded) == expected);
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(tcapi::get_elem(ctx, expanded, {1, 1, 1})),
                       5.0, tol);
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(tcapi::get_elem(ctx, expanded, {2, 3, 0})),
                       0.0, tol);
#else
  (void)fix;
#endif
}

#ifndef TCAPICT_SKIP_DIAG
// expect_vector_3, diag_input_value and diag_input_matrix are called by the
// diag tests only.

// Asserts that a first-order tensor of length 3 holds `expected` in order.
// `expected` is real, so for a complex element type the imaginary part is
// expected to be zero and is checked as such.
template <typename TenT>
void expect_vector_3(tcapi_test_fixture<TenT> &fix, const TenT &tensor,
                     const double (&expected)[3]) {
  auto &ctx = fix.context();
  TCAPICT_ASSERT(tcapi::order(ctx, tensor) == 1);
  TCAPICT_ASSERT(tcapi::size(ctx, tensor) == 3);
  // The values are exactly representable and reach the tensor by assignment
  // alone, through diag or set_elem, so they compare exactly.
  for (std::size_t i = 0; i < 3; ++i) {
    auto elem = tcapi::get_elem(ctx, tensor, coors_of<TenT>(i));
    TCAPICT_ASSERT(real_part<TenT>(elem) == expected[i]);
    if constexpr (is_complex_v<TenT>) {
      TCAPICT_ASSERT(imag_part<TenT>(elem) == 0.0);
    }
  }
}

// The element at (i, j) of the 3x3 input of the matrix-to-vector diag tests:
// `diagonal[i]` on the diagonal, which is distinct so that a reordered
// extraction does not pass, and the distinct non-zero 0.25 + 3i + j off it, so
// that neither does one that sums a row or a column. Every value is exact in
// single precision.
inline double diag_input_value(const double (&diagonal)[3], std::size_t i,
                               std::size_t j) {
  return i == j ? diagonal[i] : 0.25 + static_cast<double>(3 * i + j);
}

// A 3x3 matrix built with zeros and set_elem holding diag_input_value.
template <typename TenT>
TenT diag_input_matrix(tcapi_test_fixture<TenT> &fix,
                       const double (&diagonal)[3]) {
  auto &ctx = fix.context();
  auto matrix = tcapi::zeros<TenT>(ctx, {3, 3});
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      tcapi::set_elem(ctx, matrix, coors_of<TenT>(i, j),
                      make_elem<TenT>(diag_input_value(diagonal, i, j)));
    }
  }
  return matrix;
}

#endif

// --- diag: vector to matrix ---

template <typename TenT>
void test_diag_vec_to_mat(tcapi_test_fixture<TenT> &fix) {
#if !defined(TCAPICT_SKIP_DIAG) && defined(TCAPICT_HAS_DIAGONAL_CHECK)
  auto &ctx = fix.context();
  auto vector = tcapi::zeros<TenT>(ctx, {3});
  tcapi::set_elem(ctx, vector, {0}, make_elem<TenT>(1.0));
  tcapi::set_elem(ctx, vector, {1}, make_elem<TenT>(2.0));
  tcapi::set_elem(ctx, vector, {2}, make_elem<TenT>(3.0));

  tcapi::diag(ctx, vector);

  const double expected[3] = {1.0, 2.0, 3.0};
  TCAPICT_EXPECT_DIAGONAL(ctx, vector, expected,
                          tolerance(fix, tol_category::exact));
#else
  (void)fix;
#endif
}

// --- diag: matrix to vector ---

template <typename TenT>
void test_diag_mat_to_vec(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_DIAG
  auto &ctx = fix.context();
  const double expected[3] = {7.5, -8.5, 9.5};
  auto matrix = diag_input_matrix(fix, expected);

  TCAPICT_ASSERT_NOTHROW(tcapi::diag(ctx, matrix));

  expect_vector_3(fix, matrix, expected);
#else
  (void)fix;
#endif
}

// --- diag: matrix to vector, from a diagonal tensor ---

// Extracts distinct diagonal values from a diagonal tensor, the result of diag
// on a first-order input, of which V1 says "Implementations may store the
// result compactly when supported by the underlying TCF."
template <typename TenT>
void test_diag_mat_to_vec_from_diagonal(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_DIAG
  auto &ctx = fix.context();
  const double expected[3] = {7.5, -8.5, 9.5};

  auto vector = tcapi::zeros<TenT>(ctx, {3});
  for (std::size_t i = 0; i < 3; ++i) {
    tcapi::set_elem(ctx, vector, coors_of<TenT>(i), make_elem<TenT>(expected[i]));
  }
  TCAPICT_ASSERT_NOTHROW(tcapi::diag(ctx, vector));
  // The input is now second-order with nine elements, not the vector left as
  // it was.
  TCAPICT_ASSERT(tcapi::order(ctx, vector) == 2);
  TCAPICT_ASSERT(tcapi::size(ctx, vector) == 9);
  TCAPICT_ASSERT_NOTHROW(tcapi::diag(ctx, vector));

  expect_vector_3(fix, vector, expected);
#else
  (void)fix;
#endif
}

// --- diag: vector to matrix (out-of-place) ---

template <typename TenT>
void test_diag_vec_to_mat_outofplace(tcapi_test_fixture<TenT> &fix) {
#if !defined(TCAPICT_SKIP_DIAG) && defined(TCAPICT_HAS_DIAGONAL_CHECK)
  auto &ctx = fix.context();
  const double expected[3] = {1.5, -2.5, 3.5};

  auto vector = tcapi::zeros<TenT>(ctx, {3});
  for (std::size_t i = 0; i < 3; ++i) {
    tcapi::set_elem(ctx, vector, coors_of<TenT>(i), make_elem<TenT>(expected[i]));
  }

  TenT matrix;
  TCAPICT_ASSERT_NOTHROW(tcapi::diag(ctx, vector, matrix));

  TCAPICT_EXPECT_DIAGONAL(ctx, matrix, expected,
                          tolerance(fix, tol_category::exact));

  // Input unchanged; see "Out-of-place inputs" in CONTRIBUTING.md.
  expect_vector_3(fix, vector, expected);
#else
  (void)fix;
#endif
}

// --- diag: matrix to vector (out-of-place) ---

template <typename TenT>
void test_diag_mat_to_vec_outofplace(tcapi_test_fixture<TenT> &fix) {
#ifndef TCAPICT_SKIP_DIAG
  auto &ctx = fix.context();
  const double expected[3] = {4.5, -5.5, 6.5};
  auto matrix = diag_input_matrix(fix, expected);

  TenT vector;
  TCAPICT_ASSERT_NOTHROW(tcapi::diag(ctx, matrix, vector));

  expect_vector_3(fix, vector, expected);

  // Input unchanged, every element read back; see "Out-of-place inputs" in
  // CONTRIBUTING.md.
  TCAPICT_ASSERT(tcapi::order(ctx, matrix) == 2);
  TCAPICT_ASSERT(tcapi::size(ctx, matrix) == 9);
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      const auto elem = tcapi::get_elem(ctx, matrix, coors_of<TenT>(i, j));
      TCAPICT_ASSERT(real_part<TenT>(elem) == diag_input_value(expected, i, j));
      if constexpr (is_complex_v<TenT>) {
        TCAPICT_ASSERT(imag_part<TenT>(elem) == 0.0);
      }
    }
  }
#else
  (void)fix;
#endif
}

// --- stack: basic ---

template <typename TenT> void test_stack_basic(tcapi_test_fixture<TenT> &fix) {
#if !defined(TCAPICT_SKIP_STACK) && defined(TCAPICT_HAS_ELEM_SWEEP)
  auto &ctx = fix.context();
  // Distinct values across both inputs; the expected result is computed from
  // the same functions. See "Inputs to multi-input operations" in
  // CONTRIBUTING.md.
  auto t1_at = ramp_over<TenT>({2, 3}, 10.0);
  auto t2_at = ramp_over<TenT>({2, 3}, 20.0);
  auto t1 = tcapi::zeros<TenT>(ctx, {2, 3});
  auto t2 = tcapi::zeros<TenT>(ctx, {2, 3});
  TCAPICT_SET_ELEMS(fix, t1, t1_at);
  TCAPICT_SET_ELEMS(fix, t2, t2_at);

  // A named list; see "Inputs to multi-input operations" in CONTRIBUTING.md.
  tcapi::List<tcapi::CRef<TenT>> ins = {std::cref(t1), std::cref(t2)};
  auto result = TCAPICT_NOTHROW_VALUE(tcapi::stack<TenT>(ctx, ins, 0));

  // The new bond 0 selects the input: slice 0 is t1, slice 1 is t2
  TCAPICT_EXPECT_ELEMS(fix, result, {2, 2, 3}, [&](const auto &x) {
    const auto y = without_bond(x, 0);
    return x[0] == 0 ? t1_at(y) : t2_at(y);
  }, tol_category::exact);

  // An interior bond: the new bond 1 selects the input
  auto interior = TCAPICT_NOTHROW_VALUE(tcapi::stack<TenT>(ctx, ins, 1));
  TCAPICT_EXPECT_ELEMS(fix, interior, {2, 2, 3}, [&](const auto &x) {
    const auto y = without_bond(x, 1);
    return x[1] == 0 ? t1_at(y) : t2_at(y);
  }, tol_category::exact);

  // Inputs unchanged; see "Inputs to multi-input operations" in
  // CONTRIBUTING.md.
  TCAPICT_EXPECT_ELEMS(fix, t1, {2, 3}, t1_at, tol_category::exact);
  TCAPICT_EXPECT_ELEMS(fix, t2, {2, 3}, t2_at, tol_category::exact);
#else
  (void)fix;
#endif
}

// --- stack: last axis ---

template <typename TenT>
void test_stack_last_axis(tcapi_test_fixture<TenT> &fix) {
#if !defined(TCAPICT_SKIP_STACK) && defined(TCAPICT_HAS_ELEM_SWEEP)
  auto &ctx = fix.context();
  // Distinct values across all three inputs; the expected result is computed
  // from the same functions. See "Inputs to multi-input operations" in
  // CONTRIBUTING.md.
  auto t1_at = ramp_over<TenT>({2, 3}, 10.0);
  auto t2_at = ramp_over<TenT>({2, 3}, 20.0);
  auto t3_at = ramp_over<TenT>({2, 3}, 30.0);
  auto t1 = tcapi::zeros<TenT>(ctx, {2, 3});
  auto t2 = tcapi::zeros<TenT>(ctx, {2, 3});
  auto t3 = tcapi::zeros<TenT>(ctx, {2, 3});
  TCAPICT_SET_ELEMS(fix, t1, t1_at);
  TCAPICT_SET_ELEMS(fix, t2, t2_at);
  TCAPICT_SET_ELEMS(fix, t3, t3_at);

  auto result = TCAPICT_NOTHROW_VALUE(
      tcapi::stack<TenT>(ctx, {std::cref(t1), std::cref(t2), std::cref(t3)}, 2));

  // The new bond 2 selects the input
  TCAPICT_EXPECT_ELEMS(fix, result, {2, 3, 3}, [&](const auto &x) {
    const auto y = without_bond(x, 2);
    if (x[2] == 0) {
      return t1_at(y);
    }
    if (x[2] == 1) {
      return t2_at(y);
    }
    return t3_at(y);
  }, tol_category::exact);

  // Inputs unchanged; see "Inputs to multi-input operations" in
  // CONTRIBUTING.md.
  TCAPICT_EXPECT_ELEMS(fix, t1, {2, 3}, t1_at, tol_category::exact);
  TCAPICT_EXPECT_ELEMS(fix, t2, {2, 3}, t2_at, tol_category::exact);
  TCAPICT_EXPECT_ELEMS(fix, t3, {2, 3}, t3_at, tol_category::exact);
#else
  (void)fix;
#endif
}

} // namespace tests
} // namespace tcapict

// Bulk registration helpers: invoke X(..., "category", test_fn) once per test.
// See include/tcapict/adapters/doctest.h for usage.
//
// ALL_TYPES: safe for both real and complex TenT (imag-part assertions are
//   dual-guarded via `if constexpr (is_complex_v<TenT>)`).
#define TCAPICT_FOREACH_TENSOR_MANIPULATION_TEST_ALL_TYPES(X, ...) \
  X(__VA_ARGS__, "tensor_manipulation", test_shrink_inplace) \
  X(__VA_ARGS__, "tensor_manipulation", test_shrink_outofplace) \
  X(__VA_ARGS__, "tensor_manipulation", test_shrink_complex_values) \
  X(__VA_ARGS__, "tensor_manipulation", test_real_extraction) \
  X(__VA_ARGS__, "tensor_manipulation", test_imag_extraction) \
  X(__VA_ARGS__, "tensor_manipulation", test_real_zeroth_order) \
  X(__VA_ARGS__, "tensor_manipulation", test_imag_zeroth_order) \
  X(__VA_ARGS__, "tensor_manipulation", test_cplx_conj_inplace) \
  X(__VA_ARGS__, "tensor_manipulation", test_cplx_conj_outofplace) \
  X(__VA_ARGS__, "tensor_manipulation", test_cplx_conj_zeroth_order) \
  X(__VA_ARGS__, "tensor_manipulation", test_to_cplx_zeroth_order) \
  X(__VA_ARGS__, "tensor_manipulation", test_for_each_doubling) \
  X(__VA_ARGS__, "tensor_manipulation", test_for_each_summation) \
  X(__VA_ARGS__, "tensor_manipulation", test_for_each_capture) \
  X(__VA_ARGS__, "tensor_manipulation", test_for_each_const) \
  X(__VA_ARGS__, "tensor_manipulation", test_for_each_inversion) \
  X(__VA_ARGS__, "tensor_manipulation", test_for_each_zeroth_order) \
  X(__VA_ARGS__, "tensor_manipulation", test_for_each_with_coors) \
  X(__VA_ARGS__, "tensor_manipulation", test_for_each_with_coors_const) \
  X(__VA_ARGS__, "tensor_manipulation", test_for_each_with_coors_zeroth_order) \
  X(__VA_ARGS__, "tensor_manipulation", test_for_each_eye_traversal) \
  X(__VA_ARGS__, "tensor_manipulation", test_reshape) \
  X(__VA_ARGS__, "tensor_manipulation", test_reshape_zeroth_order) \
  X(__VA_ARGS__, "tensor_manipulation", test_transpose) \
  X(__VA_ARGS__, "tensor_manipulation", test_concatenate_basic) \
  X(__VA_ARGS__, "tensor_manipulation", test_concatenate_values) \
  X(__VA_ARGS__, "tensor_manipulation", test_extract_sub) \
  X(__VA_ARGS__, "tensor_manipulation", test_replace_sub_inplace) \
  X(__VA_ARGS__, "tensor_manipulation", test_replace_sub_outofplace) \
  X(__VA_ARGS__, "tensor_manipulation", test_expand_inplace) \
  X(__VA_ARGS__, "tensor_manipulation", test_expand_outofplace) \
  X(__VA_ARGS__, "tensor_manipulation", test_diag_vec_to_mat) \
  X(__VA_ARGS__, "tensor_manipulation", test_diag_mat_to_vec) \
  X(__VA_ARGS__, "tensor_manipulation", test_diag_mat_to_vec_from_diagonal) \
  X(__VA_ARGS__, "tensor_manipulation", test_diag_vec_to_mat_outofplace) \
  X(__VA_ARGS__, "tensor_manipulation", test_diag_mat_to_vec_outofplace) \
  X(__VA_ARGS__, "tensor_manipulation", test_stack_basic) \
  X(__VA_ARGS__, "tensor_manipulation", test_stack_last_axis)

// REAL_ONLY: the test lifts a real tensor to complex and checks the zero
//   imaginary parts it gains, so it takes only a real TenT.
#define TCAPICT_FOREACH_TENSOR_MANIPULATION_TEST_REAL_ONLY(X, ...) \
  X(__VA_ARGS__, "tensor_manipulation", test_to_cplx_outofplace)

// CPLX_ONLY: body is wrapped in `if constexpr (is_complex_v<TenT>)`; running
//   for real TenT would be a no-op, so skip registration.
#define TCAPICT_FOREACH_TENSOR_MANIPULATION_TEST_CPLX_ONLY(X, ...) \
  X(__VA_ARGS__, "tensor_manipulation", test_to_cplx_complex_to_complex)
