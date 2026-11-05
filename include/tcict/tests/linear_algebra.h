#pragma once

#include <tcict/assertion.h>
#include <tcict/diagonal.h>
#include <tcict/elem_helper.h>
#include <tcict/elem_sweep.h>
#include <tcict/fixture.h>
#include <tcict/skip.h>
#include <tcict/zeroth_order.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <exception>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace tcict {
namespace tests {

// --- norm (basic: identity matrix) ---

template <typename TenT> void test_norm_identity(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_NORM
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::reduction, 9);
  auto identity = tci::eye<TenT>(ctx, 3);

  auto norm_val = tci::norm(ctx, identity);
  // Frobenius norm of 3x3 identity = sqrt(3)
  TCICT_ASSERT_CLOSE(norm_val, std::sqrt(3.0), tol);
#else
  (void)fix;
#endif
}

// --- linear_combine: uniform coefficients ---

template <typename TenT>
void test_linear_combine_uniform(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_LINEAR_COMBINE) && defined(TCICT_HAS_ELEM_SWEEP)
  auto &ctx = fix.context();

  // Real parts a = [[1,2],[3,4]], b = [[5,6],[7,8]], c = [[20,21],[22,23]]:
  // distinct values across all three inputs; the expected result is computed
  // from the same functions. See "Inputs to multi-input operations" in
  // CONTRIBUTING.md. The offsets are not evenly spaced: evenly spaced ones
  // would make a + c == 2b, so a backend returning 3b would pass.
  auto a_at = ramp_over<TenT>({2, 2}, 1.0);
  auto b_at = ramp_over<TenT>({2, 2}, 5.0);
  auto c_at = ramp_over<TenT>({2, 2}, 20.0);
  auto tensor_a = tci::zeros<TenT>(ctx, {2, 2});
  auto tensor_b = tci::zeros<TenT>(ctx, {2, 2});
  auto tensor_c = tci::zeros<TenT>(ctx, {2, 2});
  TCICT_SET_ELEMS(fix, tensor_a, a_at);
  TCICT_SET_ELEMS(fix, tensor_b, b_at);
  TCICT_SET_ELEMS(fix, tensor_c, c_at);

  auto result = TCICT_NOTHROW_VALUE(tci::linear_combine<TenT>(
      ctx, {std::cref(tensor_a), std::cref(tensor_b), std::cref(tensor_c)}));

  // Expected real parts: a + b + c = [[26,29],[32,35]]
  TCICT_EXPECT_ELEMS(
      fix, result, {2, 2},
      [&](const auto &x) { return a_at(x) + b_at(x) + c_at(x); },
      tol_category::reduction, 3);

  // Inputs unchanged; see "Inputs to multi-input operations" in
  // CONTRIBUTING.md.
  TCICT_EXPECT_ELEMS(fix, tensor_a, {2, 2}, a_at, tol_category::exact);
  TCICT_EXPECT_ELEMS(fix, tensor_b, {2, 2}, b_at, tol_category::exact);
  TCICT_EXPECT_ELEMS(fix, tensor_c, {2, 2}, c_at, tol_category::exact);
#else
  (void)fix;
#endif
}

// --- linear_combine: weighted coefficients ---

template <typename TenT>
void test_linear_combine_weighted(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_LINEAR_COMBINE) && defined(TCICT_HAS_ELEM_SWEEP)
  auto &ctx = fix.context();

  // Real parts a = [[1,2],[3,4]], b = [[5,6],[7,8]]: distinct values across
  // both inputs; the expected result is computed from the same functions. See
  // "Inputs to multi-input operations" in CONTRIBUTING.md.
  auto a_at = ramp_over<TenT>({2, 2}, 1.0);
  auto b_at = ramp_over<TenT>({2, 2}, 5.0);
  auto tensor_a = tci::zeros<TenT>(ctx, {2, 2});
  auto tensor_b = tci::zeros<TenT>(ctx, {2, 2});
  TCICT_SET_ELEMS(fix, tensor_a, a_at);
  TCICT_SET_ELEMS(fix, tensor_b, b_at);

  // A named list; see "Inputs to multi-input operations" in CONTRIBUTING.md.
  tci::List<tci::CRef<TenT>> ins = {std::cref(tensor_a), std::cref(tensor_b)};
  // For a complex element type the coefficients have imaginary parts too, so
  // a backend that conjugates them or drops their imaginary parts fails.
  const std::complex<double> s_a(0.5, is_complex_v<TenT> ? 0.25 : 0.0);
  const std::complex<double> s_b(2.0, is_complex_v<TenT> ? -1.0 : 0.0);
  tci::List<tci::elem_t<TenT>> coefficients = {
      make_elem<TenT>(s_a.real(), s_a.imag()),
      make_elem<TenT>(s_b.real(), s_b.imag())};

  auto result =
      TCICT_NOTHROW_VALUE(tci::linear_combine<TenT>(ctx, ins, coefficients));

  // Expected: s_a*a + s_b*b; for a real element type, real parts
  // [[10.5,13],[15.5,18]]
  TCICT_EXPECT_ELEMS(
      fix, result, {2, 2},
      [&](const auto &x) {
        return s_a * written<TenT>(a_at(x)) + s_b * written<TenT>(b_at(x));
      },
      tol_category::reduction, 2);

  // Inputs unchanged; see "Inputs to multi-input operations" in
  // CONTRIBUTING.md.
  TCICT_EXPECT_ELEMS(fix, tensor_a, {2, 2}, a_at, tol_category::exact);
  TCICT_EXPECT_ELEMS(fix, tensor_b, {2, 2}, b_at, tol_category::exact);
#else
  (void)fix;
#endif
}

// --- linear_combine: single tensor ---

template <typename TenT>
void test_linear_combine_single(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_LINEAR_COMBINE) && defined(TCICT_HAS_ELEM_SWEEP)
  auto &ctx = fix.context();

  auto single_at = ramp_over<TenT>({1, 1}, 5.0);
  auto single_tensor = tci::zeros<TenT>(ctx, {1, 1});
  TCICT_SET_ELEMS(fix, single_tensor, single_at);

  // A named list, feeding both overloads; see "Inputs to multi-input
  // operations" in CONTRIBUTING.md.
  tci::List<tci::CRef<TenT>> single_list = {std::cref(single_tensor)};
  auto summed =
      TCICT_NOTHROW_VALUE(tci::linear_combine<TenT>(ctx, single_list));
  TCICT_EXPECT_ELEMS(fix, summed, {1, 1}, single_at, tol_category::exact);

  // For a complex element type the coefficient has an imaginary part too.
  // An input element is v(1 - 0.5i), and (3 - 1.5i) v (1 - 0.5i) =
  // v (2.25 - 3i) keeps an imaginary part in the result.
  const std::complex<double> s(3.0, is_complex_v<TenT> ? -1.5 : 0.0);
  tci::List<tci::elem_t<TenT>> single_coef = {make_elem<TenT>(s.real(),
                                                              s.imag())};
  auto scaled = TCICT_NOTHROW_VALUE(
      tci::linear_combine<TenT>(ctx, single_list, single_coef));
  TCICT_EXPECT_ELEMS(fix, scaled, {1, 1}, [&](const auto &x) {
    return s * written<TenT>(single_at(x));
  });

  // Input unchanged; see "Inputs to multi-input operations" in
  // CONTRIBUTING.md.
  TCICT_EXPECT_ELEMS(fix, single_tensor, {1, 1}, single_at, tol_category::exact);
#else
  (void)fix;
#endif
}

// --- linear_combine: zeroth-order inputs ---

// Both overloads on two zeroth-order inputs; the result is zeroth-order. See
// "Inputs to multi-input operations" in CONTRIBUTING.md for the named list, the
// distinct input values and the read-back.
template <typename TenT>
void test_linear_combine_zeroth_order(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_LINEAR_COMBINE) && defined(TCICT_HAS_ZEROTH_ORDER_SWEEP)
  auto &ctx = fix.context();
  auto tensor_a = tci::zeros<TenT>(ctx, {});
  auto tensor_b = tci::zeros<TenT>(ctx, {});
  TCICT_SET_ZEROTH_ORDER(fix, tensor_a, 1.25);
  TCICT_SET_ZEROTH_ORDER(fix, tensor_b, -3.5);

  tci::List<tci::CRef<TenT>> ins = {std::cref(tensor_a), std::cref(tensor_b)};
  auto sum = TCICT_NOTHROW_VALUE(tci::linear_combine<TenT>(ctx, ins));
  TCICT_EXPECT_ZEROTH_ORDER(fix, sum, written<TenT>(1.25) + written<TenT>(-3.5),
                            tol_category::reduction, 2);

  const std::complex<double> s_a(0.5, is_complex_v<TenT> ? 0.25 : 0.0);
  const std::complex<double> s_b(2.0, is_complex_v<TenT> ? -1.0 : 0.0);
  tci::List<tci::elem_t<TenT>> coefficients = {
      make_elem<TenT>(s_a.real(), s_a.imag()),
      make_elem<TenT>(s_b.real(), s_b.imag())};
  auto weighted =
      TCICT_NOTHROW_VALUE(tci::linear_combine<TenT>(ctx, ins, coefficients));
  TCICT_EXPECT_ZEROTH_ORDER(
      fix, weighted, s_a * written<TenT>(1.25) + s_b * written<TenT>(-3.5),
      tol_category::reduction, 2);

  TCICT_EXPECT_ZEROTH_ORDER(fix, tensor_a, 1.25);
  TCICT_EXPECT_ZEROTH_ORDER(fix, tensor_b, -3.5);
#else
  (void)fix;
#endif
}

// --- normalize: in-place ---

template <typename TenT>
void test_normalize_inplace(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_NORMALIZE
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::reduction, 4);

  auto tensor = tci::zeros<TenT>(ctx, {2, 2});
  // [[3,4],[0,0]] -> norm = 5
  tci::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(3.0));
  tci::set_elem(ctx, tensor, {0, 1}, make_elem<TenT>(4.0));

  auto original_norm = tci::normalize(ctx, tensor);
  TCICT_ASSERT_CLOSE(std::abs(original_norm), 5.0, tol);

  // Verify normalized values
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 0})), 0.6,
                     tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 1})), 0.8,
                     tol);

  // New norm should be 1
  auto new_norm = tci::norm(ctx, tensor);
  TCICT_ASSERT_CLOSE(new_norm, 1.0, tol);
#else
  (void)fix;
#endif
}

// --- normalize: out-of-place ---

template <typename TenT>
void test_normalize_outofplace(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_NORMALIZE
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::reduction, 3);

  auto original = tci::zeros<TenT>(ctx, {3, 1});
  // [[2],[2],[1]] -> norm = 3
  tci::set_elem(ctx, original, {0, 0}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, original, {1, 0}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, original, {2, 0}, make_elem<TenT>(1.0));

  TenT normalized;
  auto original_norm = tci::normalize(ctx, original, normalized);
  TCICT_ASSERT_CLOSE(std::abs(original_norm), 3.0, tol);

  // Input unchanged; see "Out-of-place inputs" in CONTRIBUTING.md.
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, original, {0, 0})), 2.0,
                     tol);

  // Normalized tensor
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, normalized, {0, 0})),
                     2.0 / 3.0, tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, normalized, {2, 0})),
                     1.0 / 3.0, tol);

  auto new_norm = tci::norm(ctx, normalized);
  TCICT_ASSERT_CLOSE(new_norm, 1.0, tol);
#else
  (void)fix;
#endif
}

// --- normalize: edge cases ---

template <typename TenT>
void test_normalize_edge_cases(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_NORMALIZE
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::reduction, 4);

  // Single non-zero element
  auto single_elem = tci::zeros<TenT>(ctx, {2, 2});
  tci::set_elem(ctx, single_elem, {1, 1}, make_elem<TenT>(7.0));

  auto norm1 = tci::normalize(ctx, single_elem);
  TCICT_ASSERT_CLOSE(std::abs(norm1), 7.0, tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, single_elem, {1, 1})),
                     1.0, tol);

  // Zero tensor
  auto zero_tensor = tci::zeros<TenT>(ctx, {2, 2});
  auto norm_zero = tci::normalize(ctx, zero_tensor);
  TCICT_ASSERT_CLOSE(std::abs(norm_zero), 0.0, tol);
#else
  (void)fix;
#endif
}

// --- normalize: zeroth-order ---

// The original norm of one element v is |v|, and normalizing leaves v / |v|.
template <typename TenT>
void test_normalize_zeroth_order(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_NORMALIZE) && defined(TCICT_HAS_ZEROTH_ORDER_SWEEP)
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::reduction);

  auto tensor = tci::zeros<TenT>(ctx, {});
  TCICT_SET_ZEROTH_ORDER(fix, tensor, -2.0);
  const auto w = written<TenT>(-2.0);
  const auto original = tci::normalize(ctx, tensor);
  TCICT_ASSERT_CLOSE(original, std::abs(w), tol);
  TCICT_EXPECT_ZEROTH_ORDER(fix, tensor, w / std::abs(w), tol_category::reduction);

  // Out of place; the input is read back, see "Out-of-place inputs" in
  // CONTRIBUTING.md.
  auto input = tci::zeros<TenT>(ctx, {});
  TCICT_SET_ZEROTH_ORDER(fix, input, 3.0);
  const auto w_in = written<TenT>(3.0);
  TenT output;
  const auto original_in = tci::normalize(ctx, input, output);
  TCICT_ASSERT_CLOSE(original_in, std::abs(w_in), tol);
  TCICT_EXPECT_ZEROTH_ORDER(fix, output, w_in / std::abs(w_in),
                            tol_category::reduction);
  TCICT_EXPECT_ZEROTH_ORDER(fix, input, 3.0);
#else
  (void)fix;
#endif
}

// --- norm: 2x2 identity ---

template <typename TenT> void test_norm_2x2(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_NORM
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::reduction, 4);
  auto identity = tci::eye<TenT>(ctx, 2);
  auto norm_val = tci::norm(ctx, identity);
  TCICT_ASSERT_CLOSE(norm_val, std::sqrt(2.0), tol);
#else
  (void)fix;
#endif
}

// --- norm: zeroth-order ---

// The Frobenius norm of one element v is |v|.
template <typename TenT>
void test_norm_zeroth_order(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_NORM) && defined(TCICT_HAS_ELEM_SWEEP)
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::reduction);
  auto tensor = tci::zeros<TenT>(ctx, {});
  TCICT_SET_ZEROTH_ORDER(fix, tensor, -2.0);
  TCICT_ASSERT_CLOSE(tci::norm(ctx, tensor), std::abs(written<TenT>(-2.0)), tol);
#else
  (void)fix;
#endif
}

// --- contract: matrix multiplication via Einstein notation ---

template <typename TenT>
void test_contract_matmul(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_CONTRACT
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::reduction, 2);
  auto a = tci::zeros<TenT>(ctx, {2, 2});
  auto b = tci::zeros<TenT>(ctx, {2, 2});

  // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
  tci::set_elem(ctx, a, {0, 0}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, a, {0, 1}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, a, {1, 0}, make_elem<TenT>(3.0));
  tci::set_elem(ctx, a, {1, 1}, make_elem<TenT>(4.0));
  tci::set_elem(ctx, b, {0, 0}, make_elem<TenT>(5.0));
  tci::set_elem(ctx, b, {0, 1}, make_elem<TenT>(6.0));
  tci::set_elem(ctx, b, {1, 0}, make_elem<TenT>(7.0));
  tci::set_elem(ctx, b, {1, 1}, make_elem<TenT>(8.0));

  TenT c;
  tci::contract(ctx, a, "ij", b, "jk", c, "ik");

  // A*B = [[19,22],[43,50]]
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, c, {0, 0})), 19.0, tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, c, {0, 1})), 22.0, tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, c, {1, 0})), 43.0, tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, c, {1, 1})), 50.0, tol);
#else
  (void)fix;
#endif
}

// --- contract: dot product ---

template <typename TenT>
void test_contract_dot_product(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_CONTRACT) && defined(TCICT_HAS_ZEROTH_ORDER_SWEEP)
  auto &ctx = fix.context();
  auto a = tci::zeros<TenT>(ctx, {3});
  auto b = tci::zeros<TenT>(ctx, {3});

  // a = [1,2,3], b = [4,5,6]
  tci::set_elem(ctx, a, {0}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, a, {1}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, a, {2}, make_elem<TenT>(3.0));
  tci::set_elem(ctx, b, {0}, make_elem<TenT>(4.0));
  tci::set_elem(ctx, b, {1}, make_elem<TenT>(5.0));
  tci::set_elem(ctx, b, {2}, make_elem<TenT>(6.0));

  TenT c;
  tci::contract(ctx, a, "i", b, "i", c, "");

  // TCAPI V1: a contract with an empty output label list yields a zeroth-order
  // tensor (order 0, shape {}, size 1), read at the empty coordinate {}.
  // dot = 1*4 + 2*5 + 3*6 = 32
  TCICT_EXPECT_ZEROTH_ORDER(fix, c, std::complex<double>(32.0, 0.0),
                            tol_category::reduction, 3);
#else
  (void)fix;
#endif
}

// --- contract: dot product, integer labels ---

// The same full contraction through the integer-label overload. The inputs
// carry imaginary parts for a complex element type, so a backend conjugating
// either operand fails.
template <typename TenT>
void test_contract_dot_product_int_labels(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_CONTRACT) && defined(TCICT_HAS_ZEROTH_ORDER_SWEEP)
  auto &ctx = fix.context();
  auto a_at = ramp_over<TenT>({3}, 1.0);
  auto b_at = ramp_over<TenT>({3}, 4.0);
  auto a = tci::zeros<TenT>(ctx, {3});
  auto b = tci::zeros<TenT>(ctx, {3});
  TCICT_SET_ELEMS(fix, a, a_at);
  TCICT_SET_ELEMS(fix, b, b_at);

  const tci::List<tci::bond_label_t<TenT>> labels_a = {7};
  const tci::List<tci::bond_label_t<TenT>> labels_b = {7};
  const tci::List<tci::bond_label_t<TenT>> labels_c = {};
  TenT c;
  tci::contract(ctx, a, labels_a, b, labels_b, c, labels_c);

  std::complex<double> dot = 0.0;
  for (std::size_t i = 0; i < 3; ++i) {
    const auto coors = coors_of<TenT>(i);
    dot += written<TenT>(a_at(coors)) * written<TenT>(b_at(coors));
  }
  TCICT_EXPECT_ZEROTH_ORDER(fix, c, dot, tol_category::reduction, 3);
#else
  (void)fix;
#endif
}

// --- contract: zeroth-order operands ---

// A zeroth-order operand has no labels, so nothing is contracted against it:
// against a vector the result is the vector scaled by its element, on either
// side, and against another zeroth-order operand with no output labels it is
// the zeroth-order product.
template <typename TenT>
void test_contract_zeroth_order(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_CONTRACT) && defined(TCICT_HAS_ZEROTH_ORDER_SWEEP)
  auto &ctx = fix.context();
  auto v_at = ramp_over<TenT>({3}, 1.0);
  auto s = tci::zeros<TenT>(ctx, {});
  auto t = tci::zeros<TenT>(ctx, {});
  auto v = tci::zeros<TenT>(ctx, {3});
  TCICT_SET_ZEROTH_ORDER(fix, s, 1.5);
  TCICT_SET_ZEROTH_ORDER(fix, t, -2.0);
  TCICT_SET_ELEMS(fix, v, v_at);
  auto scaled = [&](const tci::elem_coors_t<TenT> &x) {
    return written<TenT>(1.5) * written<TenT>(v_at(x));
  };

  TenT sv;
  tci::contract(ctx, s, "", v, "i", sv, "i");
  TCICT_EXPECT_ELEMS(fix, sv, {3}, scaled, tol_category::reduction);

  TenT vs;
  tci::contract(ctx, v, "i", s, "", vs, "i");
  TCICT_EXPECT_ELEMS(fix, vs, {3}, scaled, tol_category::reduction);

  TenT st;
  tci::contract(ctx, s, "", t, "", st, "");
  TCICT_EXPECT_ZEROTH_ORDER(fix, st, written<TenT>(1.5) * written<TenT>(-2.0),
                            tol_category::reduction);
#else
  (void)fix;
#endif
}

// --- contract: outer product ---

template <typename TenT>
void test_contract_outer_product(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_CONTRACT
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto a = tci::zeros<TenT>(ctx, {2});
  auto b = tci::zeros<TenT>(ctx, {3});

  tci::set_elem(ctx, a, {0}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, a, {1}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, b, {0}, make_elem<TenT>(3.0));
  tci::set_elem(ctx, b, {1}, make_elem<TenT>(4.0));
  tci::set_elem(ctx, b, {2}, make_elem<TenT>(5.0));

  TenT c;
  tci::contract(ctx, a, "i", b, "j", c, "ij");

  auto c_shape = tci::shape(ctx, c);
  TCICT_ASSERT(c_shape[0] == 2);
  TCICT_ASSERT(c_shape[1] == 3);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, c, {0, 0})), 3.0, tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, c, {1, 2})), 10.0, tol);
#else
  (void)fix;
#endif
}

// --- QR decomposition ---

template <typename TenT> void test_qr(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_QR
#ifdef TCICT_SKIP_QR_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  auto matrix = tci::zeros<TenT>(ctx, {3, 3});
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      tci::set_elem(ctx, matrix,
                    {static_cast<tci::elem_coor_t<TenT>>(i),
                     static_cast<tci::elem_coor_t<TenT>>(j)},
                    make_elem<TenT>(i * 3 + j + 1));

  TenT q, r;
  tci::qr(ctx, matrix, 1, q, r);
  TCICT_ASSERT(tci::shape(ctx, q).size() == 2);
  TCICT_ASSERT(tci::shape(ctx, r).size() == 2);

  // Verify Q * R ≈ A (reconstruction)
  TenT reconstructed;
  tci::contract(ctx, q, "ij", r, "jk", reconstructed, "ik");
  TCICT_ASSERT(tci::close(ctx, reconstructed, matrix,
                          tolerance(fix, tol_category::factorization)));

  // Verify Q†Q ≈ I (orthogonality)
  TenT q_dag;
  tci::cplx_conj(ctx, q, q_dag);
  TenT q_dag_t;
  tci::transpose(ctx, q_dag, {1, 0}, q_dag_t);
  TenT qtq;
  tci::contract(ctx, q_dag_t, "ij", q, "jk", qtq, "ik");
  auto bond_dim = tci::shape(ctx, q)[1];
  auto identity = tci::eye<TenT>(ctx, bond_dim);
  TCICT_ASSERT(tci::close(ctx, qtq, identity,
                          tolerance(fix, tol_category::factorization)));
#else
  (void)fix;
#endif
}

// --- LQ decomposition ---

template <typename TenT> void test_lq(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_LQ
#ifdef TCICT_SKIP_LQ_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  auto matrix = tci::zeros<TenT>(ctx, {3, 3});
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      tci::set_elem(ctx, matrix,
                    {static_cast<tci::elem_coor_t<TenT>>(i),
                     static_cast<tci::elem_coor_t<TenT>>(j)},
                    make_elem<TenT>(i * 3 + j + 1));

  TenT l, q;
  tci::lq(ctx, matrix, 1, l, q);
  TCICT_ASSERT(tci::shape(ctx, l).size() == 2);
  TCICT_ASSERT(tci::shape(ctx, q).size() == 2);

  // Verify L * Q ≈ A (reconstruction)
  TenT reconstructed;
  tci::contract(ctx, l, "ij", q, "jk", reconstructed, "ik");
  TCICT_ASSERT(tci::close(ctx, reconstructed, matrix,
                          tolerance(fix, tol_category::factorization)));

  // Verify QQ† ≈ I (orthogonality)
  TenT q_dag;
  tci::cplx_conj(ctx, q, q_dag);
  TenT q_dag_t;
  tci::transpose(ctx, q_dag, {1, 0}, q_dag_t);
  TenT qqt;
  tci::contract(ctx, q, "ij", q_dag_t, "jk", qqt, "ik");
  auto bond_dim = tci::shape(ctx, q)[0];
  auto identity = tci::eye<TenT>(ctx, bond_dim);
  TCICT_ASSERT(tci::close(ctx, qqt, identity,
                          tolerance(fix, tol_category::factorization)));
#else
  (void)fix;
#endif
}

// --- truncated SVD ---
//
// V1 calls sigma "a **real second-order diagonal tensor** of the retained
// singular values with shape `{chi, chi}`". Unlike its svd entry, it does not
// say in which order the diagonal holds them, so these tests compare sigma's
// diagonal as a multiset, through diag_order::any.

#if !defined(TCICT_SKIP_TRUNC_SVD) && defined(TCICT_HAS_DIAGONAL_CHECK)
// The singular values the trunc_svd fixture matrix is built from. Shared with
// trunc_svd_expected_epsilon below so the matrix and the expected values cannot
// drift apart.
inline constexpr double kTruncSvdFixtureSvs[] = {3.0, 2.0, 1.0, 0.1};
inline constexpr int kTruncSvdFixtureRank =
    static_cast<int>(sizeof(kTruncSvdFixtureSvs) / sizeof(kTruncSvdFixtureSvs[0]));

// The tests take the singular values trunc_svd retains on the fixture matrix
// to be a prefix of the table, which holds only while the table is
// non-increasing and non-negative.
constexpr bool trunc_svd_fixture_is_non_increasing_and_non_negative() {
  for (int i = 0; i < kTruncSvdFixtureRank; ++i) {
    if (kTruncSvdFixtureSvs[i] < 0.0 ||
        (i > 0 && kTruncSvdFixtureSvs[i] > kTruncSvdFixtureSvs[i - 1])) {
      return false;
    }
  }
  return true;
}
static_assert(trunc_svd_fixture_is_non_increasing_and_non_negative(),
              "kTruncSvdFixtureSvs must be non-increasing and non-negative");

// Helper: the singular values trunc_svd retains at `chi` on the fixture matrix
// built with `scale`, the first `chi` entries of the table, scaled.
inline std::vector<double> trunc_svd_fixture_svs(int chi, double scale = 1.0) {
  TCICT_ASSERT(chi >= 1 && chi <= kTruncSvdFixtureRank);
  TCICT_ASSERT(scale > 0.0);
  std::vector<double> svs(kTruncSvdFixtureSvs, kTruncSvdFixtureSvs + chi);
  for (auto &s : svs) {
    s *= scale;
  }
  return svs;
}

// Helper: build a diagonal matrix whose singular values are the fixture
// spectrum scaled by `scale`.
// Only defined when TRUNC_SVD tests are active so partial backends without
// tci::zeros / tci::set_elem do not need to declare them.
//
// `scale` multiplies every singular value uniformly. The relative truncation
// error epsilon is a ratio of sums of squares, so it is invariant under that
// scaling; a test can therefore vary `scale` to separate an implementation
// that thresholds on epsilon from one that thresholds on the raw singular
// values.
template <typename TenT>
TenT trunc_svd_test_matrix(typename tci::tensor_traits<TenT>::context_handle_t &ctx,
                           double scale = 1.0) {
  // elem_coor_t is backend-defined and may be unsigned, and a braced
  // initializer rejects a narrowing conversion unless the argument is a
  // constant expression. kTruncSvdFixtureRank is a constant expression, so the
  // shape below needs no cast; the loop counter is not, so the coordinates do.
  auto matrix = tci::zeros<TenT>(ctx, {kTruncSvdFixtureRank, kTruncSvdFixtureRank});
  for (int i = 0; i < kTruncSvdFixtureRank; ++i) {
    const auto coor = static_cast<tci::elem_coor_t<TenT>>(i);
    tci::set_elem(ctx, matrix, {coor, coor},
                  make_elem<TenT>(kTruncSvdFixtureSvs[i] * scale));
  }
  return matrix;
}

// Helper: the spec's relative truncation error for keeping `chi` of the
// singular values in `svs`,
//
//   epsilon(chi) = sum_{i>=chi} s_i^2 / sum_{i<kappa} s_i^2.
//
// Evaluated here from the spectrum rather than by calling back into the backend,
// so an assertion using it compares the implementation against the specification
// and not against a second route through the same code. Takes no scale: epsilon
// is a ratio of sums of squares and so is scale-invariant.
template <typename TenT>
tci::real_t<TenT> trunc_svd_expected_epsilon_from(const double *svs, int rank, int chi) {
  TCICT_ASSERT(chi >= 0 && chi <= rank);
  double total = 0.0;
  double discarded = 0.0;
  for (int i = 0; i < rank; ++i) {
    const double s2 = svs[i] * svs[i];
    total += s2;
    if (i >= chi) {
      discarded += s2;
    }
  }
  return static_cast<tci::real_t<TenT>>(discarded / total);
}

// Helper: the above bound to the shared [3, 2, 1, 0.1] fixture spectrum, so the
// matrix builder and the expected values cannot drift apart.
template <typename TenT> tci::real_t<TenT> trunc_svd_expected_epsilon(int chi) {
  return trunc_svd_expected_epsilon_from<TenT>(kTruncSvdFixtureSvs, kTruncSvdFixtureRank,
                                               chi);
}

// Helper: the suite's one call of overload (2), whose trailing parameters V1
// declares as
//
//   const tci::bond_dim_t<TenT> chi_max,
//   const tci::real_t<TenT> s_min,
//   const tci::bond_dim_t<TenT> chi_min,
//   const tci::real_t<TenT> target_trunc_err
//
// A call with the two integer arguments and the two real ones in other
// positions still compiles, each converting to the parameter it lands on, so
// only the values a test checks can show the order is wrong. Every overload
// (2) test calls through here, and test_trunc_svd_argument_order checks values
// that differ when this call passes them as chi_min, chi_max,
// target_trunc_err, s_min. The two real arguments are taken as
// tci::real_t<TenT>, so a value the backend reported, such as a measured
// truncation error, reaches it again at its own precision.
template <typename TenT>
void trunc_svd_by_strategy(tci::context_handle_t<TenT> &ctx, const TenT &a,
                           int num_of_bds_as_row, TenT &u, tci::real_ten_t<TenT> &sigma,
                           TenT &v_dag, tci::real_t<TenT> &trunc_err, int chi_max,
                           tci::real_t<TenT> s_min, int chi_min,
                           tci::real_t<TenT> target_trunc_err) {
  tci::trunc_svd(ctx, a, static_cast<tci::order_t<TenT>>(num_of_bds_as_row), u, sigma, v_dag,
                 trunc_err, static_cast<tci::bond_dim_t<TenT>>(chi_max), s_min,
                 static_cast<tci::bond_dim_t<TenT>>(chi_min), target_trunc_err);
}

// Helper: run overload (2) on the fixture spectrum and assert sigma, which
// holds the retained singular values, and the truncation error against the
// spec's epsilon for the expected chi.
//
// The strategy tests below differ only in the four strategy arguments, taken
// here in overload (2)'s order, the fixture scale, and the expected chi, so the
// call and its assertions live here once. They stay separate test functions:
// the registration macro expands one consumer test case per function, and
// merging them would collapse the per-case reporting and skip granularity that
// buys.
//
// Guarded with the tests it serves, and for the same reason the fixture builder
// above is: it calls both of them, and the diagonal check, so leaving it outside
// would break the header for a backend that defines TCICT_SKIP_TRUNC_SVD or
// skips an API that TCICT_HAS_DIAGONAL_CHECK requires.
template <typename TenT>
void trunc_svd_expect_retained_chi(tci_test_fixture<TenT> &fix, double scale, int chi_max,
                                   tci::real_t<TenT> s_min, int chi_min,
                                   tci::real_t<TenT> target_trunc_err, int expected_chi) {
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::factorization);
  auto matrix = trunc_svd_test_matrix<TenT>(ctx, scale);

  TenT u, v_dag;
  tci::real_ten_t<TenT> s_diag;
  tci::real_t<TenT> trunc_err = -1.0;

  trunc_svd_by_strategy(ctx, matrix, 1, u, s_diag, v_dag, trunc_err, chi_max, s_min, chi_min,
                        target_trunc_err);

  // The tolerance is absolute, so it scales with the values it bounds; epsilon,
  // a ratio, keeps the unscaled one.
  TCICT_EXPECT_DIAGONAL(ctx, s_diag, trunc_svd_fixture_svs(expected_chi, scale),
                        tol * scale, diag_order::any);
  TCICT_ASSERT_CLOSE(trunc_err, trunc_svd_expected_epsilon<TenT>(expected_chi), tol);
}
#endif

template <typename TenT> void test_trunc_svd(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_TRUNC_SVD) && defined(TCICT_HAS_DIAGONAL_CHECK)
#ifdef TCICT_SKIP_TRUNC_SVD_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  auto matrix = trunc_svd_test_matrix<TenT>(ctx);

  TenT u, v_dag;
  tci::real_ten_t<TenT> s_diag;
  tci::real_t<TenT> trunc_err;

  TCICT_ASSERT_NOTHROW(
      tci::trunc_svd(ctx, matrix, 1, u, s_diag, v_dag, trunc_err,
                     static_cast<tci::bond_dim_t<TenT>>(2), 0.5));

  // s_min = 0.5 discards 0.1, and chi_max = 2 keeps 3 and 2 of the rest.
  TCICT_EXPECT_DIAGONAL(ctx, s_diag, trunc_svd_fixture_svs(2),
                        tolerance(fix, tol_category::factorization),
                        diag_order::any);
  TCICT_ASSERT(trunc_err >= 0.0);
#else
  (void)fix;
#endif
}

/// Verify trunc_err equals the spec formula when truncation occurs.
/// SVs = [3, 2, 1, 0.1], chi_max = 2 → keep [3, 2], discard [1, 0.1].
/// epsilon = (1^2 + 0.1^2) / (3^2 + 2^2 + 1^2 + 0.1^2)
///         = 1.01 / 14.01 ≈ 0.07209
template <typename TenT>
void test_trunc_svd_trunc_err_value(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_TRUNC_SVD) && defined(TCICT_HAS_DIAGONAL_CHECK)
#ifdef TCICT_SKIP_TRUNC_SVD_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  auto matrix = trunc_svd_test_matrix<TenT>(ctx);

  TenT u, v_dag;
  tci::real_ten_t<TenT> s_diag;
  tci::real_t<TenT> trunc_err = -1.0;

  tci::trunc_svd(ctx, matrix, 1, u, s_diag, v_dag, trunc_err,
                 static_cast<tci::bond_dim_t<TenT>>(2), 0.0);

  TCICT_EXPECT_DIAGONAL(ctx, s_diag, trunc_svd_fixture_svs(2),
                        tolerance(fix, tol_category::factorization),
                        diag_order::any);
  TCICT_ASSERT_CLOSE(trunc_err, trunc_svd_expected_epsilon<TenT>(2),
                     tolerance(fix, tol_category::factorization));
#else
  (void)fix;
#endif
}

/// Verify trunc_err == 0 when no truncation occurs (chi_max >= kappa).
template <typename TenT>
void test_trunc_svd_trunc_err_no_truncation(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_TRUNC_SVD) && defined(TCICT_HAS_DIAGONAL_CHECK)
#ifdef TCICT_SKIP_TRUNC_SVD_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  auto matrix = trunc_svd_test_matrix<TenT>(ctx);

  TenT u, v_dag;
  tci::real_ten_t<TenT> s_diag;
  tci::real_t<TenT> trunc_err = -1.0;

  // chi_max = 10 > 4 (kappa), s_min = 0 → keep all SVs
  tci::trunc_svd(ctx, matrix, 1, u, s_diag, v_dag, trunc_err,
                 static_cast<tci::bond_dim_t<TenT>>(10), 0.0);

  TCICT_EXPECT_DIAGONAL(ctx, s_diag, trunc_svd_fixture_svs(4),
                        tolerance(fix, tol_category::factorization),
                        diag_order::any);
  TCICT_ASSERT_CLOSE(trunc_err, 0.0,
                     tolerance(fix, tol_category::factorization));
#else
  (void)fix;
#endif
}

/// Verify trunc_err is in [0, 1] (relative error is bounded).
template <typename TenT>
void test_trunc_svd_trunc_err_bounded(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_TRUNC_SVD) && defined(TCICT_HAS_DIAGONAL_CHECK)
#ifdef TCICT_SKIP_TRUNC_SVD_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  auto matrix = trunc_svd_test_matrix<TenT>(ctx);

  TenT u, v_dag;
  tci::real_ten_t<TenT> s_diag;
  tci::real_t<TenT> trunc_err = -1.0;

  // Keep only 1 SV: discard [2, 1, 0.1]
  tci::trunc_svd(ctx, matrix, 1, u, s_diag, v_dag, trunc_err,
                 static_cast<tci::bond_dim_t<TenT>>(1), 0.0);

  TCICT_EXPECT_DIAGONAL(ctx, s_diag, trunc_svd_fixture_svs(1),
                        tolerance(fix, tol_category::factorization),
                        diag_order::any);
  TCICT_ASSERT(trunc_err >= 0.0);
  TCICT_ASSERT(trunc_err <= 1.0);

  TCICT_ASSERT_CLOSE(trunc_err, trunc_svd_expected_epsilon<TenT>(1),
                     tolerance(fix, tol_category::factorization));
#else
  (void)fix;
#endif
}

/// Verify trunc_svd on a non-diagonal input, so a backend that copied the
/// input's diagonal into sigma fails. [[1, 2], [3, 4]] has the singular values
/// sqrt(15 +- sqrt(221)): the eigenvalues of A^T A, which has trace 30 and
/// determinant 4, are 15 +- sqrt(221). chi_max = 1 keeps the larger one, and
/// epsilon(1) is the spec's ratio for discarding the smaller.
template <typename TenT>
void test_trunc_svd_non_diagonal_input(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_TRUNC_SVD) && defined(TCICT_HAS_DIAGONAL_CHECK)
#ifdef TCICT_SKIP_TRUNC_SVD_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::factorization);
  auto matrix = tci::zeros<TenT>(ctx, {2, 2});
  tci::set_elem(ctx, matrix, {0, 0}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, matrix, {0, 1}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, matrix, {1, 0}, make_elem<TenT>(3.0));
  tci::set_elem(ctx, matrix, {1, 1}, make_elem<TenT>(4.0));

  TenT u, v_dag;
  tci::real_ten_t<TenT> s_diag;
  tci::real_t<TenT> trunc_err = -1.0;
  tci::trunc_svd(ctx, matrix, 1, u, s_diag, v_dag, trunc_err,
                 static_cast<tci::bond_dim_t<TenT>>(1),
                 static_cast<tci::real_t<TenT>>(0.0));

  const double svs[] = {std::sqrt(15.0 + std::sqrt(221.0)),
                        std::sqrt(15.0 - std::sqrt(221.0))};
  TCICT_EXPECT_DIAGONAL(ctx, s_diag, {svs[0]}, tol, diag_order::any);
  TCICT_ASSERT_CLOSE(trunc_err, trunc_svd_expected_epsilon_from<TenT>(svs, 2, 1),
                     tol);
#else
  (void)fix;
#endif
}

// --- truncated SVD, overload (2): chi_max / s_min / chi_min / target_trunc_err ---
//
// V1's rule, with epsilon(chi) = sum_{i>=chi} s_i^2 / sum_{i<kappa} s_i^2:
//
//   - Discard all s_i < s_min.
//   - Among the survivors retain at least chi_min when possible; values below
//     s_min are NOT restored to satisfy chi_min.
//   - Grow chi in descending order until epsilon <= target_trunc_err or
//     chi == chi_max.
//
// That middle clause is what keeps the retained chi from being simply "the
// smallest chi in [chi_min, chi_max] meeting the target": when fewer than
// chi_min values clear s_min, the result is below chi_min.
//
// For the [3, 2, 1, 0.1] fixture, sum s_i^2 = 14.01 and the ladder runs
// epsilon(4) = 0, epsilon(3) = 0.01/14.01 ≈ 7.138e-4,
// epsilon(2) = 1.01/14.01 ≈ 0.07209, epsilon(1) = 5.01/14.01 ≈ 0.3576. The
// targets below sit well clear of these, so the tests do not depend on how the
// quotients round; the one test that does need an exact quotient chooses a
// different spectrum and asserts the exactness it needs.

/// Verify target_trunc_err selects an interior chi via the epsilon ladder.
/// epsilon(2) ≈ 0.07209 <= 0.1 < epsilon(1) ≈ 0.3576, so chi must be 2.
template <typename TenT>
void test_trunc_svd_target_err_selects_chi(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_TRUNC_SVD) && defined(TCICT_HAS_DIAGONAL_CHECK)
#ifdef TCICT_SKIP_TRUNC_SVD_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  trunc_svd_expect_retained_chi<TenT>(fix, /*scale=*/1.0, /*chi_max=*/4, /*s_min=*/0.0,
                                     /*chi_min=*/1, /*target_trunc_err=*/0.1,
                                     /*expected_chi=*/2);
#else
  (void)fix;
#endif
}

/// Verify the target_trunc_err decision is invariant under a uniform rescaling
/// of the input. epsilon is a ratio of sums of squares, so scaling every
/// singular value by 1e-3 must not move the retained chi, while a threshold
/// applied to the raw singular values moves with the scale.
template <typename TenT>
void test_trunc_svd_target_err_scale_invariant(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_TRUNC_SVD) && defined(TCICT_HAS_DIAGONAL_CHECK)
#ifdef TCICT_SKIP_TRUNC_SVD_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  trunc_svd_expect_retained_chi<TenT>(fix, /*scale=*/1.0e-3, /*chi_max=*/4, /*s_min=*/0.0,
                                     /*chi_min=*/1, /*target_trunc_err=*/0.1,
                                     /*expected_chi=*/2);
#else
  (void)fix;
#endif
}

/// Verify chi_min floors the selection. epsilon(1) ≈ 0.3576 <= 0.5 would allow
/// chi = 1, but chi_min = 3 forbids it.
template <typename TenT>
void test_trunc_svd_target_err_chi_min_floor(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_TRUNC_SVD) && defined(TCICT_HAS_DIAGONAL_CHECK)
#ifdef TCICT_SKIP_TRUNC_SVD_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  trunc_svd_expect_retained_chi<TenT>(fix, /*scale=*/1.0, /*chi_max=*/4, /*s_min=*/0.0,
                                     /*chi_min=*/3, /*target_trunc_err=*/0.5,
                                     /*expected_chi=*/3);
#else
  (void)fix;
#endif
}

/// Verify chi_max caps the selection. No chi <= 2 reaches epsilon <= 1e-6, so
/// the growth stops at chi_max rather than continuing to satisfy the target.
template <typename TenT>
void test_trunc_svd_target_err_chi_max_cap(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_TRUNC_SVD) && defined(TCICT_HAS_DIAGONAL_CHECK)
#ifdef TCICT_SKIP_TRUNC_SVD_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  trunc_svd_expect_retained_chi<TenT>(fix, /*scale=*/1.0, /*chi_max=*/2, /*s_min=*/0.0,
                                     /*chi_min=*/1, /*target_trunc_err=*/1.0e-6,
                                     /*expected_chi=*/2);
#else
  (void)fix;
#endif
}

/// Verify overload (2) with chi_min = 1 and target_trunc_err = 0 reproduces
/// overload (1), which V1 defines as exactly that specialization.
///
/// The two calls are also pinned against the spec, not only against each other.
/// V1 lets a backend implement overload (1) by delegating to overload (2), and
/// a backend that does so satisfies any purely differential assertion no matter
/// what the shared route computes. Anchoring the general call's retained chi and
/// truncation error to the values the spec prescribes for chi_max = 2 keeps the
/// test able to fail on such a backend.
template <typename TenT>
void test_trunc_svd_target_err_zero_matches_chi_max_overload(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_TRUNC_SVD) && defined(TCICT_HAS_DIAGONAL_CHECK)
#ifdef TCICT_SKIP_TRUNC_SVD_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::factorization);

  TenT u_general, v_dag_general;
  tci::real_ten_t<TenT> s_general;
  tci::real_t<TenT> err_general = -1.0;
  auto matrix_general = trunc_svd_test_matrix<TenT>(ctx);
  trunc_svd_by_strategy(ctx, matrix_general, 1, u_general, s_general, v_dag_general,
                        err_general, /*chi_max=*/2, /*s_min=*/0.0, /*chi_min=*/1,
                        /*target_trunc_err=*/0.0);

  TenT u_simple, v_dag_simple;
  tci::real_ten_t<TenT> s_simple;
  tci::real_t<TenT> err_simple = -1.0;
  auto matrix_simple = trunc_svd_test_matrix<TenT>(ctx);
  tci::trunc_svd(ctx, matrix_simple, 1, u_simple, s_simple, v_dag_simple, err_simple,
                 static_cast<tci::bond_dim_t<TenT>>(2), static_cast<tci::real_t<TenT>>(0.0));

  // No chi below chi_max = 2 reaches epsilon <= 0, so both calls must stop at
  // 2, keep the singular values 3 and 2, and report epsilon(2). Assert that
  // against the spec first: it is what the differential assertions below
  // cannot supply on their own. Both sigmas are checked here, before any of
  // their elements is read below.
  TCICT_EXPECT_DIAGONAL(ctx, s_general, trunc_svd_fixture_svs(2), tol,
                        diag_order::any);
  TCICT_EXPECT_DIAGONAL(ctx, s_simple, trunc_svd_fixture_svs(2), tol,
                        diag_order::any);
  TCICT_ASSERT_CLOSE(err_general, trunc_svd_expected_epsilon<TenT>(2), tol);

  // Equivalence is a claim about the whole decomposition, so compare every
  // output the two calls produce, not just the retained count. The sigmas are
  // also compared with each other, as multisets, since V1 fixes no order for
  // trunc_svd's diagonal.
  TCICT_ASSERT_CLOSE(err_general, err_simple, tol);
  TCICT_EXPECT_DIAGONAL(ctx, s_simple, TCICT_REAL_DIAGONAL_OF(ctx, s_general),
                        tol, diag_order::any);

  // The factors are compared through the product they reconstruct rather than
  // elementwise: for a spectrum with distinct singular values the columns of u
  // are fixed only up to a phase, and, since V1 fixes no order for sigma's
  // diagonal, up to a permutation shared with sigma, neither of which the two
  // calls are obliged to choose identically, so an elementwise comparison would
  // assert more than the equivalence claim. The reconstruction is invariant
  // under both, and unlike a shape comparison it does read the factors' values.
  using RealTenT = tci::real_ten_t<TenT>;
  auto reconstruct = [&ctx](const TenT &u_in, const RealTenT &s_in, const TenT &v_in) {
    const auto u_shape = tci::shape(ctx, u_in);
    // u is {4, 2}: its rows are the fixture's and its bond is sigma's, already
    // checked to be {2, 2}. Assert that before reading either along it.
    TCICT_ASSERT(u_shape.size() == 2 &&
                 u_shape[0] == static_cast<tci::bond_dim_t<TenT>>(
                                   kTruncSvdFixtureRank) &&
                 u_shape[1] == 2);
    auto scaled = tci::copy(ctx, u_in);
    for (tci::bond_dim_t<TenT> j = 0; j < u_shape[1]; ++j) {
      const auto cj_sigma = static_cast<tci::elem_coor_t<RealTenT>>(j);
      const auto sj =
          real_part<RealTenT>(tci::get_elem(ctx, s_in, {cj_sigma, cj_sigma}));
      for (tci::bond_dim_t<TenT> i = 0; i < u_shape[0]; ++i) {
        const auto ci = static_cast<tci::elem_coor_t<TenT>>(i);
        const auto cj = static_cast<tci::elem_coor_t<TenT>>(j);
        const auto elem = tci::get_elem(ctx, u_in, {ci, cj});
        tci::set_elem(ctx, scaled, {ci, cj},
                      make_elem<TenT>(real_part<TenT>(elem) * sj,
                                      imag_part<TenT>(elem) * sj));
      }
    }
    // v_dag is {2, 4}, sigma's bond and the fixture's columns, before the
    // contraction over its bond.
    const auto v_shape = tci::shape(ctx, v_in);
    TCICT_ASSERT(v_shape.size() == 2 && v_shape[0] == 2 &&
                 v_shape[1] == static_cast<tci::bond_dim_t<TenT>>(
                                   kTruncSvdFixtureRank));
    TenT product;
    tci::contract(ctx, scaled, "ik", v_in, "kj", product, "ij");
    return product;
  };

  auto reconstructed_general = reconstruct(u_general, s_general, v_dag_general);
  auto reconstructed_simple = reconstruct(u_simple, s_simple, v_dag_simple);
  TCICT_ASSERT(tci::close(ctx, reconstructed_general, reconstructed_simple, tol));

  // The two products agreeing does not show that sigma is paired with the
  // right columns of u and v, since sigma's order is free: a shared route
  // could pair them wrongly in both. So the product is also compared with the
  // input's best rank-2 approximation. The fixture is diag(3, 2, 1, 0.1), whose
  // singular vectors are the unit vectors, so that approximation is
  // diag(3, 2, 0, 0).
  auto rank2 =
      tci::zeros<TenT>(ctx, {kTruncSvdFixtureRank, kTruncSvdFixtureRank});
  for (int i = 0; i < 2; ++i) {
    const auto coor = static_cast<tci::elem_coor_t<TenT>>(i);
    tci::set_elem(ctx, rank2, {coor, coor},
                  make_elem<TenT>(kTruncSvdFixtureSvs[i]));
  }
  TCICT_ASSERT(tci::close(ctx, reconstructed_general, rank2, tol));
#else
  (void)fix;
#endif
}

/// Verify that singular values discarded by s_min are not restored to satisfy
/// chi_min. s_min = 0.5 leaves survivors [3, 2, 1], and chi_min = 4 cannot be
/// met, so chi must be 3 rather than 4.
template <typename TenT>
void test_trunc_svd_chi_min_not_restored_below_s_min(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_TRUNC_SVD) && defined(TCICT_HAS_DIAGONAL_CHECK)
#ifdef TCICT_SKIP_TRUNC_SVD_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  trunc_svd_expect_retained_chi<TenT>(fix, /*scale=*/1.0, /*chi_max=*/4, /*s_min=*/0.5,
                                     /*chi_min=*/4, /*target_trunc_err=*/0.0,
                                     /*expected_chi=*/3);
#else
  (void)fix;
#endif
}

/// Verify that chi stops growing as soon as epsilon reaches target_trunc_err:
/// the comparison bounding the growth is `<=`, not `<`.
///
/// Separating the two operators needs a target that lands exactly on an
/// achievable epsilon: only there do they disagree. `<=` then retains that chi
/// and `<` grows past it.
///
/// Which bit pattern the backend's epsilon carries is not something the test may
/// assume. The spec fixes epsilon as a formula, not an arithmetic path, so two
/// conforming backends can land an ulp apart on the same spectrum — one summing
/// s_i^2 directly, another dividing through a squared Frobenius norm. A target
/// written as a literal would therefore hit the tie on the first and miss it on
/// the second, and the miss looks exactly like a `<` implementation.
///
/// So the target is measured rather than written. The first call pins chi to 1
/// through chi_max, which fixes the retained set without consulting
/// target_trunc_err at all, and reports the backend's own epsilon for that set.
/// Feeding that value back as the target of the second call puts the tie on the
/// boundary by construction, whatever route the backend computes epsilon by.
///
/// That reuse assumes the backend does not realize the one epsilon V1 names by
/// two divergent expressions — the value it reports in trunc_err has to be the
/// value it compares against target_trunc_err. V1 introduces epsilon once and uses that single
/// symbol for both, so this is the specification's own reading rather than an
/// extra requirement; but V1 constrains no arithmetic path, so an implementation
/// that summed the discarded tail for the comparison and took the complement of
/// the retained prefix for the report could land the two an ulp apart and fail
/// here while conforming. A backend in that shape should compute epsilon once.
///
/// SVs [5, 4, 3] give sum s_i^2 = 50 with 16 + 9 = 25 discarded at chi = 1, so
/// the measured epsilon is 25 / 50 up to the backend's rounding. It stays well
/// away from epsilon(2) = 9 / 50, so no rounding an implementation could
/// plausibly carry lets chi = 2 satisfy the target and blur the two operators.
template <typename TenT>
void test_trunc_svd_target_err_boundary_is_inclusive(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_TRUNC_SVD) && defined(TCICT_HAS_DIAGONAL_CHECK)
#ifdef TCICT_SKIP_TRUNC_SVD_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::factorization);

  static constexpr double kSvs[] = {5.0, 4.0, 3.0};
  static constexpr int kRank = static_cast<int>(sizeof(kSvs) / sizeof(kSvs[0]));

  auto build = [&ctx]() {
    auto m = tci::zeros<TenT>(ctx, {kRank, kRank});
    for (int i = 0; i < kRank; ++i) {
      const auto coor = static_cast<tci::elem_coor_t<TenT>>(i);
      tci::set_elem(ctx, m, {coor, coor}, make_elem<TenT>(kSvs[i]));
    }
    return m;
  };

  // Measure epsilon(1) with chi pinned by chi_max, so the value is read off a
  // call that never exercises the comparison under test.
  TenT u_pinned, v_dag_pinned;
  tci::real_ten_t<TenT> s_pinned;
  tci::real_t<TenT> epsilon_at_one = -1.0;
  auto matrix_pinned = build();
  tci::trunc_svd(ctx, matrix_pinned, 1, u_pinned, s_pinned, v_dag_pinned, epsilon_at_one,
                 static_cast<tci::bond_dim_t<TenT>>(1), static_cast<tci::real_t<TenT>>(0.0));

  TCICT_EXPECT_DIAGONAL(ctx, s_pinned, {5.0}, tol, diag_order::any);
  TCICT_ASSERT_CLOSE(epsilon_at_one, trunc_svd_expected_epsilon_from<TenT>(kSvs, kRank, 1),
                     tol);

  // Now let the same epsilon arrive as the target. chi = 1 is admissible under
  // `<=` and inadmissible under `<`.
  TenT u, v_dag;
  tci::real_ten_t<TenT> s_diag;
  tci::real_t<TenT> trunc_err = -1.0;
  auto matrix = build();
  trunc_svd_by_strategy(ctx, matrix, 1, u, s_diag, v_dag, trunc_err, /*chi_max=*/3,
                        /*s_min=*/0.0, /*chi_min=*/1, /*target_trunc_err=*/epsilon_at_one);

  TCICT_EXPECT_DIAGONAL(ctx, s_diag, {5.0}, tol, diag_order::any);
#else
  (void)fix;
#endif
}

/// Verify that overload (2) reads its arguments as chi_max, s_min, chi_min,
/// target_trunc_err. Each argument converts to whichever of the four
/// parameters it lands on, so this asserts values that differ when the order
/// is chi_min, chi_max, target_trunc_err, s_min instead, whether in the
/// backend's declaration or in the suite's call. It does not tell every
/// order apart: chi_max = 3 and s_min = 3 swapped read the same.
///
/// At scale 10 the fixture's singular values are 30, 20, 10, 1, with squares
/// summing to 1401, so epsilon(1) = 501 / 1401 ≈ 0.358, epsilon(2) =
/// 101 / 1401 ≈ 0.0721 and epsilon(3) = 1 / 1401.
///
/// - Read as chi_max = 3, s_min = 3, chi_min = 1, target_trunc_err = 0.1:
///   1 < 3 is discarded, and growth from one value stops at two, since
///   epsilon(1) > 0.1 >= epsilon(2). Sigma is 30, 20.
/// - Read as chi_min = 3, chi_max = 3, target_trunc_err = 1, s_min = 0.1, the
///   (chi_min, chi_max, target_trunc_err, s_min) declaration: nothing is
///   discarded, and three values are retained. Sigma is 30, 20, 10.
/// - Passed as (1, 3, 0.1, 3) to a (chi_max, s_min, chi_min,
///   target_trunc_err) declaration, the same permutation in the call: chi_max
///   is 1, so at most 30 is retained.
///
/// Each threshold sits at least 0.027 from the value it is compared with.
template <typename TenT>
void test_trunc_svd_argument_order(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_TRUNC_SVD) && defined(TCICT_HAS_DIAGONAL_CHECK)
#ifdef TCICT_SKIP_TRUNC_SVD_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  trunc_svd_expect_retained_chi<TenT>(fix, /*scale=*/10.0, /*chi_max=*/3, /*s_min=*/3.0,
                                     /*chi_min=*/1, /*target_trunc_err=*/0.1,
                                     /*expected_chi=*/2);
#else
  (void)fix;
#endif
}

// --- eig (general eigendecomposition of identity) ---

template <typename TenT> void test_eig_identity(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_EIG) && defined(TCICT_HAS_DIAGONAL_CHECK)
#ifdef TCICT_SKIP_EIG_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  auto matrix = tci::eye<TenT>(ctx, 2);

  using CplxTenT = tci::cplx_ten_t<TenT>;
  CplxTenT eigenvals, eigenvecs;
  tci::eig(ctx, matrix, 1, eigenvals, eigenvecs);

  TCICT_EXPECT_DIAGONAL(ctx, eigenvals, {1.0, 1.0},
                        tolerance(fix, tol_category::factorization));

  TCICT_ASSERT(tci::order(ctx, eigenvecs) == 2);
#else
  (void)fix;
#endif
}

// --- eig: distinct eigenvalues of a non-diagonal matrix ---

// [[0, 1], [2, 1]] has the characteristic polynomial
// lambda^2 - lambda - 2 = (lambda - 2)(lambda + 1), so its eigenvalues are -1
// and 2, real and distinct. The input is not diagonal and its diagonal (0, 1)
// holds neither, so a backend that returned the input or its diagonal fails.
// V1 fixes no order for eig's eigenvalues, so the diagonal is compared as a
// multiset.
template <typename TenT> void test_eig_distinct(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_EIG) && defined(TCICT_HAS_DIAGONAL_CHECK)
#ifdef TCICT_SKIP_EIG_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  auto matrix = tci::zeros<TenT>(ctx, {2, 2});
  tci::set_elem(ctx, matrix, {0, 1}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, matrix, {1, 0}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, matrix, {1, 1}, make_elem<TenT>(1.0));

  tci::cplx_ten_t<TenT> eigenvals, eigenvecs;
  tci::eig(ctx, matrix, 1, eigenvals, eigenvecs);

  TCICT_EXPECT_DIAGONAL(ctx, eigenvals, {-1.0, 2.0},
                        tolerance(fix, tol_category::factorization),
                        diag_order::any);
#else
  (void)fix;
#endif
}

// --- eigh (Hermitian eigendecomposition of identity) ---

template <typename TenT> void test_eigh_identity(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_EIGH) && defined(TCICT_HAS_DIAGONAL_CHECK)
#ifdef TCICT_SKIP_EIGH_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  auto matrix = tci::eye<TenT>(ctx, 2);

  tci::real_ten_t<TenT> eigenvals;
  TenT eigenvecs;
  tci::eigh(ctx, matrix, 1, eigenvals, eigenvecs);

  TCICT_EXPECT_DIAGONAL(ctx, eigenvals, {1.0, 1.0},
                        tolerance(fix, tol_category::factorization));
  TCICT_ASSERT(tci::order(ctx, eigenvecs) == 2);
#else
  (void)fix;
#endif
}

// --- eigh: distinct eigenvalues in ascending order ---

// The symmetric [[2, 1], [1, 2]] has det(A - lambda I) = (2 - lambda)^2 - 1,
// so its eigenvalues are 1 and 3; V1 has eigh hold them "in ascending order",
// so lambda_mat is diag(1, 3). The input's diagonal (2, 2) holds neither.
template <typename TenT>
void test_eigh_ascending(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_EIGH) && defined(TCICT_HAS_DIAGONAL_CHECK)
#ifdef TCICT_SKIP_EIGH_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  auto matrix = tci::zeros<TenT>(ctx, {2, 2});
  tci::set_elem(ctx, matrix, {0, 0}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, matrix, {0, 1}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, matrix, {1, 0}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, matrix, {1, 1}, make_elem<TenT>(2.0));

  tci::real_ten_t<TenT> eigenvals;
  TenT eigenvecs;
  tci::eigh(ctx, matrix, 1, eigenvals, eigenvecs);

  TCICT_EXPECT_DIAGONAL(ctx, eigenvals, {1.0, 3.0},
                        tolerance(fix, tol_category::factorization));
#else
  (void)fix;
#endif
}

// --- exp: identity matrix ---

template <typename TenT> void test_exp_identity(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_EXP
#ifdef TCICT_SKIP_EXP_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  auto identity = tci::eye<TenT>(ctx, 3);

  TenT result;
  tci::exp(ctx, identity, 1, result);

  // exp(I) = e*I
  double expected_e = std::exp(1.0);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {1, 1})),
                     expected_e, eps);
  TCICT_ASSERT_CLOSE(std::abs(tci::get_elem(ctx, result, {0, 1})), 0.0, eps);
#else
  (void)fix;
#endif
}

// --- exp: diagonal matrix ---

template <typename TenT> void test_exp_diagonal(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_EXP
#ifdef TCICT_SKIP_EXP_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  auto diagonal = tci::zeros<TenT>(ctx, {2, 2});
  tci::set_elem(ctx, diagonal, {0, 0}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, diagonal, {1, 1}, make_elem<TenT>(2.0));

  TenT result;
  tci::exp(ctx, diagonal, 1, result);

  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 0})),
                     std::exp(1.0), eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {1, 1})),
                     std::exp(2.0), eps);
  TCICT_ASSERT_CLOSE(std::abs(tci::get_elem(ctx, result, {0, 1})), 0.0, eps);
  TCICT_ASSERT_CLOSE(std::abs(tci::get_elem(ctx, result, {1, 0})), 0.0, eps);
#else
  (void)fix;
#endif
}

// --- exp: zero matrix → identity ---

template <typename TenT> void test_exp_zero(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_EXP
#ifdef TCICT_SKIP_EXP_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  auto zero_matrix = tci::zeros<TenT>(ctx, {2, 2});

  TenT result;
  tci::exp(ctx, zero_matrix, 1, result);

  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 0})), 1.0,
                     eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {1, 1})), 1.0,
                     eps);
  TCICT_ASSERT_CLOSE(std::abs(tci::get_elem(ctx, result, {0, 1})), 0.0, eps);
#else
  (void)fix;
#endif
}

// --- exp: anti-Hermitian → unitary ---

template <typename TenT>
void test_exp_anti_hermitian(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_EXP
#ifdef TCICT_SKIP_EXP_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::iterative);
  auto anti_herm = tci::zeros<TenT>(ctx, {2, 2});
  tci::set_elem(ctx, anti_herm, {0, 1}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, anti_herm, {1, 0}, make_elem<TenT>(-1.0));

  TenT result;
  tci::exp(ctx, anti_herm, 1, result);

  // exp([[0,1],[-1,0]]) = [[cos(1), sin(1)], [-sin(1), cos(1)]]
  auto c = std::cos(1.0);
  auto s = std::sin(1.0);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 0})), c, tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 1})), s, tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {1, 0})), -s, tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {1, 1})), c, tol);
#else
  (void)fix;
#endif
}

// --- inverse ---

template <typename TenT> void test_inverse(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_INVERSE
#ifdef TCICT_SKIP_INVERSE_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  auto matrix = tci::zeros<TenT>(ctx, {2, 2});
  // [[4,7],[2,6]] → inv = [[0.6,-0.7],[-0.2,0.4]]
  tci::set_elem(ctx, matrix, {0, 0}, make_elem<TenT>(4.0));
  tci::set_elem(ctx, matrix, {0, 1}, make_elem<TenT>(7.0));
  tci::set_elem(ctx, matrix, {1, 0}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, matrix, {1, 1}, make_elem<TenT>(6.0));

  TenT inv;
  TCICT_ASSERT_NOTHROW(tci::inverse(ctx, matrix, 1, inv));

  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, inv, {0, 0})), 0.6,
                     eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, inv, {0, 1})), -0.7,
                     eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, inv, {1, 0})), -0.2,
                     eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, inv, {1, 1})), 0.4,
                     eps);
#else
  (void)fix;
#endif
}

// --- scale: in-place ---

template <typename TenT> void test_scale_inplace(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_SCALE
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);

  auto tensor = tci::zeros<TenT>(ctx, {2, 2});
  tci::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, tensor, {0, 1}, make_elem<TenT>(4.0));
  tci::set_elem(ctx, tensor, {1, 0}, make_elem<TenT>(6.0));
  tci::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(8.0));

  tci::scale(ctx, tensor, make_elem<TenT>(0.5));

  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 0})), 1.0,
                     tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 1})), 2.0,
                     tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {1, 0})), 3.0,
                     tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {1, 1})), 4.0,
                     tol);
#else
  (void)fix;
#endif
}

// --- scale: out-of-place ---

template <typename TenT>
void test_scale_outofplace(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_SCALE
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);

  auto tensor = tci::zeros<TenT>(ctx, {2, 2});
  tci::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(3.0));
  tci::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(6.0));

  TenT result;
  tci::scale(ctx, tensor, make_elem<TenT>(-2.0), result);

  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 0})), -6.0,
                     tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {1, 1})), -12.0,
                     tol);
  // Input unchanged; see "Out-of-place inputs" in CONTRIBUTING.md.
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 0})), 3.0,
                     tol);
#else
  (void)fix;
#endif
}

// --- scale: by zero ---

template <typename TenT> void test_scale_by_zero(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_SCALE
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);

  auto tensor = tci::eye<TenT>(ctx, 2);
  tci::scale(ctx, tensor, make_elem<TenT>(0.0));

  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 0})), 0.0,
                     tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {1, 1})), 0.0,
                     tol);
#else
  (void)fix;
#endif
}

// --- scale: zeroth-order ---

// A complex scalar for a complex element type, so a backend conjugating it or
// dropping its imaginary part fails.
template <typename TenT>
void test_scale_zeroth_order(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_SCALE) && defined(TCICT_HAS_ZEROTH_ORDER_SWEEP)
  auto &ctx = fix.context();
  const std::complex<double> s(-1.5, is_complex_v<TenT> ? 0.5 : 0.0);
  const auto s_elem = make_elem<TenT>(s.real(), s.imag());

  auto tensor = tci::zeros<TenT>(ctx, {});
  TCICT_SET_ZEROTH_ORDER(fix, tensor, 2.0);
  tci::scale(ctx, tensor, s_elem);
  TCICT_EXPECT_ZEROTH_ORDER(fix, tensor, written<TenT>(2.0) * s,
                            tol_category::elementwise);

  // Out of place; the input is read back, see "Out-of-place inputs" in
  // CONTRIBUTING.md.
  auto input = tci::zeros<TenT>(ctx, {});
  TCICT_SET_ZEROTH_ORDER(fix, input, -0.5);
  TenT output;
  tci::scale(ctx, input, s_elem, output);
  TCICT_EXPECT_ZEROTH_ORDER(fix, output, written<TenT>(-0.5) * s,
                            tol_category::elementwise);
  TCICT_EXPECT_ZEROTH_ORDER(fix, input, -0.5);
#else
  (void)fix;
#endif
}

// --- trace: 2x2 matrix ---

template <typename TenT> void test_trace_partial(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_TRACE) && defined(TCICT_HAS_ZEROTH_ORDER_SWEEP)
  auto &ctx = fix.context();

  // 2x2 diagonal matrix [[1,0],[0,4]] → trace over {0,1} = 1+4 = 5
  auto matrix = tci::zeros<TenT>(ctx, {2, 2});
  tci::set_elem(ctx, matrix, {0, 0}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, matrix, {1, 1}, make_elem<TenT>(4.0));

  TenT result;
  tci::trace(ctx, matrix, {{0, 1}}, result);

  // TCAPI V1: "All bonds paired yields a scalar (zeroth-order)": order 0,
  // shape {}, size 1, read at {}.
  TCICT_EXPECT_ZEROTH_ORDER(fix, result, std::complex<double>(5.0, 0.0),
                            tol_category::reduction, 2);
#else
  (void)fix;
#endif
}

// --- trace: a bond left over ---

// A trace that pairs some bonds but not all keeps the unpaired ones, so a
// backend that makes every trace zeroth-order fails here. The {2, 3, 2} ramp
// gives element (i, j, k) the real part 1 + 6i + 2j + k; pairing bonds 0 and 2
// leaves bond 1, with element j the sum over i of element (i, j, i).
template <typename TenT>
void test_trace_leaving_bond(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_TRACE) && defined(TCICT_HAS_ELEM_SWEEP)
  auto &ctx = fix.context();
  auto at = ramp_over<TenT>({2, 3, 2}, 1.0);
  auto tensor = tci::zeros<TenT>(ctx, {2, 3, 2});
  TCICT_SET_ELEMS(fix, tensor, at);

  tci::trace(ctx, tensor, {{0, 2}});

  TCICT_EXPECT_ELEMS(
      fix, tensor, {3},
      [&](const tci::elem_coors_t<TenT> &x) {
        const auto j = static_cast<std::size_t>(x[0]);
        return at(coors_of<TenT>(0, j, 0)) + at(coors_of<TenT>(1, j, 1));
      },
      tol_category::reduction, 2);
#else
  (void)fix;
#endif
}

// --- svd: basic singular values and shapes ---

template <typename TenT> void test_svd_basic(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_SVD) && defined(TCICT_HAS_DIAGONAL_CHECK)
#ifdef TCICT_SKIP_SVD_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();

  // Diagonal matrix with known singular values [3, 1]
  auto matrix = tci::zeros<TenT>(ctx, {2, 2});
  tci::set_elem(ctx, matrix, {0, 0}, make_elem<TenT>(3.0));
  tci::set_elem(ctx, matrix, {1, 1}, make_elem<TenT>(1.0));

  TenT u, v_dag;
  tci::real_ten_t<TenT> s_diag;
  tci::svd(ctx, matrix, 1, u, s_diag, v_dag);

  // sigma is the {2, 2} diagonal tensor holding the singular values 3 and 1
  // in non-increasing order.
  TCICT_EXPECT_DIAGONAL(ctx, s_diag, {3.0, 1.0},
                        tolerance(fix, tol_category::factorization));

  // U and V† are (2, 2): their outer bonds are the input's, and their bond
  // with sigma is sigma's, {2, 2} as checked above.
  auto u_shape = tci::shape(ctx, u);
  auto v_shape = tci::shape(ctx, v_dag);
  TCICT_ASSERT(u_shape.size() == 2);
  TCICT_ASSERT(u_shape[0] == 2 && u_shape[1] == 2);
  TCICT_ASSERT(v_shape.size() == 2);
  TCICT_ASSERT(v_shape[0] == 2 && v_shape[1] == 2);
#else
  (void)fix;
#endif
}

// --- svd: reconstruction U * sigma * V† ≈ A ---

template <typename TenT>
void test_svd_reconstruction(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_SVD) && defined(TCICT_HAS_DIAGONAL_CHECK)
#ifdef TCICT_SKIP_SVD_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();

  // [[1,2],[3,4]]
  auto matrix = tci::zeros<TenT>(ctx, {2, 2});
  tci::set_elem(ctx, matrix, {0, 0}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, matrix, {0, 1}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, matrix, {1, 0}, make_elem<TenT>(3.0));
  tci::set_elem(ctx, matrix, {1, 1}, make_elem<TenT>(4.0));

  TenT u, v_dag;
  tci::real_ten_t<TenT> s_diag;
  tci::svd(ctx, matrix, 1, u, s_diag, v_dag);

  // sigma is the {2, 2} diagonal tensor of the singular values of
  // [[1, 2], [3, 4]], sqrt(15 + sqrt(221)) and sqrt(15 - sqrt(221)): the
  // eigenvalues of A^T A, which has trace 30 and determinant 4, are
  // 15 +- sqrt(221).
  TCICT_EXPECT_DIAGONAL(
      ctx, s_diag,
      {std::sqrt(15.0 + std::sqrt(221.0)), std::sqrt(15.0 - std::sqrt(221.0))},
      tolerance(fix, tol_category::factorization));

  // Reconstruct: scale column j of U by sigma's {j, j} element, then contract
  // with V†: u_scaled[i,j] = u[i,j] * sigma[j,j]
  using RealTenT = tci::real_ten_t<TenT>;
  // u is {2, 2}: its rows are the input's and its bond is sigma's, checked
  // above to be {2, 2}. Assert that before reading u.
  const auto u_shape = tci::shape(ctx, u);
  TCICT_ASSERT(u_shape.size() == 2 && u_shape[0] == 2 && u_shape[1] == 2);
  auto u_scaled = tci::copy(ctx, u);
  for (tci::elem_coor_t<TenT> j = 0; j < 2; ++j) {
    const auto cj = static_cast<tci::elem_coor_t<RealTenT>>(j);
    auto sj = real_part<RealTenT>(tci::get_elem(ctx, s_diag, {cj, cj}));
    for (tci::elem_coor_t<TenT> i = 0; i < 2; ++i) {
      auto elem = tci::get_elem(ctx, u, {i, j});
      tci::set_elem(ctx, u_scaled, {i, j},
                    make_elem<TenT>(real_part<TenT>(elem) * sj,
                                    imag_part<TenT>(elem) * sj));
    }
  }

  // Reconstruct via contract: u_scaled[i,k] * v_dag[k,j] → reconstructed[i,j],
  // after checking that v_dag is {2, 2} too.
  const auto v_shape = tci::shape(ctx, v_dag);
  TCICT_ASSERT(v_shape.size() == 2 && v_shape[0] == 2 && v_shape[1] == 2);
  TenT reconstructed;
  tci::contract(ctx, u_scaled, "ik", v_dag, "kj", reconstructed, "ij");

  TCICT_ASSERT(tci::close(ctx, reconstructed, matrix,
                          tolerance(fix, tol_category::factorization)));
#else
  (void)fix;
#endif
}

// --- svd / qr / lq: inputs other than a square matrix with one row bond ---
//
// TCAPI V1 matricizes an input of shape {d_0, ..., d_{r-1}} into an I x J
// matrix, the first k = num_of_bds_as_row bonds giving the rows, and folds the
// factors back with a bond of dimension min(I, J) between them: u (or q, or
// l) has shape {d_0, ..., d_{k-1}, min(I, J)} and v_dag (or r, or q)
// {min(I, J), d_k, ..., d_{r-1}}. V1 defines no linear order for flattening
// several bonds into one index, so these checks assert only what does not
// depend on one: shapes, sigma's diagonal, reconstruction by bond labels, and
// orthonormality as a contraction over whole bonds. Triangularity is asserted
// where the triangular factor is itself second-order: r with one column bond
// and l with one row bond. Reordering the rows of the matrix, A' -> P A',
// turns a QR factorization Q R into (P Q) R, again a QR factorization with
// the same r, and
// reordering its columns does the same for l, so r and l are triangular
// whatever order the bonds they do not keep flatten in.

#if defined(TCICT_HAS_ELEM_SWEEP) &&                                           \
    (!defined(TCICT_SKIP_SVD) || !defined(TCICT_SKIP_QR) ||                    \
     !defined(TCICT_SKIP_LQ))
namespace factor_check {

/// `count` bond labels, one letter each, starting at the `first`-th letter.
/// The factor bond takes 'x', and 'y' its partner in a Gram contraction, so
/// the labels stop at 'w': at most 23 of them.
inline std::string bond_labels(std::size_t first, std::size_t count) {
  TCICT_ASSERT(first + count <= static_cast<std::size_t>('x' - 'a'));
  std::string labels;
  for (std::size_t b = 0; b < count; ++b) {
    labels += static_cast<char>('a' + first + b);
  }
  return labels;
}

/// The shapes V1 gives the two factors of `in_shape` matricized with `k` row
/// bonds.
template <typename TenT>
std::pair<tci::shape_t<TenT>, tci::shape_t<TenT>>
factor_shapes(const tci::shape_t<TenT> &in_shape, std::size_t k) {
  // V1 matricizes with 1 <= num_of_bds_as_row < order.
  TCICT_ASSERT(k >= 1 && k < in_shape.size());
  std::size_t rows = 1, cols = 1;
  for (std::size_t b = 0; b < in_shape.size(); ++b) {
    (b < k ? rows : cols) *= static_cast<std::size_t>(in_shape[b]);
  }
  using dim_t = typename tci::shape_t<TenT>::value_type;
  const auto bond = static_cast<dim_t>(std::min(rows, cols));
  const auto split = in_shape.begin() + static_cast<std::ptrdiff_t>(k);
  tci::shape_t<TenT> left(in_shape.begin(), split);
  left.push_back(bond);
  tci::shape_t<TenT> right{bond};
  right.insert(right.end(), split, in_shape.end());
  return {left, right};
}

/// A tensor of `shape` holding zeros apart from `entries`, each a coordinate
/// and the real value there. As a function it gives each coordinate's element
/// as TCICT_EXPECT_ELEMS compares it; `fill` writes the entries into a
/// tensor of `shape` that `zeros` built.
template <typename TenT> struct sparse_values {
  tci::shape_t<TenT> shape;
  std::vector<std::pair<tci::elem_coors_t<TenT>, double>> entries;

  std::complex<double> operator()(const tci::elem_coors_t<TenT> &coors) const {
    for (const auto &entry : entries) {
      if (entry.first == coors) {
        return {entry.second, 0.0};
      }
    }
    return {0.0, 0.0};
  }

  void fill(tci::context_handle_t<TenT> &ctx, TenT &tensor) const {
    TCICT_ASSERT(tci::shape(ctx, tensor) == shape);
    for (const auto &entry : entries) {
      TCICT_ASSERT(entry.first.size() == shape.size());
      for (std::size_t b = 0; b < shape.size(); ++b) {
        TCICT_ASSERT(static_cast<std::size_t>(entry.first[b]) <
                     static_cast<std::size_t>(shape[b]));
      }
      tci::set_elem(ctx, tensor, entry.first, make_elem<TenT>(entry.second));
    }
  }
};

/// A dense tensor of `shape`: the coordinate at row-major position p holds
/// real part ((7 p) mod 11) + 1 and, for a complex element type, imaginary
/// part ((5 p) mod 7) - 3. Unlike ramp_over, whose values are affine in the
/// coordinates and so matricize to rank at most 2, the {2, 3, 4} instance has
/// full rank with one row bond and with two, and its imaginary parts are not
/// a multiple of its real parts, as TCICT_SET_ELEMS's are, so a factor that
/// should be conjugated and is not changes the product. As a function it
/// gives each coordinate's element as TCICT_EXPECT_ELEMS compares it; `fill`
/// writes them into a tensor of `shape` that `zeros` built.
template <typename TenT> struct dense_values {
  tci::shape_t<TenT> shape;

  std::complex<double> operator()(const tci::elem_coors_t<TenT> &coors) const {
    const auto p = static_cast<long>(ramp_over<TenT>(shape, 0.0)(coors));
    const double re = static_cast<double>((7 * p) % 11 + 1);
    const double im = is_complex_v<TenT> ? static_cast<double>((5 * p) % 7 - 3)
                                         : 0.0;
    return {re, im};
  }

  void fill(tci::context_handle_t<TenT> &ctx, TenT &tensor) const {
    TCICT_ASSERT(tci::shape(ctx, tensor) == shape);
    detail::for_each_coors<TenT>(
        shape, [&](const tci::elem_coors_t<TenT> &coors) {
          const auto v = (*this)(coors);
          tci::set_elem(ctx, tensor, coors,
                        make_elem<TenT>(v.real(), v.imag()));
        });
  }
};

/// The identity on {n, n}, as TCICT_EXPECT_ELEMS compares it.
template <typename TenT>
std::complex<double> identity_at(const tci::elem_coors_t<TenT> &coors) {
  return {coors[0] == coors[1] ? 1.0 : 0.0, 0.0};
}

/// Asserts that conj(f) contracted with f over the `k` bonds before f's last,
/// its factor bond, is the identity on that bond: orthonormal columns.
template <typename TenT>
void expect_orthonormal_columns(tci_test_fixture<TenT> &fix, const TenT &f,
                                std::size_t k) {
  auto &ctx = fix.context();
  const auto shape = tci::shape(ctx, f);
  TCICT_ASSERT(shape.size() == k + 1);
  const auto bond = shape.back();
  TenT f_conj;
  tci::cplx_conj(ctx, f, f_conj);
  const auto rows = bond_labels(0, k);
  TenT gram;
  tci::contract(ctx, f_conj, rows + "x", f, rows + "y", gram, "xy");
  TCICT_EXPECT_ELEMS(fix, gram, {bond, bond}, identity_at<TenT>,
                     tol_category::factorization);
}

/// Asserts that f contracted with conj(f) over the `count` bonds after f's
/// first, its factor bond, is the identity on that bond: orthonormal rows.
template <typename TenT>
void expect_orthonormal_rows(tci_test_fixture<TenT> &fix, const TenT &f,
                             std::size_t count) {
  auto &ctx = fix.context();
  const auto shape = tci::shape(ctx, f);
  TCICT_ASSERT(shape.size() == count + 1);
  const auto bond = shape.front();
  TenT f_conj;
  tci::cplx_conj(ctx, f, f_conj);
  const auto cols = bond_labels(0, count);
  TenT gram;
  tci::contract(ctx, f, "x" + cols, f_conj, "y" + cols, gram, "xy");
  TCICT_EXPECT_ELEMS(fix, gram, {bond, bond}, identity_at<TenT>,
                     tol_category::factorization);
}

/// Asserts that every element of the second-order `t` with
/// `zero_at(row, col)` is exactly zero, as V1's "upper triangular" and "lower
/// triangular" say of r and l, through TCICT_EXPECT_ELEMS with an exact
/// tolerance: zero where `zero_at` holds, and elsewhere the element read back.
template <typename TenT, typename ZeroAt>
void expect_zero_where(tci_test_fixture<TenT> &fix, const TenT &t,
                       ZeroAt zero_at) {
  auto &ctx = fix.context();
  const auto shape = tci::shape(ctx, t);
  TCICT_ASSERT(shape.size() == 2);
  TCICT_EXPECT_ELEMS(
      fix, t, shape,
      [&](const tci::elem_coors_t<TenT> &c) -> std::complex<double> {
        if (zero_at(static_cast<std::size_t>(c[0]),
                    static_cast<std::size_t>(c[1]))) {
          return {0.0, 0.0};
        }
        const auto e = tci::get_elem(ctx, t, c);
        return {real_part<TenT>(e), imag_part<TenT>(e)};
      },
      tol_category::exact);
}

/// Calls `f`, reporting an assertion failing or a std::exception thrown in it
/// with `label` in front, so a test checking several cases through one helper
/// says which case failed.
template <typename F> void in_case(const std::string &label, F f) {
  try {
    f();
  } catch (const assertion_error &e) {
    throw assertion_error(label + ": " + e.what());
  } catch (const std::exception &e) {
    throw assertion_error(label + ": unexpected exception: " + e.what());
  }
}

/// Asserts that `left`, whose last bond is the factor bond, contracted over
/// it with `right`, whose first bond is, gives back the input of shape
/// `in_shape` that `value_at` describes, matricized with `k` row bonds.
template <typename TenT, typename ValueAt>
void expect_product(tci_test_fixture<TenT> &fix, const TenT &left,
                    const TenT &right, std::size_t k,
                    const tci::shape_t<TenT> &in_shape, ValueAt value_at) {
  auto &ctx = fix.context();
  const auto rows = bond_labels(0, k);
  const auto cols = bond_labels(k, in_shape.size() - k);
  TenT product;
  tci::contract(ctx, left, rows + "x", right, "x" + cols, product, rows + cols);
  TCICT_EXPECT_ELEMS(fix, product, in_shape, value_at,
                     tol_category::factorization);
}

#if !defined(TCICT_SKIP_SVD) && defined(TCICT_HAS_DIAGONAL_CHECK)
/// svd of `input`, whose elements `value_at` gives, with `k` row bonds: the
/// factor shapes; sigma's diagonal against `singular_values` in V1's
/// non-increasing order when they are given, and otherwise sigma's shape and
/// that its diagonal, read back, is non-negative and non-increasing with
/// off-diagonal elements exactly zero; u * sigma * v_dag reproduces the
/// input; u has orthonormal columns and v_dag orthonormal rows.
template <typename TenT, typename ValueAt>
void check_svd(tci_test_fixture<TenT> &fix, const TenT &input, std::size_t k,
               ValueAt value_at, const std::vector<double> &singular_values) {
  using RealTenT = tci::real_ten_t<TenT>;
  auto &ctx = fix.context();
  const auto in_shape = tci::shape(ctx, input);
  const auto [u_shape, v_shape] = factor_shapes<TenT>(in_shape, k);
  const auto order = in_shape.size();

  TenT u, v_dag;
  RealTenT sigma;
  tci::svd(ctx, input, static_cast<tci::order_t<TenT>>(k), u, sigma, v_dag);
  TCICT_ASSERT(tci::shape(ctx, u) == u_shape);
  TCICT_ASSERT(tci::shape(ctx, v_dag) == v_shape);
  const auto bond = u_shape.back();
  const auto tol = tolerance(fix, tol_category::factorization);
  if (!singular_values.empty()) {
    TCICT_EXPECT_DIAGONAL(ctx, sigma, singular_values, tol);
  } else {
    // Values not known in closed form: sigma's diagonal, read back, must be
    // non-negative and non-increasing, and sigma diagonal around it.
    const auto s_shape = tci::shape(ctx, sigma);
    TCICT_ASSERT(s_shape.size() == 2 && s_shape[0] == bond &&
                 s_shape[1] == bond);
    const auto diagonal = TCICT_REAL_DIAGONAL_OF(ctx, sigma);
    for (std::size_t i = 0; i < diagonal.size(); ++i) {
      TCICT_ASSERT(diagonal[i] >= 0.0);
      TCICT_ASSERT(i == 0 || diagonal[i] <= diagonal[i - 1]);
    }
    TCICT_EXPECT_DIAGONAL(ctx, sigma, diagonal, tol);
  }

  // u with each element on factor-bond coordinate j scaled by sigma's {j, j}.
  auto u_scaled = tci::copy(ctx, u);
  detail::for_each_coors<TenT>(u_shape, [&](const tci::elem_coors_t<TenT> &c) {
    const auto j = c.back();
    const auto s = real_part<RealTenT>(
        tci::get_elem(ctx, sigma, coors_of<RealTenT>(j, j)));
    const auto e = tci::get_elem(ctx, u, c);
    tci::set_elem(ctx, u_scaled, c,
                  make_elem<TenT>(real_part<TenT>(e) * s,
                                  imag_part<TenT>(e) * s));
  });
  expect_product(fix, u_scaled, v_dag, k, in_shape, value_at);

  expect_orthonormal_columns(fix, u, k);
  expect_orthonormal_rows(fix, v_dag, order - k);
}
#endif

#ifndef TCICT_SKIP_QR
/// qr of `input` with `k` row bonds: the factor shapes; q * r reproduces the
/// input; q has orthonormal columns; when r is second-order, it is zero
/// below its diagonal.
template <typename TenT, typename ValueAt>
void check_qr(tci_test_fixture<TenT> &fix, const TenT &input, std::size_t k,
              ValueAt value_at) {
  auto &ctx = fix.context();
  const auto in_shape = tci::shape(ctx, input);
  const auto [q_shape, r_shape] = factor_shapes<TenT>(in_shape, k);

  TenT q, r;
  tci::qr(ctx, input, static_cast<tci::order_t<TenT>>(k), q, r);
  TCICT_ASSERT(tci::shape(ctx, q) == q_shape);
  TCICT_ASSERT(tci::shape(ctx, r) == r_shape);
  expect_product(fix, q, r, k, in_shape, value_at);
  expect_orthonormal_columns(fix, q, k);
  if (r_shape.size() == 2) {
    expect_zero_where(fix, r, [](std::size_t row, std::size_t col) {
      return row > col;
    });
  }
}
#endif

#ifndef TCICT_SKIP_LQ
/// lq of `input` with `k` row bonds: the factor shapes; l * q reproduces the
/// input; q has orthonormal rows; when l is second-order, it is zero above
/// its diagonal.
template <typename TenT, typename ValueAt>
void check_lq(tci_test_fixture<TenT> &fix, const TenT &input, std::size_t k,
              ValueAt value_at) {
  auto &ctx = fix.context();
  const auto in_shape = tci::shape(ctx, input);
  const auto [l_shape, q_shape] = factor_shapes<TenT>(in_shape, k);
  const auto order = in_shape.size();

  TenT l, q;
  tci::lq(ctx, input, static_cast<tci::order_t<TenT>>(k), l, q);
  TCICT_ASSERT(tci::shape(ctx, l) == l_shape);
  TCICT_ASSERT(tci::shape(ctx, q) == q_shape);
  expect_product(fix, l, q, k, in_shape, value_at);
  expect_orthonormal_rows(fix, q, order - k);
  if (l_shape.size() == 2) {
    expect_zero_where(fix, l, [](std::size_t row, std::size_t col) {
      return col > row;
    });
  }
}
#endif

/// [[1, 1, 0], [0, 0, 2]]: A A^T = diag(2, 4), so its singular values are 2
/// and sqrt(2); `transposed` gives the {3, 2} transpose, with the same ones.
template <typename TenT> sparse_values<TenT> wide_or_tall(bool transposed) {
  auto at = [transposed](std::size_t i, std::size_t j) {
    return transposed ? coors_of<TenT>(j, i) : coors_of<TenT>(i, j);
  };
  return {transposed ? tci::shape_t<TenT>{3, 2} : tci::shape_t<TenT>{2, 3},
          {{at(0, 0), 1.0}, {at(0, 1), 1.0}, {at(1, 2), 2.0}}};
}

/// The {2, 3, 4} tensor holding 3, 4, 5 and 12 at {0, 0, 0}, {0, 1, 1},
/// {1, 0, 2} and {1, 2, 3}. With one row bond its rows i = 0 and i = 1 have
/// disjoint supports, so A' A'^T = diag(25, 169) and the singular values are
/// 13 and 5; with two, each column k holds one entry, at distinct rows, so
/// A'^T A' = diag(9, 16, 25, 144) and they are 12, 5, 4 and 3. Reordering the
/// rows or columns of A' leaves both unchanged, so they hold for any linear
/// order. Its first and last bonds differ in dimension, so a backend forming
/// the row dimension from the last bonds instead gets factors of other
/// shapes.
template <typename TenT> sparse_values<TenT> third_order_sparse() {
  return {{2, 3, 4},
          {{coors_of<TenT>(0, 0, 0), 3.0},
           {coors_of<TenT>(0, 1, 1), 4.0},
           {coors_of<TenT>(1, 0, 2), 5.0},
           {coors_of<TenT>(1, 2, 3), 12.0}}};
}

/// Calls `check(matrix, values)` on the matrix of wide_or_tall and on its
/// transpose.
template <typename TenT, typename Check>
void on_wide_and_tall(tci_test_fixture<TenT> &fix, Check check) {
  auto &ctx = fix.context();
  for (bool transposed : {false, true}) {
    in_case(transposed ? "tall {3, 2}" : "wide {2, 3}", [&] {
      const auto at = wide_or_tall<TenT>(transposed);
      auto matrix = tci::zeros<TenT>(ctx, at.shape);
      at.fill(ctx, matrix);
      check(matrix, at);
    });
  }
}

/// Calls `check(tensor, values, k)` on the dense {2, 3, 4} fixture with one
/// and with two row bonds.
template <typename TenT, typename Check>
void on_dense_third_order(tci_test_fixture<TenT> &fix, Check check) {
  auto &ctx = fix.context();
  const dense_values<TenT> at{{2, 3, 4}};
  auto dense = tci::zeros<TenT>(ctx, at.shape);
  at.fill(ctx, dense);
  in_case("dense {2, 3, 4}, one row bond", [&] { check(dense, at, 1); });
  in_case("dense {2, 3, 4}, two row bonds", [&] { check(dense, at, 2); });
}

} // namespace factor_check
#endif

// --- svd: non-square matrices ---

template <typename TenT> void test_svd_non_square(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_SVD) && defined(TCICT_HAS_DIAGONAL_CHECK) &&           \
    defined(TCICT_HAS_ELEM_SWEEP)
#ifdef TCICT_SKIP_SVD_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  factor_check::on_wide_and_tall(fix, [&](const TenT &m, const auto &at) {
    factor_check::check_svd(fix, m, 1, at, {2.0, std::sqrt(2.0)});
  });
#else
  (void)fix;
#endif
}

// --- svd: a third-order input, one and two row bonds ---

template <typename TenT>
void test_svd_higher_order(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_SVD) && defined(TCICT_HAS_DIAGONAL_CHECK) &&           \
    defined(TCICT_HAS_ELEM_SWEEP)
#ifdef TCICT_SKIP_SVD_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  const auto sparse_at = factor_check::third_order_sparse<TenT>();
  auto sparse = tci::zeros<TenT>(ctx, sparse_at.shape);
  sparse_at.fill(ctx, sparse);
  factor_check::in_case("sparse {2, 3, 4}, one row bond", [&] {
    factor_check::check_svd(fix, sparse, 1, sparse_at, {13.0, 5.0});
  });
  factor_check::in_case("sparse {2, 3, 4}, two row bonds", [&] {
    factor_check::check_svd(fix, sparse, 2, sparse_at, {12.0, 5.0, 4.0, 3.0});
  });

  factor_check::on_dense_third_order(
      fix, [&](const TenT &d, const auto &at, std::size_t k) {
        factor_check::check_svd(fix, d, k, at, {});
      });
#else
  (void)fix;
#endif
}

// --- qr: non-square matrices ---

template <typename TenT> void test_qr_non_square(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_QR) && defined(TCICT_HAS_ELEM_SWEEP)
#ifdef TCICT_SKIP_QR_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  factor_check::on_wide_and_tall(fix, [&](const TenT &m, const auto &at) {
    factor_check::check_qr(fix, m, 1, at);
  });
#else
  (void)fix;
#endif
}

// --- qr: a third-order input, one and two row bonds ---

template <typename TenT>
void test_qr_higher_order(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_QR) && defined(TCICT_HAS_ELEM_SWEEP)
#ifdef TCICT_SKIP_QR_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  factor_check::on_dense_third_order(
      fix, [&](const TenT &d, const auto &at, std::size_t k) {
        factor_check::check_qr(fix, d, k, at);
      });
#else
  (void)fix;
#endif
}

// --- lq: non-square matrices ---

template <typename TenT> void test_lq_non_square(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_LQ) && defined(TCICT_HAS_ELEM_SWEEP)
#ifdef TCICT_SKIP_LQ_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  factor_check::on_wide_and_tall(fix, [&](const TenT &m, const auto &at) {
    factor_check::check_lq(fix, m, 1, at);
  });
#else
  (void)fix;
#endif
}

// --- lq: a third-order input, one and two row bonds ---

template <typename TenT>
void test_lq_higher_order(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_LQ) && defined(TCICT_HAS_ELEM_SWEEP)
#ifdef TCICT_SKIP_LQ_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  factor_check::on_dense_third_order(
      fix, [&](const TenT &d, const auto &at, std::size_t k) {
        factor_check::check_lq(fix, d, k, at);
      });
#else
  (void)fix;
#endif
}

// --- eigvals: diagonal matrix ---

template <typename TenT>
void test_eigvals_diagonal(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_EIGVALS
#ifdef TCICT_SKIP_EIGVALS_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::factorization);

  // diag(1, 2, 3) → eigenvalues {1, 2, 3}
  auto matrix = tci::zeros<TenT>(ctx, {3, 3});
  tci::set_elem(ctx, matrix, {0, 0}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, matrix, {1, 1}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, matrix, {2, 2}, make_elem<TenT>(3.0));

  tci::cplx_ten_t<TenT> eigenvalues;
  tci::eigvals(ctx, matrix, 1, eigenvalues);

  // V1 keeps eigvals' result a first-order list, unlike eig's lambda_mat.
  TCICT_ASSERT(tci::order(ctx, eigenvalues) == 1);
  TCICT_ASSERT(tci::size(ctx, eigenvalues) == 3);

  // Collect eigenvalues and sort by real part for order-independent comparison
  using CplxTenT = tci::cplx_ten_t<TenT>;
  std::vector<double> ev_real(3);
  for (tci::elem_coor_t<CplxTenT> i = 0; i < 3; ++i) {
    ev_real[i] = real_part<CplxTenT>(tci::get_elem(ctx, eigenvalues, {i}));
    // A NaN breaks the strict weak ordering std::sort requires; fail here
    // instead of sorting with one.
    TCICT_ASSERT(!std::isnan(ev_real[i]));
    // Imaginary parts should be ~0 for real diagonal matrix
    TCICT_ASSERT_CLOSE(
        imag_part<CplxTenT>(tci::get_elem(ctx, eigenvalues, {i})), 0.0, tol);
  }
  std::sort(ev_real.begin(), ev_real.end());
  TCICT_ASSERT_CLOSE(ev_real[0], 1.0, tol);
  TCICT_ASSERT_CLOSE(ev_real[1], 2.0, tol);
  TCICT_ASSERT_CLOSE(ev_real[2], 3.0, tol);
#else
  (void)fix;
#endif
}

// --- eigvalsh: symmetric matrix ---

template <typename TenT>
void test_eigvalsh_diagonal(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_EIGVALSH
#ifdef TCICT_SKIP_EIGVALSH_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::factorization);

  // diag(1, 2, 3) → eigenvalues {1, 2, 3} (real, ascending)
  auto matrix = tci::zeros<TenT>(ctx, {3, 3});
  tci::set_elem(ctx, matrix, {0, 0}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, matrix, {1, 1}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, matrix, {2, 2}, make_elem<TenT>(3.0));

  tci::real_ten_t<TenT> eigenvalues;
  tci::eigvalsh(ctx, matrix, 1, eigenvalues);

  // V1 keeps eigvalsh's result a first-order list, unlike eigh's lambda_mat.
  TCICT_ASSERT(tci::order(ctx, eigenvalues) == 1);
  TCICT_ASSERT(tci::size(ctx, eigenvalues) == 3);

  using RealTenT = tci::real_ten_t<TenT>;
  TCICT_ASSERT_CLOSE(real_part<RealTenT>(tci::get_elem(ctx, eigenvalues, {0})),
                     1.0, tol);
  TCICT_ASSERT_CLOSE(real_part<RealTenT>(tci::get_elem(ctx, eigenvalues, {1})),
                     2.0, tol);
  TCICT_ASSERT_CLOSE(real_part<RealTenT>(tci::get_elem(ctx, eigenvalues, {2})),
                     3.0, tol);
#else
  (void)fix;
#endif
}

} // namespace tests
} // namespace tcict

// Bulk registration helper: invokes X(..., "category", test_fn) once per test.
// See include/tcict/adapters/doctest.h for usage.
#define TCICT_FOREACH_LINEAR_ALGEBRA_TEST_ALL_TYPES(X, ...) \
  X(__VA_ARGS__, "linear_algebra", test_norm_identity) \
  X(__VA_ARGS__, "linear_algebra", test_linear_combine_uniform) \
  X(__VA_ARGS__, "linear_algebra", test_linear_combine_weighted) \
  X(__VA_ARGS__, "linear_algebra", test_linear_combine_single) \
  X(__VA_ARGS__, "linear_algebra", test_linear_combine_zeroth_order) \
  X(__VA_ARGS__, "linear_algebra", test_normalize_inplace) \
  X(__VA_ARGS__, "linear_algebra", test_normalize_outofplace) \
  X(__VA_ARGS__, "linear_algebra", test_normalize_edge_cases) \
  X(__VA_ARGS__, "linear_algebra", test_normalize_zeroth_order) \
  X(__VA_ARGS__, "linear_algebra", test_norm_2x2) \
  X(__VA_ARGS__, "linear_algebra", test_norm_zeroth_order) \
  X(__VA_ARGS__, "linear_algebra", test_contract_matmul) \
  X(__VA_ARGS__, "linear_algebra", test_contract_dot_product) \
  X(__VA_ARGS__, "linear_algebra", test_contract_dot_product_int_labels) \
  X(__VA_ARGS__, "linear_algebra", test_contract_zeroth_order) \
  X(__VA_ARGS__, "linear_algebra", test_contract_outer_product) \
  X(__VA_ARGS__, "linear_algebra", test_qr) \
  X(__VA_ARGS__, "linear_algebra", test_lq) \
  X(__VA_ARGS__, "linear_algebra", test_qr_non_square) \
  X(__VA_ARGS__, "linear_algebra", test_qr_higher_order) \
  X(__VA_ARGS__, "linear_algebra", test_lq_non_square) \
  X(__VA_ARGS__, "linear_algebra", test_lq_higher_order) \
  X(__VA_ARGS__, "linear_algebra", test_trunc_svd) \
  X(__VA_ARGS__, "linear_algebra", test_trunc_svd_trunc_err_value) \
  X(__VA_ARGS__, "linear_algebra", test_trunc_svd_trunc_err_no_truncation) \
  X(__VA_ARGS__, "linear_algebra", test_trunc_svd_trunc_err_bounded) \
  X(__VA_ARGS__, "linear_algebra", test_trunc_svd_non_diagonal_input) \
  X(__VA_ARGS__, "linear_algebra", test_trunc_svd_target_err_selects_chi) \
  X(__VA_ARGS__, "linear_algebra", test_trunc_svd_target_err_scale_invariant) \
  X(__VA_ARGS__, "linear_algebra", test_trunc_svd_target_err_chi_min_floor) \
  X(__VA_ARGS__, "linear_algebra", test_trunc_svd_target_err_chi_max_cap) \
  X(__VA_ARGS__, "linear_algebra", test_trunc_svd_target_err_zero_matches_chi_max_overload) \
  X(__VA_ARGS__, "linear_algebra", test_trunc_svd_chi_min_not_restored_below_s_min) \
  X(__VA_ARGS__, "linear_algebra", test_trunc_svd_target_err_boundary_is_inclusive) \
  X(__VA_ARGS__, "linear_algebra", test_trunc_svd_argument_order) \
  X(__VA_ARGS__, "linear_algebra", test_eig_identity) \
  X(__VA_ARGS__, "linear_algebra", test_eig_distinct) \
  X(__VA_ARGS__, "linear_algebra", test_eigh_identity) \
  X(__VA_ARGS__, "linear_algebra", test_eigh_ascending) \
  X(__VA_ARGS__, "linear_algebra", test_exp_identity) \
  X(__VA_ARGS__, "linear_algebra", test_exp_diagonal) \
  X(__VA_ARGS__, "linear_algebra", test_exp_zero) \
  X(__VA_ARGS__, "linear_algebra", test_exp_anti_hermitian) \
  X(__VA_ARGS__, "linear_algebra", test_inverse) \
  X(__VA_ARGS__, "linear_algebra", test_scale_inplace) \
  X(__VA_ARGS__, "linear_algebra", test_scale_outofplace) \
  X(__VA_ARGS__, "linear_algebra", test_scale_by_zero) \
  X(__VA_ARGS__, "linear_algebra", test_scale_zeroth_order) \
  X(__VA_ARGS__, "linear_algebra", test_trace_partial) \
  X(__VA_ARGS__, "linear_algebra", test_trace_leaving_bond) \
  X(__VA_ARGS__, "linear_algebra", test_svd_basic) \
  X(__VA_ARGS__, "linear_algebra", test_svd_reconstruction) \
  X(__VA_ARGS__, "linear_algebra", test_svd_non_square) \
  X(__VA_ARGS__, "linear_algebra", test_svd_higher_order) \
  X(__VA_ARGS__, "linear_algebra", test_eigvals_diagonal) \
  X(__VA_ARGS__, "linear_algebra", test_eigvalsh_diagonal)
