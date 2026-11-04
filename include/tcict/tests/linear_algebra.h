#pragma once

#include <tcict/assertion.h>
#include <tcict/elem_helper.h>
#include <tcict/elem_sweep.h>
#include <tcict/fixture.h>
#include <tcict/skip.h>
#include <tcict/zeroth_order.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
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
#if !defined(TCICT_SKIP_LINEAR_COMBINE) && defined(TCICT_HAS_ELEM_SWEEP)
  auto &ctx = fix.context();
  auto tensor_a = tci::zeros<TenT>(ctx, {});
  auto tensor_b = tci::zeros<TenT>(ctx, {});
  TCICT_SET_ELEMS(fix, tensor_a, [](const tci::elem_coors_t<TenT> &) { return 1.25; });
  TCICT_SET_ELEMS(fix, tensor_b, [](const tci::elem_coors_t<TenT> &) { return -3.5; });

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
#if !defined(TCICT_SKIP_NORMALIZE) && defined(TCICT_HAS_ELEM_SWEEP)
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::reduction);

  auto tensor = tci::zeros<TenT>(ctx, {});
  TCICT_SET_ELEMS(fix, tensor, [](const tci::elem_coors_t<TenT> &) { return -2.0; });
  const auto w = written<TenT>(-2.0);
  const auto original = tci::normalize(ctx, tensor);
  TCICT_ASSERT_CLOSE(original, std::abs(w), tol);
  TCICT_EXPECT_ZEROTH_ORDER(fix, tensor, w / std::abs(w), tol_category::reduction);

  // Out of place; the input is read back, see "Out-of-place inputs" in
  // CONTRIBUTING.md.
  auto input = tci::zeros<TenT>(ctx, {});
  TCICT_SET_ELEMS(fix, input, [](const tci::elem_coors_t<TenT> &) { return 3.0; });
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
  TCICT_SET_ELEMS(fix, tensor, [](const tci::elem_coors_t<TenT> &) { return -2.0; });
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
#if !defined(TCICT_SKIP_CONTRACT) && defined(TCICT_HAS_ELEM_SWEEP)
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
                            tol_category::reduction);
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
#if !defined(TCICT_SKIP_CONTRACT) && defined(TCICT_HAS_ELEM_SWEEP)
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
  TCICT_EXPECT_ZEROTH_ORDER(fix, c, dot, tol_category::reduction);
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
#if !defined(TCICT_SKIP_CONTRACT) && defined(TCICT_HAS_ELEM_SWEEP)
  auto &ctx = fix.context();
  auto v_at = ramp_over<TenT>({3}, 1.0);
  auto s = tci::zeros<TenT>(ctx, {});
  auto t = tci::zeros<TenT>(ctx, {});
  auto v = tci::zeros<TenT>(ctx, {3});
  TCICT_SET_ELEMS(fix, s, [](const tci::elem_coors_t<TenT> &) { return 1.5; });
  TCICT_SET_ELEMS(fix, t, [](const tci::elem_coors_t<TenT> &) { return -2.0; });
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

#ifndef TCICT_SKIP_TRUNC_SVD
// The singular values the trunc_svd fixture matrix is built from. Shared with
// trunc_svd_expected_epsilon below so the matrix and the expected values cannot
// drift apart.
inline constexpr double kTruncSvdFixtureSvs[] = {3.0, 2.0, 1.0, 0.1};
inline constexpr int kTruncSvdFixtureRank =
    static_cast<int>(sizeof(kTruncSvdFixtureSvs) / sizeof(kTruncSvdFixtureSvs[0]));

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
// Helper: run overload (2) on the fixture spectrum and assert the retained chi
// and the truncation error against the spec's epsilon for that chi.
//
// The strategy tests below differ only in the four strategy arguments, the
// fixture scale, and the expected chi, so the call and its two assertions live
// here once. They stay separate test functions: the registration macro expands
// one consumer test case per function, and merging them would collapse the
// per-case reporting and skip granularity that buys.
//
// Guarded with the tests it serves, and for the same reason the fixture builder
// above is: it calls both of them, so leaving it outside would break the header
// for a backend that defines TCICT_SKIP_TRUNC_SVD.
template <typename TenT>
void trunc_svd_expect_retained_chi(tci_test_fixture<TenT> &fix, double scale,
                                   int chi_min, int chi_max, double target_trunc_err,
                                   double s_min, int expected_chi) {
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::factorization);
  auto matrix = trunc_svd_test_matrix<TenT>(ctx, scale);

  TenT u, v_dag;
  tci::real_ten_t<TenT> s_diag;
  tci::real_t<TenT> trunc_err = -1.0;

  tci::trunc_svd(ctx, matrix, 1, u, s_diag, v_dag, trunc_err,
                 static_cast<tci::bond_dim_t<TenT>>(chi_min),
                 static_cast<tci::bond_dim_t<TenT>>(chi_max),
                 static_cast<tci::real_t<TenT>>(target_trunc_err),
                 static_cast<tci::real_t<TenT>>(s_min));

  // bond_dim_t is unsigned on some backends, so comparing it against the int
  // parameter warns under -Wsign-compare; the exemption the neighbouring
  // literal comparisons rely on covers only non-negative constants.
  auto s_shape = tci::shape(ctx, s_diag);
  TCICT_ASSERT(s_shape[0] == static_cast<tci::bond_dim_t<TenT>>(expected_chi));
  TCICT_ASSERT_CLOSE(trunc_err, trunc_svd_expected_epsilon<TenT>(expected_chi), tol);
}
#endif

template <typename TenT> void test_trunc_svd(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_TRUNC_SVD
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

  auto s_shape = tci::shape(ctx, s_diag);
  TCICT_ASSERT(s_shape.size() == 1);
  TCICT_ASSERT(s_shape[0] <= 2);
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
#ifndef TCICT_SKIP_TRUNC_SVD
#ifdef TCICT_SKIP_TRUNC_SVD_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  auto matrix = trunc_svd_test_matrix<TenT>(ctx);

  TenT u, v_dag;
  tci::real_ten_t<TenT> s_diag;
  tci::real_t<TenT> trunc_err = -1.0;

  tci::trunc_svd(ctx, matrix, 1, u, s_diag, v_dag, trunc_err,
                 static_cast<tci::bond_dim_t<TenT>>(2), 0.0);

  TCICT_ASSERT_CLOSE(trunc_err, trunc_svd_expected_epsilon<TenT>(2), eps);
#else
  (void)fix;
#endif
}

/// Verify trunc_err == 0 when no truncation occurs (chi_max >= kappa).
template <typename TenT>
void test_trunc_svd_trunc_err_no_truncation(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_TRUNC_SVD
#ifdef TCICT_SKIP_TRUNC_SVD_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  auto matrix = trunc_svd_test_matrix<TenT>(ctx);

  TenT u, v_dag;
  tci::real_ten_t<TenT> s_diag;
  tci::real_t<TenT> trunc_err = -1.0;

  // chi_max = 10 > 4 (kappa), s_min = 0 → keep all SVs
  tci::trunc_svd(ctx, matrix, 1, u, s_diag, v_dag, trunc_err,
                 static_cast<tci::bond_dim_t<TenT>>(10), 0.0);

  auto s_shape = tci::shape(ctx, s_diag);
  TCICT_ASSERT(s_shape.size() == 1);
  TCICT_ASSERT(s_shape[0] == 4);
  TCICT_ASSERT_CLOSE(trunc_err, 0.0, eps);
#else
  (void)fix;
#endif
}

/// Verify trunc_err is in [0, 1] (relative error is bounded).
template <typename TenT>
void test_trunc_svd_trunc_err_bounded(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_TRUNC_SVD
#ifdef TCICT_SKIP_TRUNC_SVD_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  auto matrix = trunc_svd_test_matrix<TenT>(ctx);

  TenT u, v_dag;
  tci::real_ten_t<TenT> s_diag;
  tci::real_t<TenT> trunc_err = -1.0;

  // Keep only 1 SV: discard [2, 1, 0.1]
  tci::trunc_svd(ctx, matrix, 1, u, s_diag, v_dag, trunc_err,
                 static_cast<tci::bond_dim_t<TenT>>(1), 0.0);

  TCICT_ASSERT(trunc_err >= 0.0);
  TCICT_ASSERT(trunc_err <= 1.0);

  TCICT_ASSERT_CLOSE(trunc_err, trunc_svd_expected_epsilon<TenT>(1), eps);
#else
  (void)fix;
#endif
}

// --- truncated SVD, overload (2): chi_min / chi_max / target_trunc_err / s_min ---
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
//
// These tests read the retained chi as shape(s_diag)[0], which is the retained
// chi whether s_diag is first-order or the second-order {chi, chi} diagonal
// tensor V1 defines (as sigma), so that read survives the representation
// migration the suite has not made yet.
//
// One exception, called out because the rest of the block would otherwise imply
// it does not exist: test_trunc_svd_target_err_zero_matches_chi_max_overload
// compares singular values elementwise with a single coordinate, which is only
// valid while s_diag is first-order. That read has to move when sigma does.

/// Verify target_trunc_err selects an interior chi via the epsilon ladder.
/// epsilon(2) ≈ 0.07209 <= 0.1 < epsilon(1) ≈ 0.3576, so chi must be 2.
template <typename TenT>
void test_trunc_svd_target_err_selects_chi(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_TRUNC_SVD
#ifdef TCICT_SKIP_TRUNC_SVD_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  trunc_svd_expect_retained_chi<TenT>(fix, /*scale=*/1.0, /*chi_min=*/1, /*chi_max=*/4,
                                     /*target_trunc_err=*/0.1, /*s_min=*/0.0,
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
#ifndef TCICT_SKIP_TRUNC_SVD
#ifdef TCICT_SKIP_TRUNC_SVD_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  trunc_svd_expect_retained_chi<TenT>(fix, /*scale=*/1.0e-3, /*chi_min=*/1, /*chi_max=*/4,
                                     /*target_trunc_err=*/0.1, /*s_min=*/0.0,
                                     /*expected_chi=*/2);
#else
  (void)fix;
#endif
}

/// Verify chi_min floors the selection. epsilon(1) ≈ 0.3576 <= 0.5 would allow
/// chi = 1, but chi_min = 3 forbids it.
template <typename TenT>
void test_trunc_svd_target_err_chi_min_floor(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_TRUNC_SVD
#ifdef TCICT_SKIP_TRUNC_SVD_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  trunc_svd_expect_retained_chi<TenT>(fix, /*scale=*/1.0, /*chi_min=*/3, /*chi_max=*/4,
                                     /*target_trunc_err=*/0.5, /*s_min=*/0.0,
                                     /*expected_chi=*/3);
#else
  (void)fix;
#endif
}

/// Verify chi_max caps the selection. No chi <= 2 reaches epsilon <= 1e-6, so
/// the growth stops at chi_max rather than continuing to satisfy the target.
template <typename TenT>
void test_trunc_svd_target_err_chi_max_cap(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_TRUNC_SVD
#ifdef TCICT_SKIP_TRUNC_SVD_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  trunc_svd_expect_retained_chi<TenT>(fix, /*scale=*/1.0, /*chi_min=*/1, /*chi_max=*/2,
                                     /*target_trunc_err=*/1.0e-6, /*s_min=*/0.0,
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
#ifndef TCICT_SKIP_TRUNC_SVD
#ifdef TCICT_SKIP_TRUNC_SVD_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::factorization);

  TenT u_general, v_dag_general;
  tci::real_ten_t<TenT> s_general;
  tci::real_t<TenT> err_general = -1.0;
  auto matrix_general = trunc_svd_test_matrix<TenT>(ctx);
  tci::trunc_svd(ctx, matrix_general, 1, u_general, s_general, v_dag_general, err_general,
                 static_cast<tci::bond_dim_t<TenT>>(1),
                 static_cast<tci::bond_dim_t<TenT>>(2),
                 static_cast<tci::real_t<TenT>>(0.0), static_cast<tci::real_t<TenT>>(0.0));

  TenT u_simple, v_dag_simple;
  tci::real_ten_t<TenT> s_simple;
  tci::real_t<TenT> err_simple = -1.0;
  auto matrix_simple = trunc_svd_test_matrix<TenT>(ctx);
  tci::trunc_svd(ctx, matrix_simple, 1, u_simple, s_simple, v_dag_simple, err_simple,
                 static_cast<tci::bond_dim_t<TenT>>(2), static_cast<tci::real_t<TenT>>(0.0));

  // No chi below chi_max = 2 reaches epsilon <= 0, so both calls must stop at
  // 2 and report epsilon(2). Assert that against the spec first: it is what
  // the differential assertions below cannot supply on their own.
  auto shape_general = tci::shape(ctx, s_general);
  auto shape_simple = tci::shape(ctx, s_simple);
  TCICT_ASSERT(shape_general[0] == static_cast<tci::bond_dim_t<TenT>>(2));
  TCICT_ASSERT_CLOSE(err_general, trunc_svd_expected_epsilon<TenT>(2), tol);

  // Equivalence is a claim about the whole decomposition, so compare every
  // output the two calls produce, not just the retained count.
  TCICT_ASSERT(shape_general[0] == shape_simple[0]);
  TCICT_ASSERT_CLOSE(err_general, err_simple, tol);

  for (tci::bond_dim_t<TenT> i = 0; i < shape_general[0]; ++i) {
    const auto coor = static_cast<tci::elem_coor_t<tci::real_ten_t<TenT>>>(i);
    TCICT_ASSERT_CLOSE(tci::get_elem(ctx, s_general, {coor}),
                       tci::get_elem(ctx, s_simple, {coor}), tol);
  }

  // The factors are compared through the product they reconstruct rather than
  // elementwise: for a spectrum with distinct singular values the columns of u
  // are fixed only up to a phase, which the two calls are not obliged to choose
  // identically, so an elementwise comparison would assert more than the
  // equivalence claim. The reconstruction is invariant under that phase, and
  // unlike a shape comparison it does read the factors' values.
  using RealTenT = tci::real_ten_t<TenT>;
  auto reconstruct = [&ctx](const TenT &u_in, const RealTenT &s_in, const TenT &v_in) {
    const auto u_shape = tci::shape(ctx, u_in);
    auto scaled = tci::copy(ctx, u_in);
    for (tci::bond_dim_t<TenT> j = 0; j < u_shape[1]; ++j) {
      const auto sj = real_part<RealTenT>(
          tci::get_elem(ctx, s_in, {static_cast<tci::elem_coor_t<RealTenT>>(j)}));
      for (tci::bond_dim_t<TenT> i = 0; i < u_shape[0]; ++i) {
        const auto ci = static_cast<tci::elem_coor_t<TenT>>(i);
        const auto cj = static_cast<tci::elem_coor_t<TenT>>(j);
        const auto elem = tci::get_elem(ctx, u_in, {ci, cj});
        tci::set_elem(ctx, scaled, {ci, cj},
                      make_elem<TenT>(real_part<TenT>(elem) * sj,
                                      imag_part<TenT>(elem) * sj));
      }
    }
    TenT product;
    tci::contract(ctx, scaled, "ik", v_in, "kj", product, "ij");
    return product;
  };

  auto reconstructed_general = reconstruct(u_general, s_general, v_dag_general);
  auto reconstructed_simple = reconstruct(u_simple, s_simple, v_dag_simple);
  TCICT_ASSERT(tci::close(ctx, reconstructed_general, reconstructed_simple, tol));
#else
  (void)fix;
#endif
}

/// Verify that singular values discarded by s_min are not restored to satisfy
/// chi_min. s_min = 0.5 leaves survivors [3, 2, 1], and chi_min = 4 cannot be
/// met, so chi must be 3 rather than 4.
template <typename TenT>
void test_trunc_svd_chi_min_not_restored_below_s_min(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_TRUNC_SVD
#ifdef TCICT_SKIP_TRUNC_SVD_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  trunc_svd_expect_retained_chi<TenT>(fix, /*scale=*/1.0, /*chi_min=*/4, /*chi_max=*/4,
                                     /*target_trunc_err=*/0.0, /*s_min=*/0.5,
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
#ifndef TCICT_SKIP_TRUNC_SVD
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

  auto pinned_shape = tci::shape(ctx, s_pinned);
  TCICT_ASSERT(pinned_shape[0] == static_cast<tci::bond_dim_t<TenT>>(1));
  TCICT_ASSERT_CLOSE(epsilon_at_one, trunc_svd_expected_epsilon_from<TenT>(kSvs, kRank, 1),
                     tol);

  // Now let the same epsilon arrive as the target. chi = 1 is admissible under
  // `<=` and inadmissible under `<`.
  TenT u, v_dag;
  tci::real_ten_t<TenT> s_diag;
  tci::real_t<TenT> trunc_err = -1.0;
  auto matrix = build();
  tci::trunc_svd(ctx, matrix, 1, u, s_diag, v_dag, trunc_err,
                 static_cast<tci::bond_dim_t<TenT>>(1),
                 static_cast<tci::bond_dim_t<TenT>>(3), epsilon_at_one,
                 static_cast<tci::real_t<TenT>>(0.0));

  auto s_shape = tci::shape(ctx, s_diag);
  TCICT_ASSERT(s_shape[0] == static_cast<tci::bond_dim_t<TenT>>(1));
#else
  (void)fix;
#endif
}

// --- eig (general eigendecomposition of identity) ---

template <typename TenT> void test_eig_identity(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_EIG
#ifdef TCICT_SKIP_EIG_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  auto matrix = tci::eye<TenT>(ctx, 2);

  using CplxTenT = tci::cplx_ten_t<TenT>;
  CplxTenT eigenvals, eigenvecs;
  tci::eig(ctx, matrix, 1, eigenvals, eigenvecs);

  TCICT_ASSERT(tci::order(ctx, eigenvals) == 1);
  TCICT_ASSERT(tci::size(ctx, eigenvals) == 2);
  TCICT_ASSERT_CLOSE(real_part<CplxTenT>(tci::get_elem(ctx, eigenvals, {0})),
                     1.0, eps);
  TCICT_ASSERT_CLOSE(real_part<CplxTenT>(tci::get_elem(ctx, eigenvals, {1})),
                     1.0, eps);

  TCICT_ASSERT(tci::order(ctx, eigenvecs) == 2);
#else
  (void)fix;
#endif
}

// --- eigh (Hermitian eigendecomposition of identity) ---

template <typename TenT> void test_eigh_identity(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_EIGH
#ifdef TCICT_SKIP_EIGH_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  auto matrix = tci::eye<TenT>(ctx, 2);

  tci::real_ten_t<TenT> eigenvals;
  TenT eigenvecs;
  tci::eigh(ctx, matrix, 1, eigenvals, eigenvecs);

  TCICT_ASSERT(tci::order(ctx, eigenvals) == 1);
  TCICT_ASSERT(tci::size(ctx, eigenvals) == 2);

  using RealTenT = tci::real_ten_t<TenT>;
  TCICT_ASSERT_CLOSE(real_part<RealTenT>(tci::get_elem(ctx, eigenvals, {0})),
                     1.0, eps);
  TCICT_ASSERT_CLOSE(real_part<RealTenT>(tci::get_elem(ctx, eigenvals, {1})),
                     1.0, eps);
  TCICT_ASSERT(tci::order(ctx, eigenvecs) == 2);
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
#if !defined(TCICT_SKIP_SCALE) && defined(TCICT_HAS_ELEM_SWEEP)
  auto &ctx = fix.context();
  const std::complex<double> s(-1.5, is_complex_v<TenT> ? 0.5 : 0.0);
  const auto s_elem = make_elem<TenT>(s.real(), s.imag());

  auto tensor = tci::zeros<TenT>(ctx, {});
  TCICT_SET_ELEMS(fix, tensor, [](const tci::elem_coors_t<TenT> &) { return 2.0; });
  tci::scale(ctx, tensor, s_elem);
  TCICT_EXPECT_ZEROTH_ORDER(fix, tensor, written<TenT>(2.0) * s,
                            tol_category::elementwise);

  // Out of place; the input is read back, see "Out-of-place inputs" in
  // CONTRIBUTING.md.
  auto input = tci::zeros<TenT>(ctx, {});
  TCICT_SET_ELEMS(fix, input, [](const tci::elem_coors_t<TenT> &) { return -0.5; });
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
#if !defined(TCICT_SKIP_TRACE) && defined(TCICT_HAS_ELEM_SWEEP)
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
#ifndef TCICT_SKIP_SVD
#ifdef TCICT_SKIP_SVD_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();

  // Diagonal matrix with known singular values [3, 1]
  auto matrix = tci::zeros<TenT>(ctx, {2, 2});
  tci::set_elem(ctx, matrix, {0, 0}, make_elem<TenT>(3.0));
  tci::set_elem(ctx, matrix, {1, 1}, make_elem<TenT>(1.0));

  TenT u, v_dag;
  tci::real_ten_t<TenT> s_diag;
  tci::svd(ctx, matrix, 1, u, s_diag, v_dag);

  // S should have 2 singular values
  TCICT_ASSERT(tci::size(ctx, s_diag) == 2);

  // Singular values should be non-negative and sorted descending
  using RealTenT = tci::real_ten_t<TenT>;
  auto s0 = real_part<RealTenT>(tci::get_elem(ctx, s_diag, {0}));
  auto s1 = real_part<RealTenT>(tci::get_elem(ctx, s_diag, {1}));
  TCICT_ASSERT(s0 >= s1);
  TCICT_ASSERT(s1 >= 0.0);
  TCICT_ASSERT_CLOSE(s0, 3.0, eps);
  TCICT_ASSERT_CLOSE(s1, 1.0, eps);

  // U shape: (2, 2), V† shape: (2, 2)
  auto u_shape = tci::shape(ctx, u);
  auto v_shape = tci::shape(ctx, v_dag);
  TCICT_ASSERT(u_shape.size() == 2);
  TCICT_ASSERT(u_shape[0] == 2);
  TCICT_ASSERT(v_shape.size() == 2);
  TCICT_ASSERT(v_shape[1] == 2);
#else
  (void)fix;
#endif
}

// --- svd: reconstruction U * diag(S) * V† ≈ A ---

template <typename TenT>
void test_svd_reconstruction(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_SVD
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

  // Reconstruct: scale columns of U by S, then contract with V†
  // u_scaled[i,j] = u[i,j] * s[j]
  auto bond = tci::size(ctx, s_diag);
  using RealTenT = tci::real_ten_t<TenT>;
  auto u_scaled = tci::copy(ctx, u);
  for (tci::elem_coor_t<TenT> j = 0; j < bond; ++j) {
    auto sj = real_part<RealTenT>(tci::get_elem(ctx, s_diag, {j}));
    for (tci::elem_coor_t<TenT> i = 0; i < 2; ++i) {
      auto elem = tci::get_elem(ctx, u, {i, j});
      tci::set_elem(ctx, u_scaled, {i, j},
                    make_elem<TenT>(real_part<TenT>(elem) * sj,
                                    imag_part<TenT>(elem) * sj));
    }
  }

  // Reconstruct via contract: u_scaled[i,k] * v_dag[k,j] → reconstructed[i,j]
  TenT reconstructed;
  tci::contract(ctx, u_scaled, "ik", v_dag, "kj", reconstructed, "ij");

  TCICT_ASSERT(tci::close(ctx, reconstructed, matrix,
                          tolerance(fix, tol_category::factorization)));
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
  auto eps = fix.epsilon();

  // diag(1, 2, 3) → eigenvalues {1, 2, 3}
  auto matrix = tci::zeros<TenT>(ctx, {3, 3});
  tci::set_elem(ctx, matrix, {0, 0}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, matrix, {1, 1}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, matrix, {2, 2}, make_elem<TenT>(3.0));

  tci::cplx_ten_t<TenT> eigenvalues;
  tci::eigvals(ctx, matrix, 1, eigenvalues);

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
        imag_part<CplxTenT>(tci::get_elem(ctx, eigenvalues, {i})), 0.0, eps);
  }
  std::sort(ev_real.begin(), ev_real.end());
  TCICT_ASSERT_CLOSE(ev_real[0], 1.0, eps);
  TCICT_ASSERT_CLOSE(ev_real[1], 2.0, eps);
  TCICT_ASSERT_CLOSE(ev_real[2], 3.0, eps);
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
  auto eps = fix.epsilon();

  // diag(1, 2, 3) → eigenvalues {1, 2, 3} (real, ascending)
  auto matrix = tci::zeros<TenT>(ctx, {3, 3});
  tci::set_elem(ctx, matrix, {0, 0}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, matrix, {1, 1}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, matrix, {2, 2}, make_elem<TenT>(3.0));

  tci::real_ten_t<TenT> eigenvalues;
  tci::eigvalsh(ctx, matrix, 1, eigenvalues);

  TCICT_ASSERT(tci::size(ctx, eigenvalues) == 3);

  using RealTenT = tci::real_ten_t<TenT>;
  TCICT_ASSERT_CLOSE(real_part<RealTenT>(tci::get_elem(ctx, eigenvalues, {0})),
                     1.0, eps);
  TCICT_ASSERT_CLOSE(real_part<RealTenT>(tci::get_elem(ctx, eigenvalues, {1})),
                     2.0, eps);
  TCICT_ASSERT_CLOSE(real_part<RealTenT>(tci::get_elem(ctx, eigenvalues, {2})),
                     3.0, eps);
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
  X(__VA_ARGS__, "linear_algebra", test_trunc_svd) \
  X(__VA_ARGS__, "linear_algebra", test_trunc_svd_trunc_err_value) \
  X(__VA_ARGS__, "linear_algebra", test_trunc_svd_trunc_err_no_truncation) \
  X(__VA_ARGS__, "linear_algebra", test_trunc_svd_trunc_err_bounded) \
  X(__VA_ARGS__, "linear_algebra", test_trunc_svd_target_err_selects_chi) \
  X(__VA_ARGS__, "linear_algebra", test_trunc_svd_target_err_scale_invariant) \
  X(__VA_ARGS__, "linear_algebra", test_trunc_svd_target_err_chi_min_floor) \
  X(__VA_ARGS__, "linear_algebra", test_trunc_svd_target_err_chi_max_cap) \
  X(__VA_ARGS__, "linear_algebra", test_trunc_svd_target_err_zero_matches_chi_max_overload) \
  X(__VA_ARGS__, "linear_algebra", test_trunc_svd_chi_min_not_restored_below_s_min) \
  X(__VA_ARGS__, "linear_algebra", test_trunc_svd_target_err_boundary_is_inclusive) \
  X(__VA_ARGS__, "linear_algebra", test_eig_identity) \
  X(__VA_ARGS__, "linear_algebra", test_eigh_identity) \
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
  X(__VA_ARGS__, "linear_algebra", test_eigvals_diagonal) \
  X(__VA_ARGS__, "linear_algebra", test_eigvalsh_diagonal)
