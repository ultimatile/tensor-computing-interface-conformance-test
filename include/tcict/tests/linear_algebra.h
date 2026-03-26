#pragma once

#include <tcict/assertion.h>
#include <tcict/elem_helper.h>
#include <tcict/fixture.h>
#include <tcict/skip.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace tcict { namespace tests {

// --- norm (basic: identity matrix) ---

template <typename TenT>
void test_norm_identity(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_NORM
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  auto identity = tci::eye<TenT>(ctx, 3);

  auto norm_val = tci::norm(ctx, identity);
  // Frobenius norm of 3x3 identity = sqrt(3)
  TCICT_ASSERT_CLOSE(norm_val, std::sqrt(3.0), eps);
}

// --- linear_combine: uniform coefficients ---

template <typename TenT>
void test_linear_combine_uniform(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_LINEAR_COMBINE
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();

  TenT result;
  auto tensor_a = tci::zeros<TenT>(ctx, {2, 2});
  auto tensor_b = tci::zeros<TenT>(ctx, {2, 2});
  auto tensor_c = tci::zeros<TenT>(ctx, {2, 2});

  // a = [[1,2],[3,4]], b = [[5,6],[7,8]], c = [[1,1],[1,1]]
  tci::set_elem(ctx, tensor_a, {0, 0}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, tensor_a, {0, 1}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, tensor_a, {1, 0}, make_elem<TenT>(3.0));
  tci::set_elem(ctx, tensor_a, {1, 1}, make_elem<TenT>(4.0));
  tci::set_elem(ctx, tensor_b, {0, 0}, make_elem<TenT>(5.0));
  tci::set_elem(ctx, tensor_b, {0, 1}, make_elem<TenT>(6.0));
  tci::set_elem(ctx, tensor_b, {1, 0}, make_elem<TenT>(7.0));
  tci::set_elem(ctx, tensor_b, {1, 1}, make_elem<TenT>(8.0));
  tci::set_elem(ctx, tensor_c, {0, 0}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, tensor_c, {0, 1}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, tensor_c, {1, 0}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, tensor_c, {1, 1}, make_elem<TenT>(1.0));

  tci::List<TenT> tensors = {tensor_a, tensor_b, tensor_c};
  TCICT_ASSERT_NOTHROW(result = tci::linear_combine(ctx, tensors));

  // Expected: [[7,9],[11,13]]
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 0})), 7.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 1})), 9.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {1, 0})), 11.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {1, 1})), 13.0, eps);
}

// --- linear_combine: weighted coefficients ---

template <typename TenT>
void test_linear_combine_weighted(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_LINEAR_COMBINE
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();

  TenT result;
  auto tensor_a = tci::zeros<TenT>(ctx, {2, 2});
  auto tensor_b = tci::zeros<TenT>(ctx, {2, 2});

  // a = [[2,4],[6,8]], b = [[1,3],[5,7]]
  tci::set_elem(ctx, tensor_a, {0, 0}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, tensor_a, {0, 1}, make_elem<TenT>(4.0));
  tci::set_elem(ctx, tensor_a, {1, 0}, make_elem<TenT>(6.0));
  tci::set_elem(ctx, tensor_a, {1, 1}, make_elem<TenT>(8.0));
  tci::set_elem(ctx, tensor_b, {0, 0}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, tensor_b, {0, 1}, make_elem<TenT>(3.0));
  tci::set_elem(ctx, tensor_b, {1, 0}, make_elem<TenT>(5.0));
  tci::set_elem(ctx, tensor_b, {1, 1}, make_elem<TenT>(7.0));

  tci::List<TenT> tensors = {tensor_a, tensor_b};
  tci::List<tci::elem_t<TenT>> coefficients = {make_elem<TenT>(0.5), make_elem<TenT>(2.0)};
  TCICT_ASSERT_NOTHROW(result = tci::linear_combine(ctx, tensors, coefficients));

  // Expected: 0.5*a + 2*b = [[3,8],[13,18]]
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 0})), 3.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 1})), 8.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {1, 0})), 13.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {1, 1})), 18.0, eps);
}

// --- linear_combine: single tensor ---

template <typename TenT>
void test_linear_combine_single(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_LINEAR_COMBINE
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();

  TenT result;
  auto single_tensor = tci::zeros<TenT>(ctx, {1, 1});
  tci::set_elem(ctx, single_tensor, {0, 0}, make_elem<TenT>(5.0));

  tci::List<TenT> single_list = {single_tensor};
  TCICT_ASSERT_NOTHROW(result = tci::linear_combine(ctx, single_list));
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 0})), 5.0, eps);

  tci::List<tci::elem_t<TenT>> single_coef = {make_elem<TenT>(3.0)};
  TCICT_ASSERT_NOTHROW(result = tci::linear_combine(ctx, single_list, single_coef));
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 0})), 15.0, eps);
}

// --- normalize: in-place ---

template <typename TenT>
void test_normalize_inplace(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_NORMALIZE
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();

  auto tensor = tci::zeros<TenT>(ctx, {2, 2});
  // [[3,4],[0,0]] -> norm = 5
  tci::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(3.0));
  tci::set_elem(ctx, tensor, {0, 1}, make_elem<TenT>(4.0));

  auto original_norm = tci::normalize(ctx, tensor);
  TCICT_ASSERT_CLOSE(std::abs(original_norm), 5.0, eps);

  // Verify normalized values
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 0})), 0.6, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 1})), 0.8, eps);

  // New norm should be 1
  auto new_norm = tci::norm(ctx, tensor);
  TCICT_ASSERT_CLOSE(new_norm, 1.0, eps);
}

// --- normalize: out-of-place ---

template <typename TenT>
void test_normalize_outofplace(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_NORMALIZE
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();

  TenT normalized;
  auto original = tci::zeros<TenT>(ctx, {3, 1});
  // [[2],[2],[1]] -> norm = 3
  tci::set_elem(ctx, original, {0, 0}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, original, {1, 0}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, original, {2, 0}, make_elem<TenT>(1.0));

  auto original_norm = tci::normalize(ctx, original, normalized);
  TCICT_ASSERT_CLOSE(std::abs(original_norm), 3.0, eps);

  // Original unchanged
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, original, {0, 0})), 2.0, eps);

  // Normalized tensor
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, normalized, {0, 0})), 2.0 / 3.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, normalized, {2, 0})), 1.0 / 3.0, eps);

  auto new_norm = tci::norm(ctx, normalized);
  TCICT_ASSERT_CLOSE(new_norm, 1.0, eps);
}

// --- normalize: edge cases ---

template <typename TenT>
void test_normalize_edge_cases(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_NORMALIZE
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();

  // Single non-zero element
  auto single_elem = tci::zeros<TenT>(ctx, {2, 2});
  tci::set_elem(ctx, single_elem, {1, 1}, make_elem<TenT>(7.0));

  auto norm1 = tci::normalize(ctx, single_elem);
  TCICT_ASSERT_CLOSE(std::abs(norm1), 7.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, single_elem, {1, 1})), 1.0, eps);

  // Zero tensor
  auto zero_tensor = tci::zeros<TenT>(ctx, {2, 2});
  auto norm_zero = tci::normalize(ctx, zero_tensor);
  TCICT_ASSERT_CLOSE(std::abs(norm_zero), 0.0, eps);
}

// --- norm: 2x2 identity ---

template <typename TenT>
void test_norm_2x2(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_NORM
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  auto identity = tci::eye<TenT>(ctx, 2);
  auto norm_val = tci::norm(ctx, identity);
  TCICT_ASSERT_CLOSE(norm_val, std::sqrt(2.0), eps);
}

// --- contract: matrix multiplication via Einstein notation ---

template <typename TenT>
void test_contract_matmul(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_CONTRACT
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  TenT c;
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

  tci::contract(ctx, a, "ij", b, "jk", c, "ik");

  // A*B = [[19,22],[43,50]]
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, c, {0, 0})), 19.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, c, {0, 1})), 22.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, c, {1, 0})), 43.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, c, {1, 1})), 50.0, eps);
}

// --- contract: dot product ---

template <typename TenT>
void test_contract_dot_product(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_CONTRACT
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  TenT c;
  auto a = tci::zeros<TenT>(ctx, {3});
  auto b = tci::zeros<TenT>(ctx, {3});

  // a = [1,2,3], b = [4,5,6]
  tci::set_elem(ctx, a, {0}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, a, {1}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, a, {2}, make_elem<TenT>(3.0));
  tci::set_elem(ctx, b, {0}, make_elem<TenT>(4.0));
  tci::set_elem(ctx, b, {1}, make_elem<TenT>(5.0));
  tci::set_elem(ctx, b, {2}, make_elem<TenT>(6.0));

  tci::contract(ctx, a, "i", b, "i", c, "");

  // dot = 1*4 + 2*5 + 3*6 = 32
  auto c_shape = tci::shape(ctx, c);
  TCICT_ASSERT(c_shape.size() == 1);
  TCICT_ASSERT(c_shape[0] == 1);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, c, {0})), 32.0, eps);
}

// --- contract: outer product ---

template <typename TenT>
void test_contract_outer_product(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_CONTRACT
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  TenT c;
  auto a = tci::zeros<TenT>(ctx, {2});
  auto b = tci::zeros<TenT>(ctx, {3});

  tci::set_elem(ctx, a, {0}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, a, {1}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, b, {0}, make_elem<TenT>(3.0));
  tci::set_elem(ctx, b, {1}, make_elem<TenT>(4.0));
  tci::set_elem(ctx, b, {2}, make_elem<TenT>(5.0));

  tci::contract(ctx, a, "i", b, "j", c, "ij");

  auto c_shape = tci::shape(ctx, c);
  TCICT_ASSERT(c_shape[0] == 2);
  TCICT_ASSERT(c_shape[1] == 3);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, c, {0, 0})), 3.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, c, {1, 2})), 10.0, eps);
}

// --- QR decomposition ---

template <typename TenT>
void test_qr(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_QR
  return;
#endif
  auto& ctx = fix.context();
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
}

// --- LQ decomposition ---

template <typename TenT>
void test_lq(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_LQ
  return;
#endif
  auto& ctx = fix.context();
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
}

// --- truncated SVD ---

template <typename TenT>
void test_trunc_svd(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_TRUNC_SVD
  return;
#endif
  auto& ctx = fix.context();
  auto matrix = tci::zeros<TenT>(ctx, {4, 4});
  // Diagonal with known singular values [3,2,1,0.1]
  for (int i = 0; i < 4; ++i)
    tci::set_elem(ctx, matrix,
                  {static_cast<tci::elem_coor_t<TenT>>(i),
                   static_cast<tci::elem_coor_t<TenT>>(i)},
                  make_elem<TenT>(3.0 - i * 0.9667));  // approx [3,2.1,1.1,0.1]

  // Use exact diagonal values for cleaner test
  tci::set_elem(ctx, matrix, {0, 0}, make_elem<TenT>(3.0));
  tci::set_elem(ctx, matrix, {1, 1}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, matrix, {2, 2}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, matrix, {3, 3}, make_elem<TenT>(0.1));

  TenT u, v_dag;
  tci::real_ten_t<TenT> s_diag;
  tci::real_t<TenT> trunc_err;

  TCICT_ASSERT_NOTHROW(tci::trunc_svd(
      ctx, matrix, 1, u, s_diag, v_dag, trunc_err,
      static_cast<tci::bond_dim_t<TenT>>(2), 0.5));

  auto s_shape = tci::shape(ctx, s_diag);
  TCICT_ASSERT(s_shape.size() == 1);
  TCICT_ASSERT(s_shape[0] <= 2);
  TCICT_ASSERT(trunc_err >= 0.0);
}

// --- eig (general eigendecomposition of identity) ---

template <typename TenT>
void test_eig_identity(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_EIG
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  auto matrix = tci::eye<TenT>(ctx, 2);

  TenT eigenvals, eigenvecs;
  tci::eig(ctx, matrix, 1, eigenvals, eigenvecs);

  TCICT_ASSERT(tci::order(ctx, eigenvals) == 1);
  TCICT_ASSERT(tci::size(ctx, eigenvals) == 2);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, eigenvals, {0})), 1.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, eigenvals, {1})), 1.0, eps);

  TCICT_ASSERT(tci::order(ctx, eigenvecs) == 2);
}

// --- eigh (Hermitian eigendecomposition of identity) ---

template <typename TenT>
void test_eigh_identity(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_EIGH
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  auto matrix = tci::eye<TenT>(ctx, 2);

  tci::real_ten_t<TenT> eigenvals;
  TenT eigenvecs;
  tci::eigh(ctx, matrix, 1, eigenvals, eigenvecs);

  TCICT_ASSERT(tci::order(ctx, eigenvals) == 1);
  TCICT_ASSERT(tci::size(ctx, eigenvals) == 2);

  using RealTenT = tci::real_ten_t<TenT>;
  TCICT_ASSERT_CLOSE(real_part<RealTenT>(tci::get_elem(ctx, eigenvals, {0})), 1.0, eps);
  TCICT_ASSERT_CLOSE(real_part<RealTenT>(tci::get_elem(ctx, eigenvals, {1})), 1.0, eps);
  TCICT_ASSERT(tci::order(ctx, eigenvecs) == 2);
}

// --- exp: identity matrix ---

template <typename TenT>
void test_exp_identity(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_EXP
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  auto identity = tci::eye<TenT>(ctx, 3);

  TenT result;
  tci::exp(ctx, identity, 1, result);

  // exp(I) = e*I
  double expected_e = std::exp(1.0);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {1, 1})), expected_e, eps);
  TCICT_ASSERT_CLOSE(std::abs(tci::get_elem(ctx, result, {0, 1})), 0.0, eps);
}

// --- exp: diagonal matrix ---

template <typename TenT>
void test_exp_diagonal(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_EXP
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  auto diagonal = tci::zeros<TenT>(ctx, {2, 2});
  tci::set_elem(ctx, diagonal, {0, 0}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, diagonal, {1, 1}, make_elem<TenT>(2.0));

  TenT result;
  tci::exp(ctx, diagonal, 1, result);

  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 0})), std::exp(1.0), eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {1, 1})), std::exp(2.0), eps);
  TCICT_ASSERT_CLOSE(std::abs(tci::get_elem(ctx, result, {0, 1})), 0.0, eps);
  TCICT_ASSERT_CLOSE(std::abs(tci::get_elem(ctx, result, {1, 0})), 0.0, eps);
}

// --- exp: zero matrix → identity ---

template <typename TenT>
void test_exp_zero(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_EXP
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  auto zero_matrix = tci::zeros<TenT>(ctx, {2, 2});

  TenT result;
  tci::exp(ctx, zero_matrix, 1, result);

  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 0})), 1.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {1, 1})), 1.0, eps);
  TCICT_ASSERT_CLOSE(std::abs(tci::get_elem(ctx, result, {0, 1})), 0.0, eps);
}

// --- exp: anti-Hermitian → unitary ---

template <typename TenT>
void test_exp_anti_hermitian(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_EXP
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  auto anti_herm = tci::zeros<TenT>(ctx, {2, 2});
  tci::set_elem(ctx, anti_herm, {0, 1}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, anti_herm, {1, 0}, make_elem<TenT>(-1.0));

  TenT result;
  tci::exp(ctx, anti_herm, 1, result);

  // exp(anti-Hermitian) should be unitary: column norms = 1
  auto e00 = tci::get_elem(ctx, result, {0, 0});
  auto e10 = tci::get_elem(ctx, result, {1, 0});
  double col0_norm_sq = std::norm(e00) + std::norm(e10);
  TCICT_ASSERT_CLOSE(col0_norm_sq, 1.0, eps);
}

// --- exp: error conditions ---

template <typename TenT>
void test_exp_errors(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_EXP
  return;
#endif
  auto& ctx = fix.context();
  TenT result;
  auto non_square = tci::zeros<TenT>(ctx, {2, 3});
  TCICT_ASSERT_THROWS(std::invalid_argument, tci::exp(ctx, non_square, 1, result));

  auto square = tci::zeros<TenT>(ctx, {2, 2});
  TCICT_ASSERT_THROWS(std::invalid_argument, tci::exp(ctx, square, 3, result));
}

// --- inverse ---

template <typename TenT>
void test_inverse(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_INVERSE
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  auto matrix = tci::zeros<TenT>(ctx, {2, 2});
  // [[4,7],[2,6]] → inv = [[0.6,-0.7],[-0.2,0.4]]
  tci::set_elem(ctx, matrix, {0, 0}, make_elem<TenT>(4.0));
  tci::set_elem(ctx, matrix, {0, 1}, make_elem<TenT>(7.0));
  tci::set_elem(ctx, matrix, {1, 0}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, matrix, {1, 1}, make_elem<TenT>(6.0));

  TenT inv;
  TCICT_ASSERT_NOTHROW(tci::inverse(ctx, matrix, 1, inv));

  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, inv, {0, 0})), 0.6, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, inv, {0, 1})), -0.7, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, inv, {1, 0})), -0.2, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, inv, {1, 1})), 0.4, eps);
}

// --- inverse: non-square error ---

template <typename TenT>
void test_inverse_errors(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_INVERSE
  return;
#endif
  auto& ctx = fix.context();
  TenT result;
  auto non_square = tci::zeros<TenT>(ctx, {2, 3});
  TCICT_ASSERT_THROWS(std::invalid_argument, tci::inverse(ctx, non_square, 1, result));
}

// --- scale: in-place ---

template <typename TenT>
void test_scale_inplace(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_SCALE
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();

  auto tensor = tci::zeros<TenT>(ctx, {2, 2});
  tci::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, tensor, {0, 1}, make_elem<TenT>(4.0));
  tci::set_elem(ctx, tensor, {1, 0}, make_elem<TenT>(6.0));
  tci::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(8.0));

  tci::scale(ctx, tensor, make_elem<TenT>(0.5));

  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 0})), 1.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 1})), 2.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {1, 0})), 3.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {1, 1})), 4.0, eps);
}

// --- scale: out-of-place ---

template <typename TenT>
void test_scale_outofplace(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_SCALE
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();

  TenT result;
  auto tensor = tci::zeros<TenT>(ctx, {2, 2});
  tci::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(3.0));
  tci::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(6.0));

  tci::scale(ctx, tensor, make_elem<TenT>(-2.0), result);

  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 0})), -6.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {1, 1})), -12.0, eps);
  // Original unchanged
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 0})), 3.0, eps);
}

// --- scale: by zero ---

template <typename TenT>
void test_scale_by_zero(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_SCALE
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();

  auto tensor = tci::eye<TenT>(ctx, 2);
  tci::scale(ctx, tensor, make_elem<TenT>(0.0));

  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 0})), 0.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {1, 1})), 0.0, eps);
}

// --- trace: 2x2 matrix ---

template <typename TenT>
void test_trace_partial(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_TRACE
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();

  // 2x2 diagonal matrix [[1,0],[0,4]] → trace over {0,1} = 1+4 = 5
  auto matrix = tci::zeros<TenT>(ctx, {2, 2});
  tci::set_elem(ctx, matrix, {0, 0}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, matrix, {1, 1}, make_elem<TenT>(4.0));

  TenT result;
  tci::trace(ctx, matrix, {{0, 1}}, result);

  // TCI spec: all bonds paired yields a scalar (order 0)
  TCICT_ASSERT(tci::order(ctx, result) == 0);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {})), 5.0, eps);
}

// --- svd: basic singular values and shapes ---

template <typename TenT>
void test_svd_basic(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_SVD
  return;
#endif
  auto& ctx = fix.context();
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
}

// --- svd: reconstruction U * diag(S) * V† ≈ A ---

template <typename TenT>
void test_svd_reconstruction(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_SVD
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();

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

  TCICT_ASSERT(tci::close(ctx, reconstructed, matrix, eps * 100));
}

// --- eigvals: diagonal matrix ---

template <typename TenT>
void test_eigvals_diagonal(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_EIGVALS
  return;
#endif
  auto& ctx = fix.context();
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
    // Imaginary parts should be ~0 for real diagonal matrix
    TCICT_ASSERT_CLOSE(imag_part<CplxTenT>(tci::get_elem(ctx, eigenvalues, {i})), 0.0, eps);
  }
  std::sort(ev_real.begin(), ev_real.end());
  TCICT_ASSERT_CLOSE(ev_real[0], 1.0, eps);
  TCICT_ASSERT_CLOSE(ev_real[1], 2.0, eps);
  TCICT_ASSERT_CLOSE(ev_real[2], 3.0, eps);
}

// --- eigvals: error on non-square ---

template <typename TenT>
void test_eigvals_errors(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_EIGVALS
  return;
#endif
  auto& ctx = fix.context();
  auto non_square = tci::zeros<TenT>(ctx, {2, 3});
  tci::cplx_ten_t<TenT> w;
  TCICT_ASSERT_THROWS(std::invalid_argument, tci::eigvals(ctx, non_square, 1, w));
}

// --- eigvalsh: symmetric matrix ---

template <typename TenT>
void test_eigvalsh_diagonal(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_EIGVALSH
  return;
#endif
  auto& ctx = fix.context();
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
  TCICT_ASSERT_CLOSE(real_part<RealTenT>(tci::get_elem(ctx, eigenvalues, {0})), 1.0, eps);
  TCICT_ASSERT_CLOSE(real_part<RealTenT>(tci::get_elem(ctx, eigenvalues, {1})), 2.0, eps);
  TCICT_ASSERT_CLOSE(real_part<RealTenT>(tci::get_elem(ctx, eigenvalues, {2})), 3.0, eps);
}

// --- eigvalsh: error on non-square ---

template <typename TenT>
void test_eigvalsh_errors(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_EIGVALSH
  return;
#endif
  auto& ctx = fix.context();
  auto non_square = tci::zeros<TenT>(ctx, {2, 3});
  tci::real_ten_t<TenT> w;
  TCICT_ASSERT_THROWS(std::invalid_argument, tci::eigvalsh(ctx, non_square, 1, w));
}

}}  // namespace tcict::tests
