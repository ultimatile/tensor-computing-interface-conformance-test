#pragma once

#include <tcict/assertion.h>
#include <tcict/elem_helper.h>
#include <tcict/fixture.h>
#include <tcict/skip.h>

#include <algorithm>
#include <cmath>
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
#ifndef TCICT_SKIP_LINEAR_COMBINE
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::reduction, 3);

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

  TenT result;
  tci::List<TenT> tensors = {tensor_a, tensor_b, tensor_c};
  TCICT_ASSERT_NOTHROW(result = tci::linear_combine(ctx, tensors));

  // Expected: [[7,9],[11,13]]
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 0})), 7.0,
                     tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 1})), 9.0,
                     tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {1, 0})), 11.0,
                     tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {1, 1})), 13.0,
                     tol);
#else
  (void)fix;
#endif
}

// --- linear_combine: weighted coefficients ---

template <typename TenT>
void test_linear_combine_weighted(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_LINEAR_COMBINE
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::reduction, 2);

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
  tci::List<tci::elem_t<TenT>> coefficients = {make_elem<TenT>(0.5),
                                               make_elem<TenT>(2.0)};

  TenT result;
  TCICT_ASSERT_NOTHROW(result =
                           tci::linear_combine(ctx, tensors, coefficients));

  // Expected: 0.5*a + 2*b = [[3,8],[13,18]]
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 0})), 3.0,
                     tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 1})), 8.0,
                     tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {1, 0})), 13.0,
                     tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {1, 1})), 18.0,
                     tol);
#else
  (void)fix;
#endif
}

// --- linear_combine: single tensor ---

template <typename TenT>
void test_linear_combine_single(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_LINEAR_COMBINE
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);

  auto single_tensor = tci::zeros<TenT>(ctx, {1, 1});
  tci::set_elem(ctx, single_tensor, {0, 0}, make_elem<TenT>(5.0));

  TenT result;
  tci::List<TenT> single_list = {single_tensor};
  TCICT_ASSERT_NOTHROW(result = tci::linear_combine(ctx, single_list));
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 0})), 5.0,
                     tol);

  tci::List<tci::elem_t<TenT>> single_coef = {make_elem<TenT>(3.0)};
  TCICT_ASSERT_NOTHROW(result =
                           tci::linear_combine(ctx, single_list, single_coef));
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 0})), 15.0,
                     tol);
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

  // Original unchanged
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
#ifndef TCICT_SKIP_CONTRACT
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::reduction, 3);
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

  // dot = 1*4 + 2*5 + 3*6 = 32
  auto c_shape = tci::shape(ctx, c);
  TCICT_ASSERT(c_shape.size() == 1);
  TCICT_ASSERT(c_shape[0] == 1);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, c, {0})), 32.0, tol);
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
  // bond_dim_t / elem_coor_t are backend-defined and may be unsigned, so the
  // loop counter cannot be spelled `int` — a narrowing conversion inside a
  // braced initializer is ill-formed for a non-constant expression.
  const auto rank = static_cast<tci::bond_dim_t<TenT>>(kTruncSvdFixtureRank);
  auto matrix = tci::zeros<TenT>(ctx, {rank, rank});
  for (int i = 0; i < kTruncSvdFixtureRank; ++i) {
    const auto coor = static_cast<tci::elem_coor_t<TenT>>(i);
    tci::set_elem(ctx, matrix, {coor, coor},
                  make_elem<TenT>(kTruncSvdFixtureSvs[i] * scale));
  }
  return matrix;
}

// Helper: the spec's relative truncation error for keeping `chi` of the
// fixture's singular values,
//
//   epsilon(chi) = sum_{i>=chi} s_i^2 / sum_{i<kappa} s_i^2.
//
// Evaluated here from the fixture spectrum rather than by calling back into the
// backend, so an assertion using it compares the implementation against the
// specification and not against a second route through the same code. Takes no
// scale: epsilon is a ratio of sums of squares and so is scale-invariant.
template <typename TenT> tci::real_t<TenT> trunc_svd_expected_epsilon(int chi) {
  double total = 0.0;
  double discarded = 0.0;
  for (int i = 0; i < kTruncSvdFixtureRank; ++i) {
    const double s2 = kTruncSvdFixtureSvs[i] * kTruncSvdFixtureSvs[i];
    total += s2;
    if (i >= chi) {
      discarded += s2;
    }
  }
  return static_cast<tci::real_t<TenT>>(discarded / total);
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
//   1. Discard all s_i < s_min.
//   2. Among the survivors retain at least chi_min when possible; values below
//      s_min are NOT restored to satisfy chi_min.
//   3. Grow chi in descending order until epsilon <= target_trunc_err or
//      chi == chi_max.
//
// Step 2 is what keeps the retained chi from being simply "the smallest chi in
// [chi_min, chi_max] meeting the target": when fewer than chi_min values
// survive step 1, the result is below chi_min.
//
// For the [3, 2, 1, 0.1] fixture, sum s_i^2 = 14.01 and the ladder runs
// epsilon(4) = 0, epsilon(3) = 0.01/14.01 ≈ 7.138e-4,
// epsilon(2) = 1.01/14.01 ≈ 0.07209, epsilon(1) = 5.01/14.01 ≈ 0.3576. The
// targets below sit well clear of these, so the tests do not depend on how the
// quotients round; the one test that does need an exact quotient chooses a
// different spectrum and asserts the exactness it needs.
//
// These tests read the retained chi as shape(s_diag)[0] and deliberately assert
// nothing about s_diag's order. V1 defines it (as sigma) to be a second-order
// {chi, chi} diagonal tensor while the suite elsewhere still asserts the
// first-order representation; shape[0] is the retained chi under either, so
// these tests do not need revisiting when that representation is migrated.

/// Verify target_trunc_err selects an interior chi via the epsilon ladder.
/// epsilon(2) ≈ 0.07209 <= 0.1 < epsilon(1) ≈ 0.3576, so chi must be 2.
template <typename TenT>
void test_trunc_svd_target_err_selects_chi(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_TRUNC_SVD
#ifdef TCICT_SKIP_TRUNC_SVD_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::factorization);
  auto matrix = trunc_svd_test_matrix<TenT>(ctx);

  TenT u, v_dag;
  tci::real_ten_t<TenT> s_diag;
  tci::real_t<TenT> trunc_err = -1.0;

  tci::trunc_svd(ctx, matrix, 1, u, s_diag, v_dag, trunc_err,
                 static_cast<tci::bond_dim_t<TenT>>(1),
                 static_cast<tci::bond_dim_t<TenT>>(4),
                 static_cast<tci::real_t<TenT>>(0.1), static_cast<tci::real_t<TenT>>(0.0));

  auto s_shape = tci::shape(ctx, s_diag);
  TCICT_ASSERT(s_shape[0] == 2);
  TCICT_ASSERT_CLOSE(trunc_err, trunc_svd_expected_epsilon<TenT>(2), tol);
#else
  (void)fix;
#endif
}

/// Verify the target_trunc_err decision is invariant under a uniform rescaling
/// of the input. epsilon is a ratio of sums of squares, so scaling every
/// singular value by 1e-3 must not move the retained chi. An implementation
/// that compares raw singular values against target_trunc_err fails here even
/// when it happens to agree at scale 1.
template <typename TenT>
void test_trunc_svd_target_err_scale_invariant(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_TRUNC_SVD
#ifdef TCICT_SKIP_TRUNC_SVD_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::factorization);
  auto matrix = trunc_svd_test_matrix<TenT>(ctx, 1.0e-3);

  TenT u, v_dag;
  tci::real_ten_t<TenT> s_diag;
  tci::real_t<TenT> trunc_err = -1.0;

  tci::trunc_svd(ctx, matrix, 1, u, s_diag, v_dag, trunc_err,
                 static_cast<tci::bond_dim_t<TenT>>(1),
                 static_cast<tci::bond_dim_t<TenT>>(4),
                 static_cast<tci::real_t<TenT>>(0.1), static_cast<tci::real_t<TenT>>(0.0));

  auto s_shape = tci::shape(ctx, s_diag);
  TCICT_ASSERT(s_shape[0] == 2);
  TCICT_ASSERT_CLOSE(trunc_err, trunc_svd_expected_epsilon<TenT>(2), tol);
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
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::factorization);
  auto matrix = trunc_svd_test_matrix<TenT>(ctx);

  TenT u, v_dag;
  tci::real_ten_t<TenT> s_diag;
  tci::real_t<TenT> trunc_err = -1.0;

  tci::trunc_svd(ctx, matrix, 1, u, s_diag, v_dag, trunc_err,
                 static_cast<tci::bond_dim_t<TenT>>(3),
                 static_cast<tci::bond_dim_t<TenT>>(4),
                 static_cast<tci::real_t<TenT>>(0.5), static_cast<tci::real_t<TenT>>(0.0));

  auto s_shape = tci::shape(ctx, s_diag);
  TCICT_ASSERT(s_shape[0] == 3);
  TCICT_ASSERT_CLOSE(trunc_err, trunc_svd_expected_epsilon<TenT>(3), tol);
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
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::factorization);
  auto matrix = trunc_svd_test_matrix<TenT>(ctx);

  TenT u, v_dag;
  tci::real_ten_t<TenT> s_diag;
  tci::real_t<TenT> trunc_err = -1.0;

  tci::trunc_svd(ctx, matrix, 1, u, s_diag, v_dag, trunc_err,
                 static_cast<tci::bond_dim_t<TenT>>(1),
                 static_cast<tci::bond_dim_t<TenT>>(2),
                 static_cast<tci::real_t<TenT>>(1.0e-6), static_cast<tci::real_t<TenT>>(0.0));

  auto s_shape = tci::shape(ctx, s_diag);
  TCICT_ASSERT(s_shape[0] == 2);
  TCICT_ASSERT_CLOSE(trunc_err, trunc_svd_expected_epsilon<TenT>(2), tol);
#else
  (void)fix;
#endif
}

/// Verify overload (2) with chi_min = 1 and target_trunc_err = 0 reproduces
/// overload (1), which V1 defines as exactly that specialization.
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

  // Equivalence is a claim about the whole decomposition, so compare every
  // output the two calls produce, not just the retained count.
  auto shape_general = tci::shape(ctx, s_general);
  auto shape_simple = tci::shape(ctx, s_simple);
  TCICT_ASSERT(shape_general[0] == shape_simple[0]);
  TCICT_ASSERT_CLOSE(err_general, err_simple, tol);

  for (tci::bond_dim_t<TenT> i = 0; i < shape_general[0]; ++i) {
    const auto coor = static_cast<tci::elem_coor_t<tci::real_ten_t<TenT>>>(i);
    TCICT_ASSERT_CLOSE(tci::get_elem(ctx, s_general, {coor}),
                       tci::get_elem(ctx, s_simple, {coor}), tol);
  }

  auto shape_u_general = tci::shape(ctx, u_general);
  auto shape_u_simple = tci::shape(ctx, u_simple);
  TCICT_ASSERT(shape_u_general == shape_u_simple);
  auto shape_v_general = tci::shape(ctx, v_dag_general);
  auto shape_v_simple = tci::shape(ctx, v_dag_simple);
  TCICT_ASSERT(shape_v_general == shape_v_simple);
#else
  (void)fix;
#endif
}

/// Verify step 2's exclusion: singular values discarded by s_min are not
/// restored to satisfy chi_min. s_min = 0.5 leaves survivors [3, 2, 1], and
/// chi_min = 4 cannot be met, so chi must be 3 rather than 4.
template <typename TenT>
void test_trunc_svd_chi_min_not_restored_below_s_min(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_TRUNC_SVD
#ifdef TCICT_SKIP_TRUNC_SVD_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  auto matrix = trunc_svd_test_matrix<TenT>(ctx);

  TenT u, v_dag;
  tci::real_ten_t<TenT> s_diag;
  tci::real_t<TenT> trunc_err = -1.0;

  tci::trunc_svd(ctx, matrix, 1, u, s_diag, v_dag, trunc_err,
                 static_cast<tci::bond_dim_t<TenT>>(4),
                 static_cast<tci::bond_dim_t<TenT>>(4),
                 static_cast<tci::real_t<TenT>>(0.0), static_cast<tci::real_t<TenT>>(0.5));

  auto s_shape = tci::shape(ctx, s_diag);
  TCICT_ASSERT(s_shape[0] == 3);
#else
  (void)fix;
#endif
}

/// Verify the comparison in step 3 is `epsilon <= target_trunc_err`, not `<`.
///
/// Distinguishing the two needs a target that lands exactly on an achievable
/// epsilon. SVs [5, 4, 3] give sum s_i^2 = 50 and a discarded weight of
/// 16 + 9 = 25 at chi = 1, so epsilon(1) = 25 / 50, a quotient of integers that
/// is a power of two. An implementation using `<` cannot take chi = 1 and
/// returns chi = 2 instead.
///
/// The tie only exists if the backend reaches that quotient without rounding,
/// which nothing in the spec obliges it to do. So the test asserts it: the
/// returned trunc_err must equal 0.5 by exact comparison. A backend that
/// computes epsilon through, say, a squared Frobenius norm lands a fraction of
/// an ulp away, and that assertion says so directly instead of letting the
/// retained-chi assertion below fail for an unexplained reason.
template <typename TenT>
void test_trunc_svd_target_err_boundary_is_inclusive(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_TRUNC_SVD
#ifdef TCICT_SKIP_TRUNC_SVD_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  auto matrix = tci::zeros<TenT>(ctx, {3, 3});
  tci::set_elem(ctx, matrix, {0, 0}, make_elem<TenT>(5.0));
  tci::set_elem(ctx, matrix, {1, 1}, make_elem<TenT>(4.0));
  tci::set_elem(ctx, matrix, {2, 2}, make_elem<TenT>(3.0));

  TenT u, v_dag;
  tci::real_ten_t<TenT> s_diag;
  tci::real_t<TenT> trunc_err = -1.0;

  tci::trunc_svd(ctx, matrix, 1, u, s_diag, v_dag, trunc_err,
                 static_cast<tci::bond_dim_t<TenT>>(1),
                 static_cast<tci::bond_dim_t<TenT>>(3),
                 static_cast<tci::real_t<TenT>>(0.5), static_cast<tci::real_t<TenT>>(0.0));

  auto s_shape = tci::shape(ctx, s_diag);
  TCICT_ASSERT(s_shape[0] == 1);
  TCICT_ASSERT(trunc_err == static_cast<tci::real_t<TenT>>(0.5));
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

// --- exp: error conditions ---

template <typename TenT> void test_exp_errors(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_EXP
#ifdef TCICT_SKIP_EXP_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  TenT result;
  auto non_square = tci::zeros<TenT>(ctx, {2, 3});
  TCICT_ASSERT_THROWS(std::invalid_argument,
                      tci::exp(ctx, non_square, 1, result));

  auto square = tci::zeros<TenT>(ctx, {2, 2});
  TCICT_ASSERT_THROWS(std::invalid_argument, tci::exp(ctx, square, 3, result));
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

// --- inverse: non-square error ---

template <typename TenT> void test_inverse_errors(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_INVERSE
#ifdef TCICT_SKIP_INVERSE_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  TenT result;
  auto non_square = tci::zeros<TenT>(ctx, {2, 3});
  TCICT_ASSERT_THROWS(std::invalid_argument,
                      tci::inverse(ctx, non_square, 1, result));
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
  // Original unchanged
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

// --- trace: 2x2 matrix ---

template <typename TenT> void test_trace_partial(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_TRACE
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::reduction, 2);

  // 2x2 diagonal matrix [[1,0],[0,4]] → trace over {0,1} = 1+4 = 5
  auto matrix = tci::zeros<TenT>(ctx, {2, 2});
  tci::set_elem(ctx, matrix, {0, 0}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, matrix, {1, 1}, make_elem<TenT>(4.0));

  TenT result;
  tci::trace(ctx, matrix, {{0, 1}}, result);

  // TCI spec: all bonds paired yields a scalar (order 0)
  TCICT_ASSERT(tci::order(ctx, result) == 0);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {})), 5.0, tol);
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

// --- eigvals: error on non-square ---

template <typename TenT> void test_eigvals_errors(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_EIGVALS
#ifdef TCICT_SKIP_EIGVALS_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  auto non_square = tci::zeros<TenT>(ctx, {2, 3});
  tci::cplx_ten_t<TenT> w;
  TCICT_ASSERT_THROWS(std::invalid_argument,
                      tci::eigvals(ctx, non_square, 1, w));
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

// --- eigvalsh: error on non-square ---

template <typename TenT>
void test_eigvalsh_errors(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_EIGVALSH
#ifdef TCICT_SKIP_EIGVALSH_SINGLE_PRECISION
  TCICT_RETURN_IF_SINGLE_PRECISION;
#endif
  auto &ctx = fix.context();
  auto non_square = tci::zeros<TenT>(ctx, {2, 3});
  tci::real_ten_t<TenT> w;
  TCICT_ASSERT_THROWS(std::invalid_argument,
                      tci::eigvalsh(ctx, non_square, 1, w));
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
  X(__VA_ARGS__, "linear_algebra", test_normalize_inplace) \
  X(__VA_ARGS__, "linear_algebra", test_normalize_outofplace) \
  X(__VA_ARGS__, "linear_algebra", test_normalize_edge_cases) \
  X(__VA_ARGS__, "linear_algebra", test_norm_2x2) \
  X(__VA_ARGS__, "linear_algebra", test_contract_matmul) \
  X(__VA_ARGS__, "linear_algebra", test_contract_dot_product) \
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
  X(__VA_ARGS__, "linear_algebra", test_exp_errors) \
  X(__VA_ARGS__, "linear_algebra", test_inverse) \
  X(__VA_ARGS__, "linear_algebra", test_inverse_errors) \
  X(__VA_ARGS__, "linear_algebra", test_scale_inplace) \
  X(__VA_ARGS__, "linear_algebra", test_scale_outofplace) \
  X(__VA_ARGS__, "linear_algebra", test_scale_by_zero) \
  X(__VA_ARGS__, "linear_algebra", test_trace_partial) \
  X(__VA_ARGS__, "linear_algebra", test_svd_basic) \
  X(__VA_ARGS__, "linear_algebra", test_svd_reconstruction) \
  X(__VA_ARGS__, "linear_algebra", test_eigvals_diagonal) \
  X(__VA_ARGS__, "linear_algebra", test_eigvals_errors) \
  X(__VA_ARGS__, "linear_algebra", test_eigvalsh_diagonal) \
  X(__VA_ARGS__, "linear_algebra", test_eigvalsh_errors)
