#pragma once

#include <tcict/assertion.h>
#include <tcict/elem_helper.h>
#include <tcict/fixture.h>
#include <tcict/skip.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace tcict {
namespace tests {

// --- shrink (in-place) ---

template <typename TenT> void test_shrink_inplace(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_SHRINK
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto tensor = tci::zeros<TenT>(ctx, {3, 3});

  // Fill with values 1-9
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      tci::set_elem(ctx, tensor,
                    {static_cast<tci::elem_coor_t<TenT>>(i),
                     static_cast<tci::elem_coor_t<TenT>>(j)},
                    make_elem<TenT>(i * 3 + j + 1));
    }
  }

  // Shrink to top-left 2x2
  tci::bond_idx_elem_coor_pair_map<TenT> shrink_map;
  shrink_map[0] = std::make_pair(static_cast<tci::elem_coor_t<TenT>>(0),
                                 static_cast<tci::elem_coor_t<TenT>>(2));
  shrink_map[1] = std::make_pair(static_cast<tci::elem_coor_t<TenT>>(0),
                                 static_cast<tci::elem_coor_t<TenT>>(2));
  tci::shrink(ctx, tensor, shrink_map);

  auto result_shape = tci::shape(ctx, tensor);
  TCICT_ASSERT(result_shape[0] == 2);
  TCICT_ASSERT(result_shape[1] == 2);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 0})), 1.0,
                     tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 1})), 2.0,
                     tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {1, 0})), 4.0,
                     tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {1, 1})), 5.0,
                     tol);
#else
  (void)fix;
#endif
}

// --- shrink (out-of-place) ---

template <typename TenT>
void test_shrink_outofplace(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_SHRINK
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto input = tci::zeros<TenT>(ctx, {4, 4});

  // Set values in center 2x2 region [1:3, 1:3]
  tci::set_elem(ctx, input, {1, 1}, make_elem<TenT>(11.0));
  tci::set_elem(ctx, input, {1, 2}, make_elem<TenT>(12.0));
  tci::set_elem(ctx, input, {2, 1}, make_elem<TenT>(21.0));
  tci::set_elem(ctx, input, {2, 2}, make_elem<TenT>(22.0));

  tci::bond_idx_elem_coor_pair_map<TenT> shrink_map;
  shrink_map[0] = std::make_pair(static_cast<tci::elem_coor_t<TenT>>(1),
                                 static_cast<tci::elem_coor_t<TenT>>(3));
  shrink_map[1] = std::make_pair(static_cast<tci::elem_coor_t<TenT>>(1),
                                 static_cast<tci::elem_coor_t<TenT>>(3));

  TenT output;
  tci::shrink(ctx, input, shrink_map, output);

  auto result_shape = tci::shape(ctx, output);
  TCICT_ASSERT(result_shape[0] == 2);
  TCICT_ASSERT(result_shape[1] == 2);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, output, {0, 0})), 11.0,
                     tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, output, {0, 1})), 12.0,
                     tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, output, {1, 0})), 21.0,
                     tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, output, {1, 1})), 22.0,
                     tol);
#else
  (void)fix;
#endif
}

// --- shrink preserves complex values ---

template <typename TenT>
void test_shrink_complex_values(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_SHRINK
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto tensor = tci::zeros<TenT>(ctx, {3, 3});

  tci::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(1.5, 2.5));
  tci::set_elem(ctx, tensor, {0, 1}, make_elem<TenT>(3.5, 4.5));
  tci::set_elem(ctx, tensor, {1, 0}, make_elem<TenT>(5.5, 6.5));
  tci::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(7.5, 8.5));

  tci::bond_idx_elem_coor_pair_map<TenT> shrink_map;
  shrink_map[0] = std::make_pair(static_cast<tci::elem_coor_t<TenT>>(0),
                                 static_cast<tci::elem_coor_t<TenT>>(2));
  shrink_map[1] = std::make_pair(static_cast<tci::elem_coor_t<TenT>>(0),
                                 static_cast<tci::elem_coor_t<TenT>>(2));

  TenT output;
  tci::shrink(ctx, tensor, shrink_map, output);

  auto e00 = tci::get_elem(ctx, output, {0, 0});
  auto e01 = tci::get_elem(ctx, output, {0, 1});
  TCICT_ASSERT_CLOSE(real_part<TenT>(e00), 1.5, tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(e01), 3.5, tol);
  if constexpr (is_complex_v<TenT>) {
    TCICT_ASSERT_CLOSE(imag_part<TenT>(e00), 2.5, tol);
    TCICT_ASSERT_CLOSE(imag_part<TenT>(e01), 4.5, tol);
  }
#else
  (void)fix;
#endif
}

// --- real extraction (out-of-place) ---

template <typename TenT>
void test_real_extraction(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_REAL
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto tensor = tci::zeros<TenT>(ctx, {2, 2});
  tci::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(3.14, 2.71));
  tci::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(-1.59, 0.58));

  auto real_tensor = tci::real(ctx, tensor);

  using RealTenT = tci::real_ten_t<TenT>;
  auto elem00 = tci::get_elem(ctx, real_tensor, {0, 0});
  auto elem11 = tci::get_elem(ctx, real_tensor, {1, 1});
  TCICT_ASSERT_CLOSE(real_part<RealTenT>(elem00), 3.14, tol);
  TCICT_ASSERT_CLOSE(real_part<RealTenT>(elem11), -1.59, tol);
#else
  (void)fix;
#endif
}

// --- imag extraction (out-of-place) ---

template <typename TenT>
void test_imag_extraction(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_IMAG
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto tensor = tci::zeros<TenT>(ctx, {2, 2});
  tci::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(3.14, 2.71));
  tci::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(-1.59, 0.58));

  auto imag_tensor = tci::imag(ctx, tensor);

  using RealTenT = tci::real_ten_t<TenT>;
  auto elem00 = tci::get_elem(ctx, imag_tensor, {0, 0});
  auto elem11 = tci::get_elem(ctx, imag_tensor, {1, 1});
  if constexpr (is_complex_v<TenT>) {
    // Complex input: imag(tensor) extracts the imaginary parts set above.
    TCICT_ASSERT_CLOSE(real_part<RealTenT>(elem00), 2.71, tol);
    TCICT_ASSERT_CLOSE(real_part<RealTenT>(elem11), 0.58, tol);
  } else {
    // Real input: tci::imag returns a zero tensor per TCI spec.
    TCICT_ASSERT_CLOSE(real_part<RealTenT>(elem00), 0.0, tol);
    TCICT_ASSERT_CLOSE(real_part<RealTenT>(elem11), 0.0, tol);
  }
#else
  (void)fix;
#endif
}

// --- cplx_conj (in-place) ---

template <typename TenT>
void test_cplx_conj_inplace(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_CPLX_CONJ
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto tensor = tci::zeros<TenT>(ctx, {2, 2});
  tci::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(1.0, 2.0));
  tci::set_elem(ctx, tensor, {0, 1}, make_elem<TenT>(-3.0, 4.0));
  tci::set_elem(ctx, tensor, {1, 0}, make_elem<TenT>(5.0, -6.0));
  tci::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(-7.0, -8.0));

  tci::cplx_conj(ctx, tensor);

  // Real parts unchanged for both real and complex (cplx_conj on real tensors
  // is a no-op per TCI spec); imaginary parts only exist for complex.
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 0})), 1.0,
                     tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 1})), -3.0,
                     tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {1, 0})), 5.0,
                     tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {1, 1})), -7.0,
                     tol);
  if constexpr (is_complex_v<TenT>) {
    TCICT_ASSERT_CLOSE(imag_part<TenT>(tci::get_elem(ctx, tensor, {0, 0})), -2.0,
                       tol);
    TCICT_ASSERT_CLOSE(imag_part<TenT>(tci::get_elem(ctx, tensor, {0, 1})), -4.0,
                       tol);
    TCICT_ASSERT_CLOSE(imag_part<TenT>(tci::get_elem(ctx, tensor, {1, 0})), 6.0,
                       tol);
    TCICT_ASSERT_CLOSE(imag_part<TenT>(tci::get_elem(ctx, tensor, {1, 1})), 8.0,
                       tol);
  }
#else
  (void)fix;
#endif
}

// --- cplx_conj (out-of-place) ---

template <typename TenT>
void test_cplx_conj_outofplace(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_CPLX_CONJ
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto input = tci::zeros<TenT>(ctx, {2, 2});
  tci::set_elem(ctx, input, {0, 0}, make_elem<TenT>(3.14, 2.71));
  tci::set_elem(ctx, input, {1, 1}, make_elem<TenT>(-1.41, -1.73));

  TenT output;
  tci::cplx_conj(ctx, input, output);

  // Real parts hold for both real and complex (cplx_conj out-of-place is a
  // deep copy for real tensors per TCI spec); imaginary parts only exist for
  // complex.
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, input, {0, 0})), 3.14,
                     tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, output, {0, 0})), 3.14,
                     tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, output, {1, 1})), -1.41,
                     tol);
  if constexpr (is_complex_v<TenT>) {
    TCICT_ASSERT_CLOSE(imag_part<TenT>(tci::get_elem(ctx, input, {0, 0})), 2.71,
                       tol);
    TCICT_ASSERT_CLOSE(imag_part<TenT>(tci::get_elem(ctx, output, {0, 0})), -2.71,
                       tol);
    TCICT_ASSERT_CLOSE(imag_part<TenT>(tci::get_elem(ctx, output, {1, 1})), 1.73,
                       tol);
  }
#else
  (void)fix;
#endif
}

// --- to_cplx (out-of-place, from real type) ---
// NOTE: TenT here should be a real tensor type (e.g., CytnxTensor<double>)

template <typename RealTenT>
void test_to_cplx_outofplace(tci_test_fixture<RealTenT> &fix) {
#ifndef TCICT_SKIP_TO_CPLX
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  RealTenT real_tensor;
  real_tensor = tci::zeros<RealTenT>(ctx, {2, 2});

  tci::set_elem(ctx, real_tensor, {0, 0},
                static_cast<tci::elem_t<RealTenT>>(1.5));
  tci::set_elem(ctx, real_tensor, {0, 1},
                static_cast<tci::elem_t<RealTenT>>(2.5));
  tci::set_elem(ctx, real_tensor, {1, 0},
                static_cast<tci::elem_t<RealTenT>>(3.5));
  tci::set_elem(ctx, real_tensor, {1, 1},
                static_cast<tci::elem_t<RealTenT>>(4.5));

  auto complex_tensor = tci::to_cplx(ctx, real_tensor);

  using CplxTenT = tci::cplx_ten_t<RealTenT>;
  auto elem00 = tci::get_elem(ctx, complex_tensor, {0, 0});
  auto elem11 = tci::get_elem(ctx, complex_tensor, {1, 1});
  TCICT_ASSERT_CLOSE(real_part<CplxTenT>(elem00), 1.5, tol);
  TCICT_ASSERT_CLOSE(imag_part<CplxTenT>(elem00), 0.0, tol);
  TCICT_ASSERT_CLOSE(real_part<CplxTenT>(elem11), 4.5, tol);
  TCICT_ASSERT_CLOSE(imag_part<CplxTenT>(elem11), 0.0, tol);
#else
  (void)fix;
#endif
}

// --- to_cplx (complex to complex) ---

template <typename TenT>
void test_to_cplx_complex_to_complex(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_TO_CPLX
  // For real TenT, tci::to_cplx returns cplx_ten_t<TenT> whose elements are
  // cplx_t<TenT>; calling imag_part<TenT>(cplx_elem) would be a type error.
  // This test therefore only runs when TenT is already complex.
  if constexpr (is_complex_v<TenT>) {
    auto &ctx = fix.context();
    auto tol = tolerance(fix, tol_category::elementwise);
    auto tensor = tci::zeros<TenT>(ctx, {2, 2});
    tci::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(3.14, 2.71));
    tci::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(-1.41, 1.73));

    auto result = tci::to_cplx(ctx, tensor);

    auto elem00 = tci::get_elem(ctx, result, {0, 0});
    auto elem11 = tci::get_elem(ctx, result, {1, 1});
    TCICT_ASSERT_CLOSE(real_part<TenT>(elem00), 3.14, tol);
    TCICT_ASSERT_CLOSE(imag_part<TenT>(elem00), 2.71, tol);
    TCICT_ASSERT_CLOSE(real_part<TenT>(elem11), -1.41, tol);
    TCICT_ASSERT_CLOSE(imag_part<TenT>(elem11), 1.73, tol);
  }
#else
  (void)fix;
#endif
}

// --- for_each: element doubling ---

template <typename TenT>
void test_for_each_doubling(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_FOR_EACH
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  using Elem = tci::elem_t<TenT>;

  auto tensor = tci::zeros<TenT>(ctx, {2, 3});
  tci::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, tensor, {0, 1}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, tensor, {0, 2}, make_elem<TenT>(3.0));
  tci::set_elem(ctx, tensor, {1, 0}, make_elem<TenT>(4.0));
  tci::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(5.0));
  tci::set_elem(ctx, tensor, {1, 2}, make_elem<TenT>(6.0));

  tci::for_each(ctx, tensor,
                [](Elem &elem) { elem = elem * make_elem<TenT>(2.0); });

  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 0})), 2.0,
                     tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 2})), 6.0,
                     tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {1, 2})), 12.0,
                     tol);
#else
  (void)fix;
#endif
}

// --- for_each: iteration and summation ---

template <typename TenT>
void test_for_each_summation(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_FOR_EACH
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::reduction, 4);
  using Elem = tci::elem_t<TenT>;

  auto tensor = tci::zeros<TenT>(ctx, {2, 2});
  tci::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, tensor, {0, 1}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, tensor, {1, 0}, make_elem<TenT>(3.0));
  tci::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(4.0));

  int count = 0;
  Elem sum = make_elem<TenT>(0.0);
  tci::for_each(ctx, tensor, [&count, &sum](Elem &elem) {
    count++;
    sum = sum + elem;
  });

  TCICT_ASSERT(count == 4);
  TCICT_ASSERT_CLOSE(real_part<TenT>(sum), 10.0, tol);
#else
  (void)fix;
#endif
}

// --- for_each: scalar multiplication with capture ---

template <typename TenT>
void test_for_each_capture(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_FOR_EACH
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  using Elem = tci::elem_t<TenT>;

  auto tensor = tci::fill<TenT>(ctx, {2, 2}, make_elem<TenT>(3.0, 1.0));

  auto multiplier = make_elem<TenT>(0.5);
  tci::for_each(ctx, tensor,
                [multiplier](Elem &elem) { elem = elem * multiplier; });

  auto result = tci::get_elem(ctx, tensor, {0, 0});
  TCICT_ASSERT_CLOSE(real_part<TenT>(result), 1.5, tol);
  if constexpr (is_complex_v<TenT>) {
    TCICT_ASSERT_CLOSE(imag_part<TenT>(result), 0.5, tol);
  }
#else
  (void)fix;
#endif
}

// --- for_each: const version ---

template <typename TenT> void test_for_each_const(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_FOR_EACH
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::reduction, 3);
  using Elem = tci::elem_t<TenT>;

  auto tensor = tci::fill<TenT>(ctx, {3}, make_elem<TenT>(2.0, 3.0));

  Elem sum = make_elem<TenT>(0.0);
  tci::for_each(ctx, static_cast<const TenT &>(tensor),
                [&sum](const Elem &elem) { sum = sum + elem; });

  TCICT_ASSERT_CLOSE(real_part<TenT>(sum), 6.0, tol);
  if constexpr (is_complex_v<TenT>) {
    TCICT_ASSERT_CLOSE(imag_part<TenT>(sum), 9.0, tol);
  }
#else
  (void)fix;
#endif
}

// --- for_each: element-wise inversion ---

template <typename TenT>
void test_for_each_inversion(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_FOR_EACH
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  using Elem = tci::elem_t<TenT>;

  auto tensor = tci::fill<TenT>(ctx, {2, 2}, make_elem<TenT>(0.5));

  tci::for_each(ctx, tensor, [](Elem &elem) {
    if (std::abs(elem) > 1e-12) {
      elem = make_elem<TenT>(1.0, 0.0) / elem;
    }
  });

  auto result = tci::get_elem(ctx, tensor, {0, 0});
  TCICT_ASSERT_CLOSE(real_part<TenT>(result), 2.0, tol);
#else
  (void)fix;
#endif
}

// --- for_each_with_coors: mutable ---

template <typename TenT>
void test_for_each_with_coors(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_FOR_EACH_WITH_COORS
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  using Elem = tci::elem_t<TenT>;

  TenT a = tci::template eye<TenT>(ctx, 2);

  std::size_t visits = 0;
  bool seen[2][2] = {{false, false}, {false, false}};

  tci::for_each_with_coors(
      ctx, a, [&](Elem &elem, const tci::elem_coors_t<TenT> &coors) {
        // Validate before indexing `seen`: a backend handing back a malformed
        // coordinate must fail as a conformance violation, not as an
        // out-of-bounds write that reads like a defect in the suite.
        TCICT_ASSERT(coors.size() == 2);
        const auto row = static_cast<std::size_t>(coors[0]);
        const auto col = static_cast<std::size_t>(coors[1]);
        TCICT_ASSERT(row < 2 && col < 2);
        TCICT_ASSERT(!seen[row][col]);
        seen[row][col] = true;
        ++visits;
        if (row == col) {
          elem = static_cast<Elem>(2.0);
        }
      });

  // "Visits every element exactly once" over a 2x2 diagonal tensor means 4
  // visits: a backend walking only its 2 stored diagonal entries is excluded.
  TCICT_ASSERT(visits == 4);
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 2; ++j) {
      TCICT_ASSERT(seen[i][j]);
    }
  }

  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, a, {0, 0})), 2.0, tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, a, {1, 1})), 2.0, tol);
  TCICT_ASSERT_CLOSE(std::abs(tci::get_elem(ctx, a, {0, 1})), 0.0, tol);
  TCICT_ASSERT_CLOSE(std::abs(tci::get_elem(ctx, a, {1, 0})), 0.0, tol);
#else
  (void)fix;
#endif
}

// --- for_each_with_coors: const version ---

template <typename TenT>
void test_for_each_with_coors_const(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_FOR_EACH_WITH_COORS
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::reduction, 2);
  using Elem = tci::elem_t<TenT>;

  TenT a = tci::template eye<TenT>(ctx, 2);
  const TenT &const_a = a;

  double sum_diagonal = 0.0;
  double sum_off_diagonal = 0.0;
  std::size_t visits = 0;
  bool seen[2][2] = {{false, false}, {false, false}};

  tci::for_each_with_coors(
      ctx, const_a, [&](const Elem &elem, const tci::elem_coors_t<TenT> &coors) {
        // Validate before indexing `seen`: a malformed coordinate must fail
        // as a conformance violation, not as an out-of-bounds write.
        TCICT_ASSERT(coors.size() == 2);
        const auto row = static_cast<std::size_t>(coors[0]);
        const auto col = static_cast<std::size_t>(coors[1]);
        TCICT_ASSERT(row < 2 && col < 2);
        TCICT_ASSERT(!seen[row][col]);
        seen[row][col] = true;
        ++visits;
        if (row == col) {
          sum_diagonal += real_part<TenT>(elem);
        } else {
          sum_off_diagonal += std::abs(elem);
        }
      });

  TCICT_ASSERT(visits == 4);
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 2; ++j) {
      TCICT_ASSERT(seen[i][j]);
    }
  }

  TCICT_ASSERT_CLOSE(sum_diagonal, 2.0, tol);
  TCICT_ASSERT_CLOSE(sum_off_diagonal, 0.0, tol);
#else
  (void)fix;
#endif
}

// --- for_each: traversal over a diagonal tensor ---

// V1's rule for `for_each` is "Visits every element exactly once". `eye(3)` has
// 9 logical elements — a diagonal tensor's off-diagonal zeros are elements too
// — so a backend traversing the 3 entries it stores is excluded by the visit
// count, which is what this test adds over reading values.
template <typename TenT>
void test_for_each_eye_traversal(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_FOR_EACH) && !defined(TCICT_SKIP_EYE)
  auto &ctx = fix.context();
  auto elem_tol = tolerance(fix, tol_category::elementwise);
  auto sum_tol = tolerance(fix, tol_category::reduction, 9);
  using Elem = tci::elem_t<TenT>;

  auto identity = tci::template eye<TenT>(ctx, 3);

  std::size_t visits = 0;
  std::size_t ones = 0;
  std::size_t zeros = 0;
  double sum = 0.0;

  tci::for_each(ctx, static_cast<const TenT &>(identity),
                [&](const Elem &elem) {
                  ++visits;
                  const double value = real_part<TenT>(elem);
                  sum += value;
                  if (std::abs(value - 1.0) < elem_tol) {
                    ++ones;
                  } else if (std::abs(value) < elem_tol) {
                    ++zeros;
                  }
                });

  TCICT_ASSERT(visits == 9);
  TCICT_ASSERT(ones == 3);
  TCICT_ASSERT(zeros == 6);
  TCICT_ASSERT_CLOSE(sum, 3.0, sum_tol);
#else
  (void)fix;
#endif
}

// --- distinct-value fixture shared by reshape and transpose ---

// Both tests need per-element values a constant fill cannot distinguish: with
// one value repeated, a dropped, duplicated, corrupted, or misplaced element
// leaves the tensor identical to a correct one. The offset keeps every value
// non-zero, so a slot left untouched by an incomplete write is distinguishable
// too. The imaginary part is tied to the real one, which lets a comparison on
// real parts extend to whole elements.
inline double ramp_2x3x4(std::size_t i, std::size_t j, std::size_t k) {
  return static_cast<double>(i * 12 + j * 4 + k) + 1.0;
}

// Writes the ramp into a {2, 3, 4} tensor and returns the values written.
template <typename TenT>
std::vector<double> fill_ramp_2x3x4(tci_test_fixture<TenT> &fix, TenT &tensor) {
  auto &ctx = fix.context();
  std::vector<double> values;
  values.reserve(24);
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      for (std::size_t k = 0; k < 4; ++k) {
        const double value = ramp_2x3x4(i, j, k);
        values.push_back(value);
        tci::set_elem(ctx, tensor, {i, j, k},
                      make_elem<TenT>(value, -0.5 * value));
      }
    }
  }
  return values;
}

// Asserts that every ramp value is readable at the coordinate `coors_of` maps
// it to. Callers supply the mapping, so the same sweep serves an output whose
// axes were permuted and an input that must have stayed put.
template <typename TenT, typename CoorsOf>
void expect_ramp_2x3x4(tci_test_fixture<TenT> &fix, const TenT &tensor,
                       CoorsOf coors_of) {
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      for (std::size_t k = 0; k < 4; ++k) {
        const double value = ramp_2x3x4(i, j, k);
        auto elem = tci::get_elem(ctx, tensor, coors_of(i, j, k));
        TCICT_ASSERT_CLOSE(real_part<TenT>(elem), value, tol);
        if constexpr (is_complex_v<TenT>) {
          TCICT_ASSERT_CLOSE(imag_part<TenT>(elem), -0.5 * value, tol);
        }
      }
    }
  }
}

// --- reshape (in-place) ---

template <typename TenT> void test_reshape(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_RESHAPE
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto tensor = tci::zeros<TenT>(ctx, {2, 3, 4});
  auto source = fill_ramp_2x3x4(fix, tensor);

  tci::shape_t<TenT> new_shape = {6, 4};
  TCICT_ASSERT_NOTHROW(tci::reshape(ctx, tensor, new_shape));
  TCICT_ASSERT(tci::shape(ctx, tensor) == new_shape);
  TCICT_ASSERT(tci::size(ctx, tensor) == 24);

  // V1 has reshape perform "no reordering, transposition, or value change" and
  // preserve "the linear order of logical tensor elements" — but it defines
  // that linear order for no shape, so which coordinate a given element lands
  // on is not portably assertable. The multiset of values is: it holds under
  // any ordering, and still catches a dropped, duplicated, or corrupted
  // element.
  std::vector<double> observed;
  observed.reserve(24);
  for (std::size_t i = 0; i < 6; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      auto elem = tci::get_elem(ctx, tensor, {i, j});
      const double value = real_part<TenT>(elem);
      observed.push_back(value);
      if constexpr (is_complex_v<TenT>) {
        // Pins the imaginary part to its own real part, which extends the
        // multiset comparison below from real parts to whole elements.
        TCICT_ASSERT_CLOSE(imag_part<TenT>(elem), -0.5 * value, tol);
      }
    }
  }

  TCICT_ASSERT(observed.size() == source.size());
  std::sort(source.begin(), source.end());
  std::sort(observed.begin(), observed.end());
  for (std::size_t n = 0; n < source.size(); ++n) {
    TCICT_ASSERT_CLOSE(observed[n], source[n], tol);
  }
#else
  (void)fix;
#endif
}

// --- transpose (out-of-place) ---

template <typename TenT> void test_transpose(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_TRANSPOSE
  auto &ctx = fix.context();
  tci::shape_t<TenT> original_shape = {2, 3, 4};
  auto tensor = tci::zeros<TenT>(ctx, original_shape);
  fill_ramp_2x3x4(fix, tensor);

  TenT transposed;
  tci::List<tci::bond_idx_t<TenT>> new_order = {2, 0, 1};
  TCICT_ASSERT_NOTHROW(tci::transpose(ctx, tensor, new_order, transposed));

  tci::shape_t<TenT> expected_shape = {4, 2, 3};
  TCICT_ASSERT(tci::shape(ctx, transposed) == expected_shape);

  // The shape expectation above is consistent only with
  // new_coord[p] = old_coord[new_order[p]], so for new_order = {2, 0, 1} the
  // element expectation it fixes is out[k, i, j] == a[i, j, k]. Unlike
  // reshape, this needs no linear-order convention: the coordinate map is
  // what `new_order` states.
  expect_ramp_2x3x4(fix, transposed,
                    [](std::size_t i, std::size_t j, std::size_t k) {
                      return tci::elem_coors_t<TenT>{k, i, j};
                    });

  // V1 does not state that an out-of-place overload leaves its input alone;
  // this asserts the reading that "out-of-place" means exactly that.
  TCICT_ASSERT(tci::shape(ctx, tensor) == original_shape);
  expect_ramp_2x3x4(fix, tensor,
                    [](std::size_t i, std::size_t j, std::size_t k) {
                      return tci::elem_coors_t<TenT>{i, j, k};
                    });
#else
  (void)fix;
#endif
}

// --- concatenate: basic 2D ---

template <typename TenT>
void test_concatenate_basic(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_CONCATENATE
  auto &ctx = fix.context();
  auto t1 = tci::fill<TenT>(ctx, {2, 3}, make_elem<TenT>(1.0));
  auto t2 = tci::fill<TenT>(ctx, {2, 3}, make_elem<TenT>(2.0));

  tci::List<TenT> tensors = {t1, t2};

  // Vertical concatenation
  TenT result;
  TCICT_ASSERT_NOTHROW(tci::concatenate(ctx, tensors, 0, result));
  tci::shape_t<TenT> expected_v = {4, 3};
  TCICT_ASSERT(tci::shape(ctx, result) == expected_v);

  // Horizontal concatenation
  TCICT_ASSERT_NOTHROW(tci::concatenate(ctx, tensors, 1, result));
  tci::shape_t<TenT> expected_h = {2, 6};
  TCICT_ASSERT(tci::shape(ctx, result) == expected_h);
#else
  (void)fix;
#endif
}

// --- concatenate: multi-tensor with value verification ---

template <typename TenT>
void test_concatenate_values(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_CONCATENATE
  auto &ctx = fix.context();
  auto a = tci::fill<TenT>(ctx, {2, 3, 4}, make_elem<TenT>(1.0));
  auto b = tci::fill<TenT>(ctx, {2, 1, 4}, make_elem<TenT>(2.0));
  auto c = tci::fill<TenT>(ctx, {2, 2, 4}, make_elem<TenT>(3.0));

  TenT d;
  tci::List<TenT> tensors = {a, b, c};
  TCICT_ASSERT_NOTHROW(tci::concatenate(ctx, tensors, 1, d));

  tci::shape_t<TenT> expected = {2, 6, 4};
  TCICT_ASSERT(tci::shape(ctx, d) == expected);

  // Verify element positions
  auto el_b = tci::get_elem(ctx, b, {0, 0, 0});
  auto el_d3 = tci::get_elem(ctx, d, {0, 3, 0});
  TCICT_ASSERT(el_b == el_d3);
#else
  (void)fix;
#endif
}

// --- extract_sub (out-of-place) ---

template <typename TenT> void test_extract_sub(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_EXTRACT_SUB
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto a = tci::zeros<TenT>(ctx, {3, 4, 2});
  tci::set_elem(ctx, a, {1, 0, 0}, make_elem<TenT>(42.0));
  tci::set_elem(ctx, a, {2, 1, 1}, make_elem<TenT>(13.0));

  TenT sub;
  tci::List<tci::Pair<tci::elem_coor_t<TenT>, tci::elem_coor_t<TenT>>>
      coor_pairs = {{1, 3}, {0, 2}, {0, 2}};
  TCICT_ASSERT_NOTHROW(tci::extract_sub(ctx, a, coor_pairs, sub));

  tci::shape_t<TenT> expected = {2, 2, 2};
  TCICT_ASSERT(tci::shape(ctx, sub) == expected);

  // (1,0,0) in original maps to (0,0,0) in sub
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, sub, {0, 0, 0})), 42.0,
                     tol);
  // (2,1,1) in original maps to (1,1,1) in sub
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, sub, {1, 1, 1})), 13.0,
                     tol);
#else
  (void)fix;
#endif
}

// --- replace_sub (in-place) ---

template <typename TenT>
void test_replace_sub_inplace(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_REPLACE_SUB
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto a = tci::zeros<TenT>(ctx, {3, 4, 2});
  auto sub = tci::zeros<TenT>(ctx, {2, 2, 2});
  tci::set_elem(ctx, sub, {0, 0, 0}, make_elem<TenT>(42.0));
  tci::set_elem(ctx, sub, {1, 1, 1}, make_elem<TenT>(13.0));

  tci::elem_coors_t<TenT> begin_pt = {1, 2, 0};
  TCICT_ASSERT_NOTHROW(tci::replace_sub(ctx, a, sub, begin_pt));

  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, a, {1, 2, 0})), 42.0,
                     tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, a, {2, 3, 1})), 13.0,
                     tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, a, {0, 0, 0})), 0.0,
                     tol);
#else
  (void)fix;
#endif
}

// --- replace_sub (out-of-place) ---

template <typename TenT>
void test_replace_sub_outofplace(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_REPLACE_SUB
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto a = tci::zeros<TenT>(ctx, {4, 4});
  tci::set_elem(ctx, a, {0, 0}, make_elem<TenT>(99.0));

  auto sub = tci::fill<TenT>(ctx, {2, 2}, make_elem<TenT>(1.0));

  TenT result;
  tci::elem_coors_t<TenT> begin_pt = {1, 1};
  TCICT_ASSERT_NOTHROW(tci::replace_sub(ctx, a, sub, begin_pt, result));

  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {1, 1})), 1.0,
                     tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 0})), 99.0,
                     tol);
  // Original unchanged
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, a, {1, 1})), 0.0, tol);
#else
  (void)fix;
#endif
}

// --- expand (in-place) ---

template <typename TenT> void test_expand_inplace(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_EXPAND
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto a = tci::zeros<TenT>(ctx, {2, 2, 2});

  tci::Map<tci::bond_idx_t<TenT>, tci::bond_dim_t<TenT>> bond_map = {{1, 2},
                                                                     {0, 1}};
  TCICT_ASSERT_NOTHROW(tci::expand(ctx, a, bond_map));

  tci::shape_t<TenT> expected = {3, 4, 2};
  TCICT_ASSERT(tci::shape(ctx, a) == expected);

  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, a, {2, 3, 0})), 0.0,
                     tol);
#else
  (void)fix;
#endif
}

// --- expand (out-of-place) ---

template <typename TenT>
void test_expand_outofplace(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_EXPAND
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto a = tci::zeros<TenT>(ctx, {2, 2, 2});
  tci::set_elem(ctx, a, {1, 1, 1}, make_elem<TenT>(5.0));

  tci::Map<tci::bond_idx_t<TenT>, tci::bond_dim_t<TenT>> bond_map = {{1, 2},
                                                                     {0, 1}};
  TenT expanded;
  TCICT_ASSERT_NOTHROW(tci::expand(ctx, a, bond_map, expanded));

  tci::shape_t<TenT> expected = {3, 4, 2};
  TCICT_ASSERT(tci::shape(ctx, expanded) == expected);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, expanded, {1, 1, 1})),
                     5.0, tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, expanded, {2, 3, 0})),
                     0.0, tol);
#else
  (void)fix;
#endif
}

// Asserts that a {3, 3} tensor holds `expected` on its main diagonal and zero
// everywhere else. The off-diagonal check goes through the modulus, so it
// covers both parts of a complex element in one assertion.
template <typename TenT>
void expect_diagonal_3x3(tci_test_fixture<TenT> &fix, const TenT &tensor,
                         const double (&expected)[3]) {
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      if (i == j) {
        TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {i, j})),
                           expected[i], tol);
      } else {
        TCICT_ASSERT_CLOSE(std::abs(tci::get_elem(ctx, tensor, {i, j})), 0.0,
                           tol);
      }
    }
  }
}

// --- diag: vector to matrix ---

template <typename TenT>
void test_diag_vec_to_mat(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_DIAG
  auto &ctx = fix.context();
  auto vector = tci::zeros<TenT>(ctx, {3});
  tci::set_elem(ctx, vector, {0}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, vector, {1}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, vector, {2}, make_elem<TenT>(3.0));

  tci::diag(ctx, vector);

  TCICT_ASSERT(tci::order(ctx, vector) == 2);
  TCICT_ASSERT(tci::shape(ctx, vector)[0] == 3);
  TCICT_ASSERT(tci::shape(ctx, vector)[1] == 3);
  // The promoted tensor is diagonal, and a diagonal tensor's off-diagonal
  // zeros count as logical elements however few entries the backend stores.
  TCICT_ASSERT(tci::size(ctx, vector) == 9);

  const double expected[3] = {1.0, 2.0, 3.0};
  expect_diagonal_3x3(fix, vector, expected);
#else
  (void)fix;
#endif
}

// --- diag: matrix to vector ---

template <typename TenT>
void test_diag_mat_to_vec(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_DIAG
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  auto identity = tci::eye<TenT>(ctx, 3);

  tci::diag(ctx, identity);

  TCICT_ASSERT(tci::order(ctx, identity) == 1);
  TCICT_ASSERT(tci::size(ctx, identity) == 3);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, identity, {0})), 1.0,
                     tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, identity, {1})), 1.0,
                     tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, identity, {2})), 1.0,
                     tol);
#else
  (void)fix;
#endif
}

// --- diag: vector to matrix (out-of-place) ---

template <typename TenT>
void test_diag_vec_to_mat_outofplace(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_DIAG
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  const double expected[3] = {1.5, -2.5, 3.5};

  auto vector = tci::zeros<TenT>(ctx, {3});
  for (std::size_t i = 0; i < 3; ++i) {
    tci::set_elem(ctx, vector, {i}, make_elem<TenT>(expected[i]));
  }

  TenT matrix;
  TCICT_ASSERT_NOTHROW(tci::diag(ctx, vector, matrix));

  TCICT_ASSERT(tci::order(ctx, matrix) == 2);
  TCICT_ASSERT(tci::shape(ctx, matrix)[0] == 3);
  TCICT_ASSERT(tci::shape(ctx, matrix)[1] == 3);
  TCICT_ASSERT(tci::size(ctx, matrix) == 9);

  expect_diagonal_3x3(fix, matrix, expected);

  // V1 does not state that an out-of-place overload leaves its input alone;
  // this asserts the reading that "out-of-place" means exactly that.
  TCICT_ASSERT(tci::order(ctx, vector) == 1);
  TCICT_ASSERT(tci::size(ctx, vector) == 3);
  for (std::size_t i = 0; i < 3; ++i) {
    TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, vector, {i})),
                       expected[i], tol);
  }
#else
  (void)fix;
#endif
}

// --- diag: matrix to vector (out-of-place) ---

template <typename TenT>
void test_diag_mat_to_vec_outofplace(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_DIAG
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  // Distinct diagonal values rather than an identity: all-ones cannot tell a
  // correct extraction from a reordered one. Building the input with `zeros`
  // also keeps this test under TCICT_SKIP_DIAG alone, so a backend that
  // implements diag but skips eye can still run it.
  const double expected[3] = {4.5, -5.5, 6.5};

  auto matrix = tci::zeros<TenT>(ctx, {3, 3});
  for (std::size_t i = 0; i < 3; ++i) {
    tci::set_elem(ctx, matrix, {i, i}, make_elem<TenT>(expected[i]));
  }

  TenT vector;
  TCICT_ASSERT_NOTHROW(tci::diag(ctx, matrix, vector));

  TCICT_ASSERT(tci::order(ctx, vector) == 1);
  TCICT_ASSERT(tci::size(ctx, vector) == 3);
  for (std::size_t i = 0; i < 3; ++i) {
    TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, vector, {i})),
                       expected[i], tol);
  }

  // V1 does not state that an out-of-place overload leaves its input alone;
  // this asserts the reading that "out-of-place" means exactly that.
  TCICT_ASSERT(tci::order(ctx, matrix) == 2);
  TCICT_ASSERT(tci::size(ctx, matrix) == 9);
#else
  (void)fix;
#endif
}

// --- stack: basic ---

template <typename TenT> void test_stack_basic(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_STACK
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);

  auto t1 = tci::fill<TenT>(ctx, {2, 3}, make_elem<TenT>(1.0));
  auto t2 = tci::fill<TenT>(ctx, {2, 3}, make_elem<TenT>(2.0));

  TenT result;
  tci::List<TenT> tensors = {t1, t2};
  tci::stack(ctx, tensors, 0, result);

  auto s = tci::shape(ctx, result);
  TCICT_ASSERT(s.size() == 3);
  TCICT_ASSERT(s[0] == 2); // stacked dimension
  TCICT_ASSERT(s[1] == 2);
  TCICT_ASSERT(s[2] == 3);

  // First slice filled with 1, second with 2
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 0, 0})),
                     1.0, tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 1, 2})),
                     1.0, tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {1, 0, 0})),
                     2.0, tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {1, 1, 2})),
                     2.0, tol);
#else
  (void)fix;
#endif
}

// --- stack: last axis ---

template <typename TenT>
void test_stack_last_axis(tci_test_fixture<TenT> &fix) {
#ifndef TCICT_SKIP_STACK
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);

  auto t1 = tci::fill<TenT>(ctx, {2, 3}, make_elem<TenT>(1.0));
  auto t2 = tci::fill<TenT>(ctx, {2, 3}, make_elem<TenT>(2.0));
  auto t3 = tci::fill<TenT>(ctx, {2, 3}, make_elem<TenT>(3.0));

  TenT result;
  tci::List<TenT> tensors = {t1, t2, t3};
  tci::stack(ctx, tensors, 2, result);

  auto s = tci::shape(ctx, result);
  TCICT_ASSERT(s.size() == 3);
  TCICT_ASSERT(s[0] == 2);
  TCICT_ASSERT(s[1] == 3);
  TCICT_ASSERT(s[2] == 3); // stacked dimension

  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 0, 0})),
                     1.0, tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 0, 1})),
                     2.0, tol);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 0, 2})),
                     3.0, tol);
#else
  (void)fix;
#endif
}

} // namespace tests
} // namespace tcict

// Bulk registration helpers: invoke X(..., "category", test_fn) once per test.
// See include/tcict/adapters/doctest.h for usage.
//
// ALL_TYPES: safe for both real and complex TenT (imag-part assertions are
//   dual-guarded via `if constexpr (is_complex_v<TenT>)`).
#define TCICT_FOREACH_TENSOR_MANIPULATION_TEST_ALL_TYPES(X, ...) \
  X(__VA_ARGS__, "tensor_manipulation", test_shrink_inplace) \
  X(__VA_ARGS__, "tensor_manipulation", test_shrink_outofplace) \
  X(__VA_ARGS__, "tensor_manipulation", test_shrink_complex_values) \
  X(__VA_ARGS__, "tensor_manipulation", test_real_extraction) \
  X(__VA_ARGS__, "tensor_manipulation", test_imag_extraction) \
  X(__VA_ARGS__, "tensor_manipulation", test_cplx_conj_inplace) \
  X(__VA_ARGS__, "tensor_manipulation", test_cplx_conj_outofplace) \
  X(__VA_ARGS__, "tensor_manipulation", test_for_each_doubling) \
  X(__VA_ARGS__, "tensor_manipulation", test_for_each_summation) \
  X(__VA_ARGS__, "tensor_manipulation", test_for_each_capture) \
  X(__VA_ARGS__, "tensor_manipulation", test_for_each_const) \
  X(__VA_ARGS__, "tensor_manipulation", test_for_each_inversion) \
  X(__VA_ARGS__, "tensor_manipulation", test_for_each_with_coors) \
  X(__VA_ARGS__, "tensor_manipulation", test_for_each_with_coors_const) \
  X(__VA_ARGS__, "tensor_manipulation", test_for_each_eye_traversal) \
  X(__VA_ARGS__, "tensor_manipulation", test_reshape) \
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
  X(__VA_ARGS__, "tensor_manipulation", test_diag_vec_to_mat_outofplace) \
  X(__VA_ARGS__, "tensor_manipulation", test_diag_mat_to_vec_outofplace) \
  X(__VA_ARGS__, "tensor_manipulation", test_stack_basic) \
  X(__VA_ARGS__, "tensor_manipulation", test_stack_last_axis)

// REAL_ONLY: TCI `to_cplx` takes a real tensor and lifts to complex; these
//   tests are only meaningful for real TenT.
#define TCICT_FOREACH_TENSOR_MANIPULATION_TEST_REAL_ONLY(X, ...) \
  X(__VA_ARGS__, "tensor_manipulation", test_to_cplx_outofplace)

// CPLX_ONLY: body is wrapped in `if constexpr (is_complex_v<TenT>)`; running
//   for real TenT would be a no-op, so skip registration.
#define TCICT_FOREACH_TENSOR_MANIPULATION_TEST_CPLX_ONLY(X, ...) \
  X(__VA_ARGS__, "tensor_manipulation", test_to_cplx_complex_to_complex)
