#pragma once

#include <tcict/assertion.h>
#include <tcict/elem_helper.h>
#include <tcict/fixture.h>
#include <tcict/skip.h>

#include <cmath>

namespace tcict {
namespace tests {

// --- shrink (in-place) ---

template <typename TenT> void test_shrink_inplace(tci_test_fixture<TenT> &fix) {
#ifdef TCICT_SKIP_SHRINK
  return;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
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
                     eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 1})), 2.0,
                     eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {1, 0})), 4.0,
                     eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {1, 1})), 5.0,
                     eps);
}

// --- shrink (out-of-place) ---

template <typename TenT>
void test_shrink_outofplace(tci_test_fixture<TenT> &fix) {
#ifdef TCICT_SKIP_SHRINK
  return;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
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
                     eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, output, {0, 1})), 12.0,
                     eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, output, {1, 0})), 21.0,
                     eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, output, {1, 1})), 22.0,
                     eps);
}

// --- shrink preserves complex values ---

template <typename TenT>
void test_shrink_complex_values(tci_test_fixture<TenT> &fix) {
#ifdef TCICT_SKIP_SHRINK
  return;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
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
  TCICT_ASSERT_CLOSE(real_part<TenT>(e00), 1.5, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(e01), 3.5, eps);
  if constexpr (is_complex_v<TenT>) {
    TCICT_ASSERT_CLOSE(imag_part<TenT>(e00), 2.5, eps);
    TCICT_ASSERT_CLOSE(imag_part<TenT>(e01), 4.5, eps);
  }
}

// --- real extraction (out-of-place) ---

template <typename TenT>
void test_real_extraction(tci_test_fixture<TenT> &fix) {
#ifdef TCICT_SKIP_REAL
  return;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  auto tensor = tci::zeros<TenT>(ctx, {2, 2});
  tci::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(3.14, 2.71));
  tci::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(-1.59, 0.58));

  auto real_tensor = tci::real(ctx, tensor);

  using RealTenT = tci::real_ten_t<TenT>;
  auto elem00 = tci::get_elem(ctx, real_tensor, {0, 0});
  auto elem11 = tci::get_elem(ctx, real_tensor, {1, 1});
  TCICT_ASSERT_CLOSE(real_part<RealTenT>(elem00), 3.14, eps);
  TCICT_ASSERT_CLOSE(real_part<RealTenT>(elem11), -1.59, eps);
}

// --- imag extraction (out-of-place) ---

template <typename TenT>
void test_imag_extraction(tci_test_fixture<TenT> &fix) {
#ifdef TCICT_SKIP_IMAG
  return;
#endif
  // Per TCI spec, tci::imag on a real tensor yields a zero tensor, so the
  // non-zero imaginary expectations below are meaningful only for complex TenT.
  if constexpr (is_complex_v<TenT>) {
    auto &ctx = fix.context();
    auto eps = fix.epsilon();
    auto tensor = tci::zeros<TenT>(ctx, {2, 2});
    tci::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(3.14, 2.71));
    tci::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(-1.59, 0.58));

    auto imag_tensor = tci::imag(ctx, tensor);

    using RealTenT = tci::real_ten_t<TenT>;
    auto elem00 = tci::get_elem(ctx, imag_tensor, {0, 0});
    auto elem11 = tci::get_elem(ctx, imag_tensor, {1, 1});
    TCICT_ASSERT_CLOSE(real_part<RealTenT>(elem00), 2.71, eps);
    TCICT_ASSERT_CLOSE(real_part<RealTenT>(elem11), 0.58, eps);
  }
}

// --- real and imag extraction (in-place) ---

template <typename TenT>
void test_real_imag_inplace(tci_test_fixture<TenT> &fix) {
#if defined(TCICT_SKIP_REAL) || defined(TCICT_SKIP_IMAG)
  return;
#endif
  // Per TCI spec, tci::imag on a real tensor yields a zero tensor; the
  // imaginary-side expectations below assume complex input.
  if constexpr (is_complex_v<TenT>) {
    auto &ctx = fix.context();
    auto eps = fix.epsilon();
    auto tensor = tci::zeros<TenT>(ctx, {2, 2});
    tci::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(5.25, 7.75));
    tci::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(-2.25, -3.75));

    tci::real_ten_t<TenT> real_output, imag_output;
    tci::real(ctx, tensor, real_output);
    tci::imag(ctx, tensor, imag_output);

    using RealTenT = tci::real_ten_t<TenT>;
    TCICT_ASSERT_CLOSE(
        real_part<RealTenT>(tci::get_elem(ctx, real_output, {0, 0})), 5.25, eps);
    TCICT_ASSERT_CLOSE(
        real_part<RealTenT>(tci::get_elem(ctx, real_output, {1, 1})), -2.25, eps);
    TCICT_ASSERT_CLOSE(
        real_part<RealTenT>(tci::get_elem(ctx, imag_output, {0, 0})), 7.75, eps);
    TCICT_ASSERT_CLOSE(
        real_part<RealTenT>(tci::get_elem(ctx, imag_output, {1, 1})), -3.75, eps);
  }
}

// --- cplx_conj (in-place) ---

template <typename TenT>
void test_cplx_conj_inplace(tci_test_fixture<TenT> &fix) {
#ifdef TCICT_SKIP_CPLX_CONJ
  return;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  auto tensor = tci::zeros<TenT>(ctx, {2, 2});
  tci::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(1.0, 2.0));
  tci::set_elem(ctx, tensor, {0, 1}, make_elem<TenT>(-3.0, 4.0));
  tci::set_elem(ctx, tensor, {1, 0}, make_elem<TenT>(5.0, -6.0));
  tci::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(-7.0, -8.0));

  tci::cplx_conj(ctx, tensor);

  // Real parts unchanged for both real and complex (cplx_conj on real tensors
  // is a no-op per TCI spec); imaginary parts only exist for complex.
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 0})), 1.0,
                     eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 1})), -3.0,
                     eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {1, 0})), 5.0,
                     eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {1, 1})), -7.0,
                     eps);
  if constexpr (is_complex_v<TenT>) {
    TCICT_ASSERT_CLOSE(imag_part<TenT>(tci::get_elem(ctx, tensor, {0, 0})), -2.0,
                       eps);
    TCICT_ASSERT_CLOSE(imag_part<TenT>(tci::get_elem(ctx, tensor, {0, 1})), -4.0,
                       eps);
    TCICT_ASSERT_CLOSE(imag_part<TenT>(tci::get_elem(ctx, tensor, {1, 0})), 6.0,
                       eps);
    TCICT_ASSERT_CLOSE(imag_part<TenT>(tci::get_elem(ctx, tensor, {1, 1})), 8.0,
                       eps);
  }
}

// --- cplx_conj (out-of-place) ---

template <typename TenT>
void test_cplx_conj_outofplace(tci_test_fixture<TenT> &fix) {
#ifdef TCICT_SKIP_CPLX_CONJ
  return;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  auto input = tci::zeros<TenT>(ctx, {2, 2});
  tci::set_elem(ctx, input, {0, 0}, make_elem<TenT>(3.14, 2.71));
  tci::set_elem(ctx, input, {1, 1}, make_elem<TenT>(-1.41, -1.73));

  TenT output;
  tci::cplx_conj(ctx, input, output);

  // Real parts hold for both real and complex (cplx_conj out-of-place is a
  // deep copy for real tensors per TCI spec); imaginary parts only exist for
  // complex.
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, input, {0, 0})), 3.14,
                     eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, output, {0, 0})), 3.14,
                     eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, output, {1, 1})), -1.41,
                     eps);
  if constexpr (is_complex_v<TenT>) {
    TCICT_ASSERT_CLOSE(imag_part<TenT>(tci::get_elem(ctx, input, {0, 0})), 2.71,
                       eps);
    TCICT_ASSERT_CLOSE(imag_part<TenT>(tci::get_elem(ctx, output, {0, 0})), -2.71,
                       eps);
    TCICT_ASSERT_CLOSE(imag_part<TenT>(tci::get_elem(ctx, output, {1, 1})), 1.73,
                       eps);
  }
}

// --- to_cplx (out-of-place, from real type) ---
// NOTE: TenT here should be a real tensor type (e.g., CytnxTensor<double>)

template <typename RealTenT>
void test_to_cplx_outofplace(tci_test_fixture<RealTenT> &fix) {
#ifdef TCICT_SKIP_TO_CPLX
  return;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
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
  TCICT_ASSERT_CLOSE(real_part<CplxTenT>(elem00), 1.5, eps);
  TCICT_ASSERT_CLOSE(imag_part<CplxTenT>(elem00), 0.0, eps);
  TCICT_ASSERT_CLOSE(real_part<CplxTenT>(elem11), 4.5, eps);
  TCICT_ASSERT_CLOSE(imag_part<CplxTenT>(elem11), 0.0, eps);
}

// --- to_cplx (in-place, from real type) ---

template <typename RealTenT>
void test_to_cplx_inplace(tci_test_fixture<RealTenT> &fix) {
#ifdef TCICT_SKIP_TO_CPLX
  return;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  RealTenT real_tensor;
  real_tensor = tci::zeros<RealTenT>(ctx, {2, 2});

  tci::set_elem(ctx, real_tensor, {0, 0},
                static_cast<tci::elem_t<RealTenT>>(7.25));
  tci::set_elem(ctx, real_tensor, {1, 1},
                static_cast<tci::elem_t<RealTenT>>(8.75));

  tci::cplx_ten_t<RealTenT> complex_output;
  tci::to_cplx(ctx, real_tensor, complex_output);

  using CplxTenT = tci::cplx_ten_t<RealTenT>;
  auto elem00 = tci::get_elem(ctx, complex_output, {0, 0});
  auto elem11 = tci::get_elem(ctx, complex_output, {1, 1});
  TCICT_ASSERT_CLOSE(real_part<CplxTenT>(elem00), 7.25, eps);
  TCICT_ASSERT_CLOSE(imag_part<CplxTenT>(elem00), 0.0, eps);
  TCICT_ASSERT_CLOSE(real_part<CplxTenT>(elem11), 8.75, eps);
  TCICT_ASSERT_CLOSE(imag_part<CplxTenT>(elem11), 0.0, eps);
}

// --- to_cplx (complex to complex) ---

template <typename TenT>
void test_to_cplx_complex_to_complex(tci_test_fixture<TenT> &fix) {
#ifdef TCICT_SKIP_TO_CPLX
  return;
#endif
  // For real TenT, tci::to_cplx returns cplx_ten_t<TenT> whose elements are
  // cplx_t<TenT>; calling imag_part<TenT>(cplx_elem) would be a type error.
  // This test therefore only runs when TenT is already complex.
  if constexpr (is_complex_v<TenT>) {
    auto &ctx = fix.context();
    auto eps = fix.epsilon();
    auto tensor = tci::zeros<TenT>(ctx, {2, 2});
    tci::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(3.14, 2.71));
    tci::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(-1.41, 1.73));

    auto result = tci::to_cplx(ctx, tensor);

    auto elem00 = tci::get_elem(ctx, result, {0, 0});
    auto elem11 = tci::get_elem(ctx, result, {1, 1});
    TCICT_ASSERT_CLOSE(real_part<TenT>(elem00), 3.14, eps);
    TCICT_ASSERT_CLOSE(imag_part<TenT>(elem00), 2.71, eps);
    TCICT_ASSERT_CLOSE(real_part<TenT>(elem11), -1.41, eps);
    TCICT_ASSERT_CLOSE(imag_part<TenT>(elem11), 1.73, eps);
  }
}

// --- for_each: element doubling ---

template <typename TenT>
void test_for_each_doubling(tci_test_fixture<TenT> &fix) {
#ifdef TCICT_SKIP_FOR_EACH
  return;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  using Elem = tci::elem_t<TenT>;

  auto tensor = tci::zeros<TenT>(ctx, {2, 3});
  tci::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, tensor, {0, 1}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, tensor, {0, 2}, make_elem<TenT>(3.0));
  tci::set_elem(ctx, tensor, {1, 0}, make_elem<TenT>(4.0));
  tci::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(5.0));
  tci::set_elem(ctx, tensor, {1, 2}, make_elem<TenT>(6.0));

  tci::for_each(ctx, tensor, [](Elem &elem) { elem = elem * 2.0; });

  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 0})), 2.0,
                     eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 2})), 6.0,
                     eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {1, 2})), 12.0,
                     eps);
}

// --- for_each: iteration and summation ---

template <typename TenT>
void test_for_each_summation(tci_test_fixture<TenT> &fix) {
#ifdef TCICT_SKIP_FOR_EACH
  return;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
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
  TCICT_ASSERT_CLOSE(real_part<TenT>(sum), 10.0, eps);
}

// --- for_each: scalar multiplication with capture ---

template <typename TenT>
void test_for_each_capture(tci_test_fixture<TenT> &fix) {
#ifdef TCICT_SKIP_FOR_EACH
  return;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  using Elem = tci::elem_t<TenT>;

  auto tensor = tci::fill<TenT>(ctx, {2, 2}, make_elem<TenT>(3.0, 1.0));

  double multiplier = 0.5;
  tci::for_each(ctx, tensor,
                [multiplier](Elem &elem) { elem = elem * multiplier; });

  auto result = tci::get_elem(ctx, tensor, {0, 0});
  TCICT_ASSERT_CLOSE(real_part<TenT>(result), 1.5, eps);
  if constexpr (is_complex_v<TenT>) {
    TCICT_ASSERT_CLOSE(imag_part<TenT>(result), 0.5, eps);
  }
}

// --- for_each: const version ---

template <typename TenT> void test_for_each_const(tci_test_fixture<TenT> &fix) {
#ifdef TCICT_SKIP_FOR_EACH
  return;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  using Elem = tci::elem_t<TenT>;

  auto tensor = tci::fill<TenT>(ctx, {3}, make_elem<TenT>(2.0, 3.0));

  Elem sum = make_elem<TenT>(0.0);
  tci::for_each(ctx, static_cast<const TenT &>(tensor),
                [&sum](const Elem &elem) { sum = sum + elem; });

  TCICT_ASSERT_CLOSE(real_part<TenT>(sum), 6.0, eps);
  if constexpr (is_complex_v<TenT>) {
    TCICT_ASSERT_CLOSE(imag_part<TenT>(sum), 9.0, eps);
  }
}

// --- for_each: element-wise inversion ---

template <typename TenT>
void test_for_each_inversion(tci_test_fixture<TenT> &fix) {
#ifdef TCICT_SKIP_FOR_EACH
  return;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  using Elem = tci::elem_t<TenT>;

  auto tensor = tci::fill<TenT>(ctx, {2, 2}, make_elem<TenT>(0.5));

  tci::for_each(ctx, tensor, [](Elem &elem) {
    if (std::abs(elem) > 1e-12) {
      elem = make_elem<TenT>(1.0, 0.0) / elem;
    }
  });

  auto result = tci::get_elem(ctx, tensor, {0, 0});
  TCICT_ASSERT_CLOSE(real_part<TenT>(result), 2.0, eps);
}

// --- for_each_with_coors: mutable ---

template <typename TenT>
void test_for_each_with_coors(tci_test_fixture<TenT> &fix) {
#ifdef TCICT_SKIP_FOR_EACH_WITH_COORS
  return;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  using Elem = tci::elem_t<TenT>;

  TenT a = tci::template eye<TenT>(ctx, 2);

  tci::for_each_with_coors(
      ctx, a, [](Elem &elem, const tci::elem_coors_t<TenT> &coors) {
        if (coors[0] == coors[1]) {
          elem = static_cast<Elem>(2.0);
        }
      });

  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, a, {0, 0})), 2.0, eps);
}

// --- for_each_with_coors: const version ---

template <typename TenT>
void test_for_each_with_coors_const(tci_test_fixture<TenT> &fix) {
#ifdef TCICT_SKIP_FOR_EACH_WITH_COORS
  return;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  using Elem = tci::elem_t<TenT>;

  TenT a = tci::template eye<TenT>(ctx, 2);
  const TenT &const_a = a;

  double sum_diagonal = 0.0;
  tci::for_each_with_coors(
      ctx, const_a,
      [&sum_diagonal](const Elem &elem, const tci::elem_coors_t<TenT> &coors) {
        if (coors[0] == coors[1]) {
          sum_diagonal += real_part<TenT>(elem);
        }
      });

  TCICT_ASSERT_CLOSE(sum_diagonal, 2.0, eps);
}

// --- reshape (in-place) ---

template <typename TenT> void test_reshape(tci_test_fixture<TenT> &fix) {
#ifdef TCICT_SKIP_RESHAPE
  return;
#endif
  auto &ctx = fix.context();
  auto tensor = tci::fill<TenT>(ctx, {2, 3, 4}, make_elem<TenT>(1.0));

  tci::shape_t<TenT> new_shape = {6, 4};
  TCICT_ASSERT_NOTHROW(tci::reshape(ctx, tensor, new_shape));
  TCICT_ASSERT(tci::shape(ctx, tensor) == new_shape);
  TCICT_ASSERT(tci::size(ctx, tensor) == 24);
}

// --- transpose (out-of-place) ---

template <typename TenT> void test_transpose(tci_test_fixture<TenT> &fix) {
#ifdef TCICT_SKIP_TRANSPOSE
  return;
#endif
  auto &ctx = fix.context();
  auto tensor = tci::fill<TenT>(ctx, {2, 3, 4}, make_elem<TenT>(1.0));

  TenT transposed;
  tci::List<tci::bond_idx_t<TenT>> new_order = {2, 0, 1};
  TCICT_ASSERT_NOTHROW(tci::transpose(ctx, tensor, new_order, transposed));

  tci::shape_t<TenT> expected_shape = {4, 2, 3};
  TCICT_ASSERT(tci::shape(ctx, transposed) == expected_shape);
}

// --- concatenate: basic 2D ---

template <typename TenT>
void test_concatenate_basic(tci_test_fixture<TenT> &fix) {
#ifdef TCICT_SKIP_CONCATENATE
  return;
#endif
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
}

// --- concatenate: multi-tensor with value verification ---

template <typename TenT>
void test_concatenate_values(tci_test_fixture<TenT> &fix) {
#ifdef TCICT_SKIP_CONCATENATE
  return;
#endif
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
}

// --- concatenate: error cases ---

template <typename TenT>
void test_concatenate_errors(tci_test_fixture<TenT> &fix) {
#ifdef TCICT_SKIP_CONCATENATE
  return;
#endif
  auto &ctx = fix.context();
  auto tensor = tci::fill<TenT>(ctx, {2, 3, 4}, make_elem<TenT>(1.0));

  TenT result;
  tci::List<TenT> single = {tensor};
  TCICT_ASSERT_THROWS(std::invalid_argument,
                      tci::concatenate(ctx, single, 3, result));

  tci::List<TenT> empty;
  TCICT_ASSERT_THROWS(std::invalid_argument,
                      tci::concatenate(ctx, empty, 0, result));
}

// --- extract_sub (out-of-place) ---

template <typename TenT> void test_extract_sub(tci_test_fixture<TenT> &fix) {
#ifdef TCICT_SKIP_EXTRACT_SUB
  return;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
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
                     eps);
  // (2,1,1) in original maps to (1,1,1) in sub
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, sub, {1, 1, 1})), 13.0,
                     eps);
}

// --- extract_sub: error handling ---

template <typename TenT>
void test_extract_sub_errors(tci_test_fixture<TenT> &fix) {
#ifdef TCICT_SKIP_EXTRACT_SUB
  return;
#endif
  auto &ctx = fix.context();
  auto a = tci::zeros<TenT>(ctx, {3, 3});

  // Wrong number of coordinate pairs
  tci::List<tci::Pair<tci::elem_coor_t<TenT>, tci::elem_coor_t<TenT>>>
      wrong_count = {{0, 2}, {0, 2}, {0, 1}};
  TCICT_ASSERT_THROWS(std::exception, tci::extract_sub(ctx, a, wrong_count));

  // Invalid range (start >= end)
  tci::List<tci::Pair<tci::elem_coor_t<TenT>, tci::elem_coor_t<TenT>>>
      invalid_range = {{2, 1}, {0, 2}};
  TCICT_ASSERT_THROWS(std::exception, tci::extract_sub(ctx, a, invalid_range));
}

// --- replace_sub (in-place) ---

template <typename TenT>
void test_replace_sub_inplace(tci_test_fixture<TenT> &fix) {
#ifdef TCICT_SKIP_REPLACE_SUB
  return;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  auto a = tci::zeros<TenT>(ctx, {3, 4, 2});
  auto sub = tci::zeros<TenT>(ctx, {2, 2, 2});
  tci::set_elem(ctx, sub, {0, 0, 0}, make_elem<TenT>(42.0));
  tci::set_elem(ctx, sub, {1, 1, 1}, make_elem<TenT>(13.0));

  tci::elem_coors_t<TenT> begin_pt = {1, 2, 0};
  TCICT_ASSERT_NOTHROW(tci::replace_sub(ctx, a, sub, begin_pt));

  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, a, {1, 2, 0})), 42.0,
                     eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, a, {2, 3, 1})), 13.0,
                     eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, a, {0, 0, 0})), 0.0,
                     eps);
}

// --- replace_sub (out-of-place) ---

template <typename TenT>
void test_replace_sub_outofplace(tci_test_fixture<TenT> &fix) {
#ifdef TCICT_SKIP_REPLACE_SUB
  return;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  auto a = tci::zeros<TenT>(ctx, {4, 4});
  tci::set_elem(ctx, a, {0, 0}, make_elem<TenT>(99.0));

  auto sub = tci::fill<TenT>(ctx, {2, 2}, make_elem<TenT>(1.0));

  TenT result;
  tci::elem_coors_t<TenT> begin_pt = {1, 1};
  TCICT_ASSERT_NOTHROW(tci::replace_sub(ctx, a, sub, begin_pt, result));

  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {1, 1})), 1.0,
                     eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 0})), 99.0,
                     eps);
  // Original unchanged
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, a, {1, 1})), 0.0, eps);
}

// --- replace_sub: error cases ---

template <typename TenT>
void test_replace_sub_errors(tci_test_fixture<TenT> &fix) {
#ifdef TCICT_SKIP_REPLACE_SUB
  return;
#endif
  auto &ctx = fix.context();
  auto a = tci::zeros<TenT>(ctx, {3, 3});

  // Dimension mismatch
  auto sub = tci::zeros<TenT>(ctx, {2, 2, 2});
  TCICT_ASSERT_THROWS(std::exception, tci::replace_sub(ctx, a, sub, {0, 0}));

  // Out of bounds
  auto sub2 = tci::zeros<TenT>(ctx, {2, 2});
  TCICT_ASSERT_THROWS(std::exception, tci::replace_sub(ctx, a, sub2, {2, 2}));
}

// --- expand (in-place) ---

template <typename TenT> void test_expand_inplace(tci_test_fixture<TenT> &fix) {
#ifdef TCICT_SKIP_EXPAND
  return;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  auto a = tci::zeros<TenT>(ctx, {2, 2, 2});

  tci::Map<tci::bond_idx_t<TenT>, tci::bond_dim_t<TenT>> bond_map = {{1, 2},
                                                                     {0, 1}};
  TCICT_ASSERT_NOTHROW(tci::expand(ctx, a, bond_map));

  tci::shape_t<TenT> expected = {3, 4, 2};
  TCICT_ASSERT(tci::shape(ctx, a) == expected);

  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, a, {2, 3, 0})), 0.0,
                     eps);
}

// --- expand (out-of-place) ---

template <typename TenT>
void test_expand_outofplace(tci_test_fixture<TenT> &fix) {
#ifdef TCICT_SKIP_EXPAND
  return;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  auto a = tci::zeros<TenT>(ctx, {2, 2, 2});
  tci::set_elem(ctx, a, {1, 1, 1}, make_elem<TenT>(5.0));

  tci::Map<tci::bond_idx_t<TenT>, tci::bond_dim_t<TenT>> bond_map = {{1, 2},
                                                                     {0, 1}};
  TenT expanded;
  TCICT_ASSERT_NOTHROW(tci::expand(ctx, a, bond_map, expanded));

  tci::shape_t<TenT> expected = {3, 4, 2};
  TCICT_ASSERT(tci::shape(ctx, expanded) == expected);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, expanded, {1, 1, 1})),
                     5.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, expanded, {2, 3, 0})),
                     0.0, eps);
}

// --- expand: invalid bond throws ---

template <typename TenT>
void test_expand_invalid_throws(tci_test_fixture<TenT> &fix) {
#ifdef TCICT_SKIP_EXPAND
  return;
#endif
  auto &ctx = fix.context();
  auto a = tci::zeros<TenT>(ctx, {2, 2});

  tci::Map<tci::bond_idx_t<TenT>, tci::bond_dim_t<TenT>> invalid_map = {{3, 1}};
  TCICT_ASSERT_THROWS(std::exception, tci::expand(ctx, a, invalid_map));
}

// --- diag: vector to matrix ---

template <typename TenT>
void test_diag_vec_to_mat(tci_test_fixture<TenT> &fix) {
#ifdef TCICT_SKIP_DIAG
  return;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  auto vector = tci::zeros<TenT>(ctx, {3});
  tci::set_elem(ctx, vector, {0}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, vector, {1}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, vector, {2}, make_elem<TenT>(3.0));

  tci::diag(ctx, vector);

  TCICT_ASSERT(tci::order(ctx, vector) == 2);
  TCICT_ASSERT(tci::shape(ctx, vector)[0] == 3);
  TCICT_ASSERT(tci::shape(ctx, vector)[1] == 3);

  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, vector, {0, 0})), 1.0,
                     eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, vector, {1, 1})), 2.0,
                     eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, vector, {2, 2})), 3.0,
                     eps);
  TCICT_ASSERT_CLOSE(std::abs(tci::get_elem(ctx, vector, {0, 1})), 0.0, eps);
}

// --- diag: matrix to vector ---

template <typename TenT>
void test_diag_mat_to_vec(tci_test_fixture<TenT> &fix) {
#ifdef TCICT_SKIP_DIAG
  return;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();
  auto identity = tci::eye<TenT>(ctx, 3);

  tci::diag(ctx, identity);

  TCICT_ASSERT(tci::order(ctx, identity) == 1);
  TCICT_ASSERT(tci::size(ctx, identity) == 3);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, identity, {0})), 1.0,
                     eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, identity, {1})), 1.0,
                     eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, identity, {2})), 1.0,
                     eps);
}

// --- stack: basic ---

template <typename TenT> void test_stack_basic(tci_test_fixture<TenT> &fix) {
#ifdef TCICT_SKIP_STACK
  return;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();

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
                     1.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 1, 2})),
                     1.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {1, 0, 0})),
                     2.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {1, 1, 2})),
                     2.0, eps);
}

// --- stack: last axis ---

template <typename TenT>
void test_stack_last_axis(tci_test_fixture<TenT> &fix) {
#ifdef TCICT_SKIP_STACK
  return;
#endif
  auto &ctx = fix.context();
  auto eps = fix.epsilon();

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
                     1.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 0, 1})),
                     2.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 0, 2})),
                     3.0, eps);
}

// --- stack: errors ---

template <typename TenT> void test_stack_errors(tci_test_fixture<TenT> &fix) {
#ifdef TCICT_SKIP_STACK
  return;
#endif
  auto &ctx = fix.context();

  // Empty list
  TenT result;
  tci::List<TenT> empty;
  TCICT_ASSERT_THROWS(std::invalid_argument, tci::stack(ctx, empty, 0, result));
}

} // namespace tests
} // namespace tcict
