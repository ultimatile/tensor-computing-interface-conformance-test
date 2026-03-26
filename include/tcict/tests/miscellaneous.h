#pragma once

#include <tcict/assertion.h>
#include <tcict/elem_helper.h>
#include <tcict/fixture.h>
#include <tcict/skip.h>

#include <functional>
#include <string>
#include <vector>

namespace tcict { namespace tests {

// --- close (eq) : identical tensors ---

template <typename TenT>
void test_close_identical(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_CLOSE
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  TenT tensor1, tensor2;
  tci::eye(ctx, 2, tensor1);
  tci::eye(ctx, 2, tensor2);

  bool are_equal = tci::eq(ctx, tensor1, tensor2, eps);
  TCICT_ASSERT(are_equal == true);
}

// --- close (eq) : different tensors ---

template <typename TenT>
void test_close_different(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_CLOSE
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  TenT tensor1, tensor2;
  tci::eye(ctx, 2, tensor1);
  tci::zeros(ctx, {2, 2}, tensor2);

  bool are_equal = tci::eq(ctx, tensor1, tensor2, eps);
  TCICT_ASSERT(are_equal == false);
}

// --- to_range (to_container) ---

template <typename TenT>
void test_to_range(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_TO_RANGE
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();

  TenT a = tci::template zeros<TenT>(ctx, {2, 3});
  tci::for_each_with_coors(
      ctx, a,
      [](tci::elem_t<TenT>& elem, const tci::elem_coors_t<TenT>& coors) {
        elem = static_cast<tci::elem_t<TenT>>(coors[0] * 3 + coors[1]);
      });

  std::vector<tci::elem_t<TenT>> container(6);

  std::function<std::ptrdiff_t(const tci::elem_coors_t<TenT>&)> row_major_map
      = [](const tci::elem_coors_t<TenT>& coors) -> std::ptrdiff_t {
    return coors[0] * 3 + coors[1];
  };

  tci::to_container(ctx, a, container.begin(), row_major_map);

  for (int i = 0; i < 6; ++i) {
    TCICT_ASSERT_CLOSE(real_part<TenT>(container[i]), static_cast<double>(i), eps);
  }
}

// --- show (does not throw) ---

template <typename TenT>
void test_show(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_SHOW
  return;
#endif
  auto& ctx = fix.context();
  TenT a;
  tci::eye(ctx, 2, a);
  TCICT_ASSERT_NOTHROW(tci::show(ctx, a));
}

// --- convert (same context) ---

template <typename TenT>
void test_convert_same_context(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_CONVERT
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  TenT a, b;
  tci::eye(ctx, 3, a);

  TCICT_ASSERT_NOTHROW(tci::convert(ctx, a, ctx, b));
  TCICT_ASSERT(tci::eq(ctx, a, b, eps));
  TCICT_ASSERT(tci::shape(ctx, a) == tci::shape(ctx, b));

  // Verify independence
  TenT original_b = tci::copy(ctx, b);
  tci::set_elem(ctx, a, {0, 0}, make_elem<TenT>(999.0));
  TCICT_ASSERT(tci::eq(ctx, b, original_b, eps));
}

// --- convert (different contexts) ---

template <typename TenT>
void test_convert_different_context(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_CONVERT
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  TenT a, b;
  tci::eye(ctx, 3, a);

  tci::context_handle_t<TenT> ctx2;
  tci::create_context(ctx2);

  TCICT_ASSERT_NOTHROW(tci::convert(ctx, a, ctx2, b));
  TCICT_ASSERT(tci::eq(ctx, a, b, eps));
  TCICT_ASSERT(tci::shape(ctx, a) == tci::shape(ctx, b));

  tci::destroy_context(ctx2);
}

// --- convert (data integrity) ---

template <typename TenT>
void test_convert_data_integrity(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_CONVERT
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  TenT a;
  tci::zeros(ctx, {2, 3}, a);
  tci::set_elem(ctx, a, {0, 0}, make_elem<TenT>(1.23, 4.56));
  tci::set_elem(ctx, a, {1, 2}, make_elem<TenT>(-7.89, 0.12));

  TenT b;
  tci::context_handle_t<TenT> ctx2;
  tci::create_context(ctx2);
  tci::convert(ctx, a, ctx2, b);

  auto val1 = tci::get_elem(ctx, b, {0, 0});
  TCICT_ASSERT_CLOSE(real_part<TenT>(val1), 1.23, eps);
  TCICT_ASSERT_CLOSE(imag_part<TenT>(val1), 4.56, eps);

  auto val2 = tci::get_elem(ctx, b, {1, 2});
  TCICT_ASSERT_CLOSE(real_part<TenT>(val2), -7.89, eps);
  TCICT_ASSERT_CLOSE(imag_part<TenT>(val2), 0.12, eps);

  tci::destroy_context(ctx2);
}

// --- version ---

template <typename TenT>
void test_version(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_VERSION
  return;
#endif
  std::string ver = tci::template version<TenT>();
  TCICT_ASSERT(!ver.empty());
  // Version string should contain digits and a dot
  bool has_version_pattern = (ver.find_first_of("0123456789") != std::string::npos)
                             && (ver.find('.') != std::string::npos);
  TCICT_ASSERT(has_version_pattern);
}

}}  // namespace tcict::tests
