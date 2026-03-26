#pragma once

#include <tcict/assertion.h>
#include <tcict/elem_helper.h>
#include <tcict/fixture.h>
#include <tcict/skip.h>

#include <string>

namespace tcict { namespace tests {

// --- save/load roundtrip ---

template <typename TenT>
void test_save_load_roundtrip(tci_test_fixture<TenT>& fix) {
#if defined(TCICT_SKIP_SAVE) || defined(TCICT_SKIP_LOAD)
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();

  TenT tensor;
  tci::zeros(ctx, {2, 2}, tensor);
  tci::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(1.0, 0.0));
  tci::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(1.0, 0.0));

  std::string filepath = "/tmp/tcict_test_roundtrip";

  TCICT_ASSERT_NOTHROW(tci::save(ctx, tensor, filepath));

  TenT loaded_tensor;
  TCICT_ASSERT_NOTHROW(tci::load(ctx, filepath, loaded_tensor));

  TCICT_ASSERT(tci::shape(ctx, loaded_tensor) == tci::shape(ctx, tensor));

  auto val00 = tci::get_elem(ctx, loaded_tensor, {0, 0});
  auto val11 = tci::get_elem(ctx, loaded_tensor, {1, 1});
  TCICT_ASSERT_CLOSE(real_part<TenT>(val00), 1.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(val11), 1.0, eps);
}

// --- load verifies data integrity against original ---

template <typename TenT>
void test_load_data_integrity(tci_test_fixture<TenT>& fix) {
#if defined(TCICT_SKIP_SAVE) || defined(TCICT_SKIP_LOAD)
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();

  TenT original;
  tci::eye(ctx, 2, original);
  std::string filepath = "/tmp/tcict_test_load_integrity";
  tci::save(ctx, original, filepath);

  TenT loaded;
  TCICT_ASSERT_NOTHROW(tci::load(ctx, filepath, loaded));
  TCICT_ASSERT(tci::shape(ctx, original) == tci::shape(ctx, loaded));
  TCICT_ASSERT(tci::eq(ctx, original, loaded, eps));
}

}}  // namespace tcict::tests
