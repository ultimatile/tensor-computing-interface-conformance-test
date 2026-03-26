#pragma once

#include <tci/tensor_traits.h>

namespace tcict {

/// Test fixture that manages a TCI context.
/// Backends may specialize this for custom context creation (e.g. GPU) or epsilon.
template <typename TenT>
struct tci_test_fixture {
  tci::context_handle_t<TenT> ctx;

  tci_test_fixture() { tci::create_context(ctx); }
  ~tci_test_fixture() { tci::destroy_context(ctx); }

  tci_test_fixture(const tci_test_fixture&) = delete;
  tci_test_fixture& operator=(const tci_test_fixture&) = delete;

  auto& context() { return ctx; }

  auto epsilon() -> tci::real_t<TenT> { return static_cast<tci::real_t<TenT>>(1e-10); }
};

}  // namespace tcict
