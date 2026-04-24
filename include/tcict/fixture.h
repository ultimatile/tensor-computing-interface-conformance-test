#pragma once

#include <tci/tensor_traits.h>

#include <limits>
#include <type_traits>

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

  /// Comparison tolerance sized to the element type's real precision.
  /// The known-type branches match values empirically sufficient for the
  /// small test fixtures in this suite; the fallback scales machine epsilon
  /// to absorb O(N) accumulation errors for unfamiliar real_t types.
  auto epsilon() -> tci::real_t<TenT> {
    using real_type = tci::real_t<TenT>;
    if constexpr (std::is_same_v<real_type, float>) {
      return 1e-5F;
    } else if constexpr (std::is_same_v<real_type, double>) {
      return 1e-10;
    } else if constexpr (std::is_same_v<real_type, long double>) {
      return 1e-15L;
    } else {
      return std::numeric_limits<real_type>::epsilon() * static_cast<real_type>(100);
    }
  }
};

}  // namespace tcict
