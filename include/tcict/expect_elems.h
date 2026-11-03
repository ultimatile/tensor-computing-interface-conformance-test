#pragma once

#include <tcict/assertion.h>
#include <tcict/elem_helper.h>
#include <tcict/fixture.h>

#include <cstddef>

namespace tcict {

// Compiled exactly when at least one of its callers is: the concatenate,
// stack and linear_combine tests.
#if !defined(TCICT_SKIP_CONCATENATE) || !defined(TCICT_SKIP_STACK) ||         \
    !defined(TCICT_SKIP_LINEAR_COMBINE)
/// Asserts that `tensor` has `shape`, then that the element at every
/// coordinate of that shape has real part `value_at(coors)` and, for a complex
/// element type, imaginary part zero. The shape is asserted first, so the
/// sweep reads only coordinates the tensor has.
template <typename TenT, typename ValueAt>
void expect_elems(tci_test_fixture<TenT> &fix, const TenT &tensor,
                  const tci::shape_t<TenT> &shape, ValueAt value_at) {
  auto &ctx = fix.context();
  auto tol = tolerance(fix, tol_category::elementwise);
  TCICT_ASSERT(tci::shape(ctx, tensor) == shape);

  const std::size_t order = shape.size();
  for (std::size_t b = 0; b < order; ++b) {
    if (shape[b] == 0) {
      return;
    }
  }
  tci::elem_coors_t<TenT> coors(order, 0);
  while (true) {
    auto elem = tci::get_elem(ctx, tensor, coors);
    TCICT_ASSERT_CLOSE(real_part<TenT>(elem),
                       static_cast<double>(value_at(coors)), tol);
    if constexpr (is_complex_v<TenT>) {
      TCICT_ASSERT_CLOSE(imag_part<TenT>(elem), 0.0, tol);
    }
    // Advance the last bond fastest; done once every bond has wrapped.
    std::size_t b = order;
    while (true) {
      if (b == 0) {
        return;
      }
      --b;
      ++coors[b];
      if (static_cast<std::size_t>(coors[b]) <
          static_cast<std::size_t>(shape[b])) {
        break;
      }
      coors[b] = 0;
    }
  }
}
#endif

} // namespace tcict
