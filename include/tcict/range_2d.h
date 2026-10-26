#pragma once

#include <tcict/assertion.h>
#include <tcict/elem_helper.h>

#include <cstddef>

namespace tcict {

/// The linear order a test lays a second-order tensor out in for `to_range`
/// and `assign_from_range`. Both APIs take the coordinate-to-index map as a
/// parameter, so this fixes no convention for them; a test picks the one it
/// wants.
enum class layout_2d { row_major, column_major };

namespace detail {

// Sends {row, col} to its index under `layout`. A coordinate coors_2d rejects
// sets `malformed` and maps to index 0, which keeps the backend's access
// inside any non-empty range; what it reads or writes there does not matter,
// since the wrappers below fail on `malformed`.
template <typename TenT>
auto map_2d(std::ptrdiff_t rows, std::ptrdiff_t cols, layout_2d layout,
            bool& malformed) {
  const std::ptrdiff_t row_stride = layout == layout_2d::row_major ? cols : 1;
  const std::ptrdiff_t col_stride = layout == layout_2d::row_major ? 1 : rows;
  return [rows, cols, row_stride, col_stride, &malformed](
             const tci::elem_coors_t<TenT>& coors) -> std::ptrdiff_t {
    std::ptrdiff_t row = 0;
    std::ptrdiff_t col = 0;
    if (!coors_2d(coors, rows, cols, row, col)) {
      malformed = true;
      return 0;
    }
    return row * row_stride + col * col_stride;
  };
}

}  // namespace detail

// Each wrapper is guarded by the skip macro of the API it wraps, and every test
// that calls it is guarded by that macro too. Skipping those tests therefore
// removes the reference to the API, while skipping any other test leaves the
// wrapper available to its callers.

#ifndef TCICT_SKIP_TO_RANGE
/// `to_range` of the second-order tensor `a` under `layout`. Asserts that the
/// call throws no std::exception and that the backend passed only valid
/// coordinates, so a test cannot leave the coordinate check unasserted.
template <typename TenT, typename RandomIt>
void to_range_2d(tci::context_handle_t<TenT>& ctx, const TenT& a,
                 RandomIt first, layout_2d layout) {
  const auto shape = tci::shape(ctx, a);
  TCICT_ASSERT(shape.size() == 2);
  bool malformed = false;
  TCICT_ASSERT_NOTHROW(tci::to_range(
      ctx, a, first,
      detail::map_2d<TenT>(static_cast<std::ptrdiff_t>(shape[0]),
                           static_cast<std::ptrdiff_t>(shape[1]), layout,
                           malformed)));
  TCICT_ASSERT_MSG(!malformed,
                   "to_range_2d: the backend passed a coordinate outside the "
                   "tensor's shape");
}
#endif

#ifndef TCICT_SKIP_ASSIGN_FROM_RANGE
/// `assign_from_range` of a second-order `shape` under `layout`, into `out`.
/// Asserts as to_range_2d does. The result goes to an output parameter so
/// that this wrapper asks no more of TenT than a test calling
/// assign_from_range directly does: assignment from its result.
template <typename TenT, typename RandomIt>
void assign_from_range_2d(tci::context_handle_t<TenT>& ctx,
                          const tci::shape_t<TenT>& shape, RandomIt first,
                          layout_2d layout, TenT& out) {
  TCICT_ASSERT(shape.size() == 2);
  bool malformed = false;
  TCICT_ASSERT_NOTHROW(
      out = tci::template assign_from_range<TenT>(
          ctx, shape, first,
          detail::map_2d<TenT>(static_cast<std::ptrdiff_t>(shape[0]),
                               static_cast<std::ptrdiff_t>(shape[1]), layout,
                               malformed)));
  TCICT_ASSERT_MSG(!malformed,
                   "assign_from_range_2d: the backend passed a coordinate "
                   "outside the given shape");
}
#endif

}  // namespace tcict
