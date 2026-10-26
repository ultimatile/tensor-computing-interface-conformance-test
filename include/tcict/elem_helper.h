#pragma once

#include <tci/tensor_traits.h>
#include <complex>
#include <cstddef>
#include <type_traits>

namespace tcict {

/// Construct an element value from real and imaginary parts.
/// For real tensor types, the imaginary part is ignored.
/// Backends with non-standard element types (e.g. cuDoubleComplex) should specialize this.
template <typename TenT>
tci::elem_t<TenT> make_elem(double real, double imag = 0.0) {
  using elem_type = tci::elem_t<TenT>;
  using cplx_type = tci::cplx_t<TenT>;
  if constexpr (std::is_same_v<elem_type, cplx_type>) {
    return elem_type(real, imag);
  } else {
    // NOLINTNEXTLINE(bugprone-narrowing-conversions) -- elem_t is always floating-point in TCI
    return static_cast<elem_type>(real);
  }
}

/// Reads `coors` as a {row, col} coordinate of a `rows` x `cols` tensor.
/// Returns false, leaving `row` and `col` unspecified, when it is not one.
///
/// Callbacks the backend invokes (coordinate maps, generators, traversal
/// callbacks) run inside the backend's frames, which a conformance suite
/// cannot assume are able to carry an exception. So they neither assert, nor
/// allocate, nor index with a coordinate they have not checked: they record
/// into flags and counters, and the test asserts after the call returns. A
/// callback that receives a coordinate passes it through this first.
template <typename Coors>
bool coors_2d(const Coors& coors, std::ptrdiff_t rows, std::ptrdiff_t cols,
              std::ptrdiff_t& row, std::ptrdiff_t& col) {
  if (coors.size() != 2) {
    return false;
  }
  row = static_cast<std::ptrdiff_t>(coors[0]);
  col = static_cast<std::ptrdiff_t>(coors[1]);
  return row >= 0 && row < rows && col >= 0 && col < cols;
}

/// Coordinate-to-index maps for a `rows` x `cols` tensor, in the form
/// `to_range` and `assign_from_range` accept. Both APIs take the map as a
/// parameter, so these fix no convention for them; a test picks the one it
/// wants.
///
/// A coordinate `coors_2d` rejects sets `malformed` and maps to index 0, so
/// the backend's access stays inside the range; what it reads or writes there
/// does not matter, since the test fails on `malformed`.
template <typename TenT>
auto row_major_2d(std::ptrdiff_t rows, std::ptrdiff_t cols, bool& malformed) {
  return [rows, cols, &malformed](
             const tci::elem_coors_t<TenT>& coors) -> std::ptrdiff_t {
    std::ptrdiff_t row = 0;
    std::ptrdiff_t col = 0;
    if (!coors_2d(coors, rows, cols, row, col)) {
      malformed = true;
      return 0;
    }
    return row * cols + col;
  };
}

template <typename TenT>
auto column_major_2d(std::ptrdiff_t rows, std::ptrdiff_t cols, bool& malformed) {
  return [rows, cols, &malformed](
             const tci::elem_coors_t<TenT>& coors) -> std::ptrdiff_t {
    std::ptrdiff_t row = 0;
    std::ptrdiff_t col = 0;
    if (!coors_2d(coors, rows, cols, row, col)) {
      malformed = true;
      return 0;
    }
    return col * rows + row;
  };
}

/// Extract the real part of an element as double.
/// Backends with non-standard element types should specialize.
template <typename TenT>
double real_part(tci::elem_t<TenT> elem) {
  using elem_type = tci::elem_t<TenT>;
  using cplx_type = tci::cplx_t<TenT>;
  if constexpr (std::is_same_v<elem_type, cplx_type>) {
    return std::real(elem);
  } else {
    return static_cast<double>(elem);
  }
}

/// Extract the imaginary part of an element as double.
/// Returns 0.0 for real tensor types.
template <typename TenT>
double imag_part(tci::elem_t<TenT> elem) {
  using elem_type = tci::elem_t<TenT>;
  using cplx_type = tci::cplx_t<TenT>;
  if constexpr (std::is_same_v<elem_type, cplx_type>) {
    return std::imag(elem);
  } else {
    return 0.0;
  }
}

/// True when the tensor type's element type is complex.
/// Tests branch on this to guard assertions that only make sense for complex
/// elements (e.g. verifying a non-zero imaginary part).
template <typename TenT>
inline constexpr bool is_complex_v
    = std::is_same_v<tci::elem_t<TenT>, tci::cplx_t<TenT>>;

/// True when the tensor type's real_t is single-precision (float).
/// Tests can branch on this when accumulation-heavy operations need a
/// coarser tolerance than the fixture's default epsilon.
template <typename TenT>
inline constexpr bool is_single_precision_v
    = std::is_same_v<tci::real_t<TenT>, float>;

}  // namespace tcict
