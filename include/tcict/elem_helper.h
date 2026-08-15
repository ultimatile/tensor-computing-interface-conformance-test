#pragma once

#include <tci/tensor_traits.h>
#include <complex>
#include <cstddef>
#include <functional>
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

/// Row-major coordinate-to-index map for a second-order tensor of `ncols`
/// columns, in the form `to_range` and `assign_from_range` accept.
/// These APIs take the map from the caller precisely so no convention is
/// privileged; this is a convenience for the tests that want row-major, and a
/// test wanting another convention passes its own lambda instead.
template <typename TenT>
std::function<std::ptrdiff_t(const tci::elem_coors_t<TenT>&)> row_major_2d(
    std::ptrdiff_t ncols) {
  return [ncols](const tci::elem_coors_t<TenT>& coors) -> std::ptrdiff_t {
    return static_cast<std::ptrdiff_t>(coors[0]) * ncols
           + static_cast<std::ptrdiff_t>(coors[1]);
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
