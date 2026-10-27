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

/// Builds a coordinate from loop indices, converting each to the backend's
/// `elem_coor_t`. A braced list of `std::size_t` indices is a narrowing
/// conversion, which list-initialization rejects, whenever `elem_coor_t`
/// cannot represent every `std::size_t` value.
template <typename TenT, typename... Ix>
tci::elem_coors_t<TenT> coors_of(Ix... ix) {
  return {static_cast<tci::elem_coor_t<TenT>>(ix)...};
}

/// Reads `coors` as a {row, col} coordinate of a `rows` x `cols` tensor.
/// Returns false, leaving `row` and `col` unspecified, when it is not one.
/// "Callbacks the backend invokes" in CONTRIBUTING.md says which callbacks
/// call this.
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
