#pragma once

#include <tci/tensor_traits.h>
#include <complex>
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

/// True when the tensor type's real_t is single-precision (float).
/// Tests can branch on this when accumulation-heavy operations need a
/// coarser tolerance than the fixture's default epsilon.
template <typename TenT>
inline constexpr bool is_single_precision_v
    = std::is_same_v<tci::real_t<TenT>, float>;

}  // namespace tcict
