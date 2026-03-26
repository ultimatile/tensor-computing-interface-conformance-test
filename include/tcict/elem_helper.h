#pragma once

#include <tci/tensor_traits.h>
#include <complex>
#include <type_traits>

namespace tcict {

/// Construct an element value from real and imaginary parts.
/// For real tensor types, the imaginary part is ignored.
/// Backends with non-standard element types (e.g. cuDoubleComplex) should specialize this.
template <typename TenT>
tci::elem_t<TenT> make_elem(double re, double im = 0.0) {
  using elem_type = tci::elem_t<TenT>;
  using cplx_type = tci::cplx_t<TenT>;
  if constexpr (std::is_same_v<elem_type, cplx_type>) {
    return elem_type(re, im);
  } else {
    return static_cast<elem_type>(re);
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

}  // namespace tcict
