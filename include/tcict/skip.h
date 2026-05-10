#pragma once

#include <type_traits>

// TCICT skip macros for unimplemented functions.
//
// Default: all tests are enabled (no skipping).
// Backends define these to skip specific tests for unimplemented functions.
//
// Usage in backend CMakeLists.txt:
//   target_compile_definitions(my_tests PRIVATE
//       TCICT_SKIP_QR
//       TCICT_SKIP_LQ
//   )
//
// Available skip macros:
//
// Construction:
//   TCICT_SKIP_ALLOCATE, TCICT_SKIP_ZEROS, TCICT_SKIP_FILL, TCICT_SKIP_EYE,
//   TCICT_SKIP_RANDOM, TCICT_SKIP_ASSIGN_FROM_RANGE, TCICT_SKIP_COPY,
//   TCICT_SKIP_MOVE, TCICT_SKIP_CLEAR
//
// Read-only getters:
//   TCICT_SKIP_ORDER, TCICT_SKIP_SHAPE, TCICT_SKIP_SIZE, TCICT_SKIP_SIZE_BYTES,
//   TCICT_SKIP_GET_ELEM
//
// Tensor manipulation:
//   TCICT_SKIP_SET_ELEM, TCICT_SKIP_RESHAPE, TCICT_SKIP_TRANSPOSE,
//   TCICT_SKIP_CPLX_CONJ, TCICT_SKIP_TO_CPLX, TCICT_SKIP_REAL, TCICT_SKIP_IMAG,
//   TCICT_SKIP_EXPAND, TCICT_SKIP_SHRINK, TCICT_SKIP_EXTRACT_SUB,
//   TCICT_SKIP_REPLACE_SUB, TCICT_SKIP_CONCATENATE, TCICT_SKIP_STACK,
//   TCICT_SKIP_FOR_EACH, TCICT_SKIP_FOR_EACH_WITH_COORS
//
// Linear algebra:
//   TCICT_SKIP_DIAG, TCICT_SKIP_NORM, TCICT_SKIP_NORMALIZE, TCICT_SKIP_SCALE,
//   TCICT_SKIP_TRACE, TCICT_SKIP_EXP, TCICT_SKIP_INVERSE, TCICT_SKIP_CONTRACT,
//   TCICT_SKIP_LINEAR_COMBINE, TCICT_SKIP_SVD, TCICT_SKIP_TRUNC_SVD,
//   TCICT_SKIP_QR, TCICT_SKIP_LQ, TCICT_SKIP_EIGVALS, TCICT_SKIP_EIGVALSH,
//   TCICT_SKIP_EIG, TCICT_SKIP_EIGH
//
// I/O:
//   TCICT_SKIP_SAVE, TCICT_SKIP_LOAD
//
// Miscellaneous:
//   TCICT_SKIP_TO_RANGE, TCICT_SKIP_SHOW, TCICT_SKIP_CLOSE, TCICT_SKIP_CONVERT,
//   TCICT_SKIP_VERSION
//
// Precision-conditional skip macros (factorization + iterative APIs):
//
// When defined, the corresponding test functions perform a compile-time
// early return for single-precision instantiations (real_t<TenT> == float,
// covering both `float` and `std::complex<float>`). Useful when a backend
// has known precision-specific bugs (e.g., a buggy single-precision LAPACK
// eig) that are difficult to fix and the goal is to keep the suite green
// for double precision while flagging single precision separately.
//
//   TCICT_SKIP_EIG_SINGLE_PRECISION,    TCICT_SKIP_EIGH_SINGLE_PRECISION,
//   TCICT_SKIP_EIGVALS_SINGLE_PRECISION, TCICT_SKIP_EIGVALSH_SINGLE_PRECISION,
//   TCICT_SKIP_SVD_SINGLE_PRECISION,    TCICT_SKIP_TRUNC_SVD_SINGLE_PRECISION,
//   TCICT_SKIP_QR_SINGLE_PRECISION,     TCICT_SKIP_LQ_SINGLE_PRECISION,
//   TCICT_SKIP_EXP_SINGLE_PRECISION,    TCICT_SKIP_INVERSE_SINGLE_PRECISION
//
// Limitation: this is a *runtime* skip — the test body is still part of
// the function template and gets instantiated for the matching precision.
// API calls inside the body must be valid for that precision at compile
// time. This pattern handles backends with runtime-buggy APIs (the call
// compiles but returns wrong values, e.g., cytnx's single-precision eig).
// For backends that intentionally make a (API × precision) combination
// compile-time-unavailable (via SFINAE / static_assert / = delete), use
// the whole-API TCICT_SKIP_<API> macro instead. A compile-time body-
// discard variant is planned; see issue #50.

// Compile-time-evaluated early return when the elem_t of TenT is single
// precision. Requires `TenT` to be the template parameter of the enclosing
// function and `tci::real_t<TenT>` to be in scope (brought in by
// tcict/fixture.h via <tci/tensor_traits.h>). Used inside test bodies,
// gated by per-API TCICT_SKIP_*_SINGLE_PRECISION flags. See the Limitation
// note above on runtime-vs-compile-time skip semantics.
#define TCICT_RETURN_IF_SINGLE_PRECISION                                       \
  do {                                                                         \
    if constexpr (std::is_same_v<tci::real_t<TenT>, float>) {                  \
      return;                                                                  \
    }                                                                          \
  } while (false)
