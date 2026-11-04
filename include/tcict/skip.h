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
// Tests that build and check tensors element by element:
//
// The tests that build or check tensors through the helpers in
// include/tcict/elem_sweep.h and include/tcict/zeroth_order.h, among them the
// concatenate, stack and linear_combine tests and most zeroth-order tests,
// reach set_elem, shape and get_elem through them and, apart from the tests of
// the other constructors, build their inputs with zeros. Defining any of
// TCICT_SKIP_ZEROS, TCICT_SKIP_SET_ELEM, TCICT_SKIP_GET_ELEM or
// TCICT_SKIP_SHAPE therefore skips those tests too, so a gap in one of those
// four APIs is not reported as a failure of the API a test is about. The
// tests using the zeroth-order checks also reach order and size, and
// TCICT_SKIP_ORDER or TCICT_SKIP_SIZE skips them as well. TCICT_HAS_ELEM_SWEEP,
// defined below, holds exactly when none of the four is skipped, and
// TCICT_HAS_ZEROTH_ORDER_SWEEP when in addition neither TCICT_SKIP_ORDER nor
// TCICT_SKIP_SIZE is defined.
//
// The tests that check a diagonal tensor through TCICT_EXPECT_DIAGONAL in
// include/tcict/diagonal.h, test_eye and the diag, svd, trunc_svd, eig and
// eigh tests apart from test_diag_mat_to_vec, reach order, shape, size and get_elem through it,
// and defining any of TCICT_SKIP_ORDER, TCICT_SKIP_SHAPE, TCICT_SKIP_SIZE or
// TCICT_SKIP_GET_ELEM skips them too. TCICT_HAS_DIAGONAL_CHECK, defined below,
// holds exactly when none of those four is skipped.
//
// These macros are fixed when this file is first included, so define the skip
// macros they read before including any tcict header, for instance as compile
// definitions (see "Usage" above).
//
// Precision-conditional skip macros (factorization + iterative APIs):
//
// When the corresponding TCICT_SKIP_<API>_SINGLE_PRECISION macro is
// defined, the matching test function expands TCICT_RETURN_IF_SINGLE_PRECISION
// at the top of its body. That helper is an `if constexpr`-guarded early
// return: when the test is instantiated for a TenT whose
// `tci::real_t<TenT>` is `float` (covering both `float` and
// `std::complex<float>` element types), the return statement executes
// before any assertion runs; for any other instantiation (`real_t<TenT>`
// is anything other than `float`), the `if constexpr`'s discarded-
// statement rule means the return is compiled out entirely (zero
// runtime cost).
//
// Useful when a backend has known precision-specific bugs (e.g., a buggy
// single-precision LAPACK eig) that are difficult to fix, and the goal
// is to keep the suite green for double precision while flagging single
// precision separately.
//
// Available per-API gating macros (10 total, defined by the backend):
//   TCICT_SKIP_EIG_SINGLE_PRECISION,    TCICT_SKIP_EIGH_SINGLE_PRECISION,
//   TCICT_SKIP_EIGVALS_SINGLE_PRECISION, TCICT_SKIP_EIGVALSH_SINGLE_PRECISION,
//   TCICT_SKIP_SVD_SINGLE_PRECISION,    TCICT_SKIP_TRUNC_SVD_SINGLE_PRECISION,
//   TCICT_SKIP_QR_SINGLE_PRECISION,     TCICT_SKIP_LQ_SINGLE_PRECISION,
//   TCICT_SKIP_EXP_SINGLE_PRECISION,    TCICT_SKIP_INVERSE_SINGLE_PRECISION
//
// Limitation: this is a *runtime* skip path. The test body following the
// helper is always part of the function template and gets instantiated
// for every TenT, including the matching single-precision ones. API
// calls inside the body must therefore be valid for that precision at
// compile time. This pattern handles backends with runtime-buggy APIs
// (the call compiles but returns wrong values, e.g., cytnx's
// single-precision eig). For backends that intentionally make a
// (API × precision) combination compile-time-unavailable (via SFINAE /
// `static_assert` / `= delete`), use the whole-API TCICT_SKIP_<API>
// macro instead. A compile-time body-discard variant is planned; see
// issue #50.

// `if constexpr`-guarded early return that fires when `tci::real_t<TenT>`
// is `float` (i.e., TenT's real precision is single — covering both
// `float` and `std::complex<float>` element types). Requires `TenT` to be
// the template parameter of the enclosing function and `tci::real_t<TenT>`
// to be in scope (brought in by tcict/fixture.h via
// <tci/tensor_traits.h>). Used inside test bodies, gated by the per-API
// TCICT_SKIP_*_SINGLE_PRECISION flags listed above. See the Limitation
// note for the runtime-vs-compile-time skip caveat.
//
// The do-while(false) wrapper is macro hygiene: it lets callers append
// the usual statement-terminating `;` and prevents dangling-else issues
// when the helper is used inside an unbraced `if`/`else`.
#define TCICT_RETURN_IF_SINGLE_PRECISION                                       \
  do {                                                                         \
    if constexpr (std::is_same_v<tci::real_t<TenT>, float>) {                  \
      return;                                                                  \
    }                                                                          \
  } while (false)

// Defined when the APIs the element-by-element tests need are all available;
// see "Tests that build and check tensors element by element" above.
#if !defined(TCICT_SKIP_ZEROS) && !defined(TCICT_SKIP_SET_ELEM) &&            \
    !defined(TCICT_SKIP_GET_ELEM) && !defined(TCICT_SKIP_SHAPE)
#define TCICT_HAS_ELEM_SWEEP
#endif

// Defined when, in addition, order and size are available: the zeroth-order
// checks in include/tcict/zeroth_order.h call them, so every test using those
// checks needs them.
#if defined(TCICT_HAS_ELEM_SWEEP) && !defined(TCICT_SKIP_ORDER) &&             \
    !defined(TCICT_SKIP_SIZE)
#define TCICT_HAS_ZEROTH_ORDER_SWEEP
#endif

// Defined when the APIs the diagonal check in include/tcict/diagonal.h calls
// are all available; see "Tests that build and check tensors element by
// element" above.
#if !defined(TCICT_SKIP_ORDER) && !defined(TCICT_SKIP_SHAPE) &&                \
    !defined(TCICT_SKIP_SIZE) && !defined(TCICT_SKIP_GET_ELEM)
#define TCICT_HAS_DIAGONAL_CHECK
#endif
