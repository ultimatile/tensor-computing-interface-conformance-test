#pragma once

#include <type_traits>

// TCAPICT skip macros for unimplemented functions.
//
// Default: all tests are enabled (no skipping).
// Backends define these to skip specific tests for unimplemented functions.
//
// Usage in backend CMakeLists.txt:
//   target_compile_definitions(my_tests PRIVATE
//       TCAPICT_SKIP_QR
//       TCAPICT_SKIP_LQ
//   )
//
// Available skip macros:
//
// Construction:
//   TCAPICT_SKIP_ALLOCATE, TCAPICT_SKIP_ZEROS, TCAPICT_SKIP_FILL, TCAPICT_SKIP_EYE,
//   TCAPICT_SKIP_RANDOM, TCAPICT_SKIP_ASSIGN_FROM_RANGE, TCAPICT_SKIP_COPY,
//   TCAPICT_SKIP_MOVE, TCAPICT_SKIP_CLEAR
//
// Read-only getters:
//   TCAPICT_SKIP_ORDER, TCAPICT_SKIP_SHAPE, TCAPICT_SKIP_SIZE, TCAPICT_SKIP_SIZE_BYTES,
//   TCAPICT_SKIP_GET_ELEM
//
// Tensor manipulation:
//   TCAPICT_SKIP_SET_ELEM, TCAPICT_SKIP_RESHAPE, TCAPICT_SKIP_TRANSPOSE,
//   TCAPICT_SKIP_CPLX_CONJ, TCAPICT_SKIP_TO_CPLX, TCAPICT_SKIP_REAL, TCAPICT_SKIP_IMAG,
//   TCAPICT_SKIP_EXPAND, TCAPICT_SKIP_SHRINK, TCAPICT_SKIP_EXTRACT_SUB,
//   TCAPICT_SKIP_REPLACE_SUB, TCAPICT_SKIP_CONCATENATE, TCAPICT_SKIP_STACK,
//   TCAPICT_SKIP_FOR_EACH, TCAPICT_SKIP_FOR_EACH_WITH_COORS
//
// Linear algebra:
//   TCAPICT_SKIP_DIAG, TCAPICT_SKIP_NORM, TCAPICT_SKIP_NORMALIZE, TCAPICT_SKIP_SCALE,
//   TCAPICT_SKIP_TRACE, TCAPICT_SKIP_EXP, TCAPICT_SKIP_INVERSE, TCAPICT_SKIP_CONTRACT,
//   TCAPICT_SKIP_LINEAR_COMBINE, TCAPICT_SKIP_SVD, TCAPICT_SKIP_TRUNC_SVD,
//   TCAPICT_SKIP_QR, TCAPICT_SKIP_LQ, TCAPICT_SKIP_EIGVALS, TCAPICT_SKIP_EIGVALSH,
//   TCAPICT_SKIP_EIG, TCAPICT_SKIP_EIGH
//
// I/O:
//   TCAPICT_SKIP_SAVE, TCAPICT_SKIP_LOAD
//
// Miscellaneous:
//   TCAPICT_SKIP_TO_RANGE, TCAPICT_SKIP_SHOW, TCAPICT_SKIP_CLOSE, TCAPICT_SKIP_CONVERT,
//   TCAPICT_SKIP_VERSION
//
// Type system:
//   TCAPICT_SKIP_SUPPORTED_TEN_TYPES, for a backend that does not yet provide the
//   tcapi::supported_ten_types_t catalog. test_convert_precision, which takes
//   its destination types from the catalog, is skipped with it.
//
// Tests that build and check tensors element by element:
//
// The tests that build or check tensors through the helpers in
// include/tcapict/elem_sweep.h and include/tcapict/zeroth_order.h, among them the
// concatenate, stack and linear_combine tests and most zeroth-order tests,
// reach set_elem, shape and get_elem through them and, apart from the tests of
// the other constructors, build their inputs with zeros. Defining any of
// TCAPICT_SKIP_ZEROS, TCAPICT_SKIP_SET_ELEM, TCAPICT_SKIP_GET_ELEM or
// TCAPICT_SKIP_SHAPE therefore skips those tests too, so a gap in one of those
// four APIs is not reported as a failure of the API a test is about. The
// tests using the zeroth-order checks also reach order and size, and
// TCAPICT_SKIP_ORDER or TCAPICT_SKIP_SIZE skips them as well. TCAPICT_HAS_ELEM_SWEEP,
// defined below, holds exactly when none of the four is skipped, and
// TCAPICT_HAS_ZEROTH_ORDER_SWEEP when in addition neither TCAPICT_SKIP_ORDER nor
// TCAPICT_SKIP_SIZE is defined.
//
// The tests that check a diagonal tensor through TCAPICT_EXPECT_DIAGONAL in
// include/tcapict/diagonal.h, test_eye and the diag, svd, trunc_svd, eig and
// eigh tests apart from test_diag_mat_to_vec and
// test_diag_mat_to_vec_from_diagonal, reach order, shape, size and
// get_elem through it, and defining any of TCAPICT_SKIP_ORDER, TCAPICT_SKIP_SHAPE,
// TCAPICT_SKIP_SIZE or TCAPICT_SKIP_GET_ELEM skips them too.
// TCAPICT_HAS_DIAGONAL_CHECK, defined below, holds exactly when none of those
// four is skipped.
//
// These macros are fixed when this file is first included, so define the skip
// macros they read before including any tcapict header, for instance as compile
// definitions (see "Usage" above).
//
// Precision-conditional skip macros (factorization + iterative APIs):
//
// When the corresponding TCAPICT_SKIP_<API>_SINGLE_PRECISION macro is
// defined, the matching test function expands TCAPICT_RETURN_IF_SINGLE_PRECISION
// at the top of its body. That helper is an `if constexpr`-guarded early
// return: when the test is instantiated for a TenT whose
// `tcapi::real_t<TenT>` is `float` (covering both `float` and
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
//   TCAPICT_SKIP_EIG_SINGLE_PRECISION,    TCAPICT_SKIP_EIGH_SINGLE_PRECISION,
//   TCAPICT_SKIP_EIGVALS_SINGLE_PRECISION, TCAPICT_SKIP_EIGVALSH_SINGLE_PRECISION,
//   TCAPICT_SKIP_SVD_SINGLE_PRECISION,    TCAPICT_SKIP_TRUNC_SVD_SINGLE_PRECISION,
//   TCAPICT_SKIP_QR_SINGLE_PRECISION,     TCAPICT_SKIP_LQ_SINGLE_PRECISION,
//   TCAPICT_SKIP_EXP_SINGLE_PRECISION,    TCAPICT_SKIP_INVERSE_SINGLE_PRECISION
//
// Limitation: this is a *runtime* skip path. The test body following the
// helper is always part of the function template and gets instantiated
// for every TenT, including the matching single-precision ones. API
// calls inside the body must therefore be valid for that precision at
// compile time. This pattern handles backends with runtime-buggy APIs
// (the call compiles but returns wrong values, e.g., cytnx's
// single-precision eig). For backends that intentionally make a
// (API × precision) combination compile-time-unavailable (via SFINAE /
// `static_assert` / `= delete`), use the whole-API TCAPICT_SKIP_<API>
// macro instead. A compile-time body-discard variant is planned; see
// https://github.com/ultimatile/tcapi-conformance-test/issues/50.

// `if constexpr`-guarded early return that fires when `tcapi::real_t<TenT>`
// is `float` (i.e., TenT's real precision is single — covering both
// `float` and `std::complex<float>` element types). Requires `TenT` to be
// the template parameter of the enclosing function and `tcapi::real_t<TenT>`
// to be in scope (brought in by tcapict/fixture.h via
// <tcapi/tcapi.h>). Used inside test bodies, gated by the per-API
// TCAPICT_SKIP_*_SINGLE_PRECISION flags listed above. See the Limitation
// note for the runtime-vs-compile-time skip caveat.
//
// The do-while(false) wrapper is macro hygiene: it lets callers append
// the usual statement-terminating `;` and prevents dangling-else issues
// when the helper is used inside an unbraced `if`/`else`.
#define TCAPICT_RETURN_IF_SINGLE_PRECISION                                     \
  do {                                                                         \
    if constexpr (std::is_same_v<tcapi::real_t<TenT>, float>) {                \
      return;                                                                  \
    }                                                                          \
  } while (false)

// Defined when the APIs the element-by-element tests need are all available;
// see "Tests that build and check tensors element by element" above.
#if !defined(TCAPICT_SKIP_ZEROS) && !defined(TCAPICT_SKIP_SET_ELEM) &&        \
    !defined(TCAPICT_SKIP_GET_ELEM) && !defined(TCAPICT_SKIP_SHAPE)
#define TCAPICT_HAS_ELEM_SWEEP
#endif

// Defined when, in addition, order and size are available: the zeroth-order
// checks in include/tcapict/zeroth_order.h call them, so every test using those
// checks needs them.
#if defined(TCAPICT_HAS_ELEM_SWEEP) && !defined(TCAPICT_SKIP_ORDER) &&         \
    !defined(TCAPICT_SKIP_SIZE)
#define TCAPICT_HAS_ZEROTH_ORDER_SWEEP
#endif

// Defined when the APIs the diagonal check in include/tcapict/diagonal.h calls
// are all available; see "Tests that build and check tensors element by
// element" above.
#if !defined(TCAPICT_SKIP_ORDER) && !defined(TCAPICT_SKIP_SHAPE) &&            \
    !defined(TCAPICT_SKIP_SIZE) && !defined(TCAPICT_SKIP_GET_ELEM)
#define TCAPICT_HAS_DIAGONAL_CHECK
#endif
