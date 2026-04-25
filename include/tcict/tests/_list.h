#pragma once

// Aggregates the per-header TCICT_FOREACH_*_TEST macros into suite-wide
// iterators used by framework adapters (see include/tcict/adapters/*.h).
//
// Contract: each TCICT_FOREACH_*(X, ...) expands X once per test as
//   X(__VA_ARGS__, category_str, test_fn_name)
// Callers must supply at least one forwarded argument (C++17 has no portable
// __VA_OPT__). The documented adapters pass (tag, TenT), satisfying this.

#include <tcict/tests/construction.h>
#include <tcict/tests/io_operations.h>
#include <tcict/tests/linear_algebra.h>
#include <tcict/tests/miscellaneous.h>
#include <tcict/tests/read_only_getters.h>
#include <tcict/tests/tensor_manipulation.h>

// All tests that are safe for both real and complex TenT.
#define TCICT_FOREACH_TEST_ALL_TYPES(X, ...) \
  TCICT_FOREACH_CONSTRUCTION_TEST_ALL_TYPES(X, __VA_ARGS__) \
  TCICT_FOREACH_IO_OPERATIONS_TEST_ALL_TYPES(X, __VA_ARGS__) \
  TCICT_FOREACH_LINEAR_ALGEBRA_TEST_ALL_TYPES(X, __VA_ARGS__) \
  TCICT_FOREACH_MISCELLANEOUS_TEST_ALL_TYPES(X, __VA_ARGS__) \
  TCICT_FOREACH_READ_ONLY_GETTERS_TEST_ALL_TYPES(X, __VA_ARGS__) \
  TCICT_FOREACH_TENSOR_MANIPULATION_TEST_ALL_TYPES(X, __VA_ARGS__)

// Tests meaningful only for real TenT (e.g., tci::to_cplx input must be real).
#define TCICT_FOREACH_TEST_REAL_ONLY(X, ...) \
  TCICT_FOREACH_TENSOR_MANIPULATION_TEST_REAL_ONLY(X, __VA_ARGS__)

// Tests whose body is guarded `if constexpr (is_complex_v<TenT>)`; registering
// for real TenT would produce redundant no-op test cases.
#define TCICT_FOREACH_TEST_CPLX_ONLY(X, ...) \
  TCICT_FOREACH_TENSOR_MANIPULATION_TEST_CPLX_ONLY(X, __VA_ARGS__)
