#pragma once

// Aggregates the per-header TCAPICT_FOREACH_*_TEST macros into suite-wide
// iterators used by framework adapters (see include/tcapict/adapters/*.h).
//
// Contract: each TCAPICT_FOREACH_*(X, ...) expands X once per test as
//   X(__VA_ARGS__, category_str, test_fn_name)
// Callers must supply at least one forwarded argument (C++17 has no portable
// __VA_OPT__). The documented adapters pass (tag, TenT), satisfying this.

#include <tcapict/tests/construction.h>
#include <tcapict/tests/io_operations.h>
#include <tcapict/tests/linear_algebra.h>
#include <tcapict/tests/miscellaneous.h>
#include <tcapict/tests/read_only_getters.h>
#include <tcapict/tests/tensor_manipulation.h>

// All tests that are safe for both real and complex TenT.
#define TCAPICT_FOREACH_TEST_ALL_TYPES(X, ...) \
  TCAPICT_FOREACH_CONSTRUCTION_TEST_ALL_TYPES(X, __VA_ARGS__) \
  TCAPICT_FOREACH_IO_OPERATIONS_TEST_ALL_TYPES(X, __VA_ARGS__) \
  TCAPICT_FOREACH_LINEAR_ALGEBRA_TEST_ALL_TYPES(X, __VA_ARGS__) \
  TCAPICT_FOREACH_MISCELLANEOUS_TEST_ALL_TYPES(X, __VA_ARGS__) \
  TCAPICT_FOREACH_READ_ONLY_GETTERS_TEST_ALL_TYPES(X, __VA_ARGS__) \
  TCAPICT_FOREACH_TENSOR_MANIPULATION_TEST_ALL_TYPES(X, __VA_ARGS__)

// Tests that lift a real tensor to complex and check the zero imaginary parts
// it gains, so they take only a real TenT.
#define TCAPICT_FOREACH_TEST_REAL_ONLY(X, ...) \
  TCAPICT_FOREACH_MISCELLANEOUS_TEST_REAL_ONLY(X, __VA_ARGS__) \
  TCAPICT_FOREACH_TENSOR_MANIPULATION_TEST_REAL_ONLY(X, __VA_ARGS__)

// Tests whose body is guarded `if constexpr (is_complex_v<TenT>)`; registering
// for real TenT would produce redundant no-op test cases.
#define TCAPICT_FOREACH_TEST_CPLX_ONLY(X, ...) \
  TCAPICT_FOREACH_MISCELLANEOUS_TEST_CPLX_ONLY(X, __VA_ARGS__) \
  TCAPICT_FOREACH_TENSOR_MANIPULATION_TEST_CPLX_ONLY(X, __VA_ARGS__)
