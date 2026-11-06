#pragma once

// Doctest adapter for bulk registration of TCAPICT conformance tests.
//
// The suite includes <tcapi/tcapi.h> itself, so this adapter may come before
// or after the implementation's header. Define any macro that header reads
// before the first include of either.
//
// Usage:
//   #include <tcapict/adapters/doctest.h>
//   using MyTen_F  = backend::Tensor<float>;
//   using MyTen_CD = backend::Tensor<std::complex<double>>;
//   TCAPICT_DOCTEST_REGISTER_REAL(float,   MyTen_F)
//   TCAPICT_DOCTEST_REGISTER_CPLX(cdouble, MyTen_CD)
//
// `tag` is stringized into the TEST_CASE name to disambiguate across type
// variants; it must be identifier-like.
// `TenT` must be passable as a single macro argument. Types with
// unparenthesized commas (e.g., `std::map<K, V>`) cannot be passed directly;
// in those cases create a `using` alias first.

#include <tcapict/fixture.h>
#include <tcapict/tests/_list.h>

#include <doctest/doctest.h>

// doctest derives anonymous TEST_CASE registration symbols from __COUNTER__
// when available, and falls back to __LINE__ otherwise. Because the bulk
// macros below expand many TEST_CASE invocations from a single source line,
// the __LINE__ fallback would collide. Fail loudly here instead of producing
// confusing duplicate-symbol errors at link time.
#if !defined(__COUNTER__)
#error "TCAPICT bulk registration (TCAPICT_DOCTEST_REGISTER_*) requires __COUNTER__ support; use the per-test TCAPICT_DOCTEST_CASE pattern on this compiler."
#endif

// Single-case bridge; suitable as the X-callable of TCAPICT_FOREACH_*.
// Uses the prefixed DOCTEST_TEST_CASE so this adapter still compiles when
// users enable DOCTEST_CONFIG_NO_SHORT_MACRO_NAMES.
#define TCAPICT_DOCTEST_CASE_BRIDGE(tag, TenT, category, fn) \
  DOCTEST_TEST_CASE("TCAPICT[" #tag "]: " category " - " #fn) { \
    tcapict::tcapi_test_fixture<TenT> fix_; \
    tcapict::tests::fn<TenT>(fix_); \
  }

// Register ALL_TYPES + REAL_ONLY tests for a real TenT.
#define TCAPICT_DOCTEST_REGISTER_REAL(tag, TenT) \
  TCAPICT_FOREACH_TEST_ALL_TYPES(TCAPICT_DOCTEST_CASE_BRIDGE, tag, TenT) \
  TCAPICT_FOREACH_TEST_REAL_ONLY(TCAPICT_DOCTEST_CASE_BRIDGE, tag, TenT)

// Register ALL_TYPES + CPLX_ONLY tests for a complex TenT.
#define TCAPICT_DOCTEST_REGISTER_CPLX(tag, TenT) \
  TCAPICT_FOREACH_TEST_ALL_TYPES(TCAPICT_DOCTEST_CASE_BRIDGE, tag, TenT) \
  TCAPICT_FOREACH_TEST_CPLX_ONLY(TCAPICT_DOCTEST_CASE_BRIDGE, tag, TenT)
