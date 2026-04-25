#pragma once

// Doctest adapter for bulk registration of TCICT conformance tests.
//
// Usage:
//   #include <tcict/adapters/doctest.h>
//   using MyTen_F  = backend::Tensor<float>;
//   using MyTen_CD = backend::Tensor<std::complex<double>>;
//   TCICT_DOCTEST_REGISTER_REAL(float,   MyTen_F)
//   TCICT_DOCTEST_REGISTER_CPLX(cdouble, MyTen_CD)
//
// `tag` is stringized into the TEST_CASE name to disambiguate across type
// variants; it must be identifier-like.
// `TenT` must be passable as a single macro argument. Types with
// unparenthesized commas (e.g., `std::map<K, V>`) cannot be passed directly;
// in those cases create a `using` alias first.

#include <tcict/fixture.h>
#include <tcict/tests/_list.h>

#include <doctest/doctest.h>

// doctest derives anonymous TEST_CASE registration symbols from __COUNTER__
// when available, and falls back to __LINE__ otherwise. Because the bulk
// macros below expand many TEST_CASE invocations from a single source line,
// the __LINE__ fallback would collide. Fail loudly here instead of producing
// confusing duplicate-symbol errors at link time.
#if !defined(__COUNTER__)
#error "TCICT bulk registration (TCICT_DOCTEST_REGISTER_*) requires __COUNTER__ support; use the per-test TCICT_DOCTEST_CASE pattern on this compiler."
#endif

// Single-case bridge; suitable as the X-callable of TCICT_FOREACH_*.
// Uses the prefixed DOCTEST_TEST_CASE so this adapter still compiles when
// users enable DOCTEST_CONFIG_NO_SHORT_MACRO_NAMES.
#define TCICT_DOCTEST_CASE_BRIDGE(tag, TenT, category, fn) \
  DOCTEST_TEST_CASE("TCICT[" #tag "]: " category " - " #fn) { \
    tcict::tci_test_fixture<TenT> fix_; \
    tcict::tests::fn<TenT>(fix_); \
  }

// Register ALL_TYPES + REAL_ONLY tests for a real TenT.
#define TCICT_DOCTEST_REGISTER_REAL(tag, TenT) \
  TCICT_FOREACH_TEST_ALL_TYPES(TCICT_DOCTEST_CASE_BRIDGE, tag, TenT) \
  TCICT_FOREACH_TEST_REAL_ONLY(TCICT_DOCTEST_CASE_BRIDGE, tag, TenT)

// Register ALL_TYPES + CPLX_ONLY tests for a complex TenT.
#define TCICT_DOCTEST_REGISTER_CPLX(tag, TenT) \
  TCICT_FOREACH_TEST_ALL_TYPES(TCICT_DOCTEST_CASE_BRIDGE, tag, TenT) \
  TCICT_FOREACH_TEST_CPLX_ONLY(TCICT_DOCTEST_CASE_BRIDGE, tag, TenT)
