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
// `TenT` must be a single preprocessor token (a typedef/`using` alias).
// Types with unparenthesized commas (`std::map<K, V>`) cannot be passed
// directly — alias them first.

#include <tcict/fixture.h>
#include <tcict/tests/_list.h>

#include <doctest/doctest.h>

// Single-case bridge; suitable as the X-callable of TCICT_FOREACH_*.
#define TCICT_DOCTEST_CASE_BRIDGE(tag, TenT, category, fn) \
  TEST_CASE("TCICT[" #tag "]: " category " - " #fn) { \
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
