# Tensor Computing Interface Conformance Test Suite (TCICT)

Header-only, test-framework-agnostic conformance tests for the [Tensor Computing Interface (TCI)](https://arxiv.org/abs/2512.23917) specification.

## Overview

TCICT verifies that a TCI backend correctly implements the spec-defined API. Tests are written as template functions that throw `tcict::assertion_error` on failure, making them usable with any test framework (doctest, Google Test, Catch2) or standalone.

Known TCI backend implementations:
- [Cytnx backend](https://github.com/r-ccs-cms/tensor-computing-interface-backend-cytnx)

## Usage

### 1. Add as a git submodule

```bash
git submodule add https://github.com/ultimatile/tensor-computing-interface-conformance-test.git external/tcict
```

### 2. Add the include path

```cmake
target_include_directories(my_tests PRIVATE ${CMAKE_SOURCE_DIR}/external/tcict/include)
```

### 3. Register tests

Each backend provides a small bridge file that maps TCICT test functions to the backend's test framework. Two options are available, depending on the backend's needs.

#### Option A — Bulk registration (recommended)

Use the provided doctest adapter to register all applicable tests for each type variant with a single macro call:

The backend's TCI header must be included **before** the adapter, because some test templates call non-dependent `tci::` functions (e.g., `tci::create_context` in the fixture constructor) that are resolved at the first parsing phase.

```cpp
#include <my_backend/tci.h>           // backend first: declares tci::create_context, tci::zeros, ...
#include <tcict/adapters/doctest.h>   // then the adapter, which pulls in the test templates

#include <complex>

using MyTen_F  = my_backend::Tensor<float>;
using MyTen_D  = my_backend::Tensor<double>;
using MyTen_CF = my_backend::Tensor<std::complex<float>>;
using MyTen_CD = my_backend::Tensor<std::complex<double>>;

TCICT_DOCTEST_REGISTER_REAL(float,   MyTen_F)
TCICT_DOCTEST_REGISTER_REAL(double,  MyTen_D)
TCICT_DOCTEST_REGISTER_CPLX(cfloat,  MyTen_CF)
TCICT_DOCTEST_REGISTER_CPLX(cdouble, MyTen_CD)
```

`TCICT_DOCTEST_REGISTER_REAL` registers all `ALL_TYPES` tests plus the `REAL_ONLY` tests (currently `test_to_cplx_outofplace`, `test_to_cplx_inplace`).
`TCICT_DOCTEST_REGISTER_CPLX` registers all `ALL_TYPES` tests plus the `CPLX_ONLY` tests (currently `test_to_cplx_complex_to_complex`).

Constraints:
- `tag` must be identifier-like (stringized into the `TEST_CASE` name to disambiguate type variants).
- `TenT` must be passable as a single macro argument. Types with unparenthesized commas (for example, `std::map<K, V>`) cannot be passed directly, so use a `using` alias in those cases.

#### Option B — Per-test registration (fine-grained control)

For selectively registering specific tests (e.g., during incremental backend development), use the per-test bridge:

```cpp
#include <my_backend/tci.h>           // backend first (see note in Option A)
#include <tcict/tcict.h>
#include <doctest/doctest.h>

#include <complex>

#define TCICT_DOCTEST_CASE(category, test_func, TenT)          \
  DOCTEST_TEST_CASE("TCICT: " category " - " #test_func) {     \
    tcict::tci_test_fixture<TenT> fix;                         \
    tcict::tests::test_func<TenT>(fix);                        \
  }

using MyTensor = my_backend::Tensor<std::complex<double>>;

TCICT_DOCTEST_CASE("construction", test_zeros, MyTensor)
TCICT_DOCTEST_CASE("construction", test_eye, MyTensor)
// ... register more tests
```

### 4. Skip unimplemented functions

Define `TCICT_SKIP_*` macros to opt out of tests for functions a backend has not yet implemented. Each skip macro **excludes — at preprocessing time — the body of the test(s) it directly guards**, removing those references to the corresponding `tci::*` API:

```cmake
target_compile_definitions(my_tests PRIVATE
    TCICT_SKIP_QR
    TCICT_SKIP_LQ
)
```

A skipped test still compiles, registers as a test case, and immediately passes (it has no assertions). See `include/tcict/skip.h` for the full list.

**Caveat: skip is per-test, not per-API.** The same `tci::*` function may be called as a setup / verification helper in unrelated tests. For example, `tci::get_elem` is used inside many tests beyond the one guarded by `TCICT_SKIP_GET_ELEM`. Skipping a single feature therefore does not guarantee that every reference to its API disappears — to compile against a backend that omits a particular declaration, you may need to skip every test that uses it.

**Baseline requirement.** Some functions must always be declared, regardless of skip macros, because they are used as setup / teardown by every test fixture:

- `tci::create_context`, `tci::destroy_context` (called by `tci_test_fixture`'s constructor/destructor)
- `tci::tensor_traits<TenT>` and the type aliases it exports

A backend that does not provide these cannot use TCICT at all, with or without skip macros.

## Test Categories

| Category | Header | Functions |
|---|---|---|
| Construction | `tests/construction.h` | allocate, zeros, fill, eye, random, copy, move, clear, assign_from_range |
| Read-only getters | `tests/read_only_getters.h` | order, shape, size, size_bytes, get_elem |
| Tensor manipulation | `tests/tensor_manipulation.h` | reshape, transpose, shrink, expand, concatenate, extract_sub, replace_sub, diag, for_each, cplx_conj, to_cplx, real, imag |
| Linear algebra | `tests/linear_algebra.h` | norm, normalize, linear_combine, contract, QR, LQ, truncated SVD, eig, eigh, exp, inverse |
| I/O | `tests/io_operations.h` | save, load |
| Miscellaneous | `tests/miscellaneous.h` | close/eq, to_range, show, convert, version |

## Architecture

- **`assertion.h`** -- Exception-based assertion macros (`TCICT_ASSERT`, `TCICT_ASSERT_CLOSE`, etc.)
- **`fixture.h`** -- `tci_test_fixture<TenT>` template managing context lifecycle and epsilon
- **`elem_helper.h`** -- `make_elem<TenT>(re, im)` and `real_part`/`imag_part` helpers for backend-agnostic element construction
- **`skip.h`** -- `TCICT_SKIP_*` opt-out macros for unimplemented functions
- **`tests/_list.h`** -- X-macro aggregators (`TCICT_FOREACH_TEST_ALL_TYPES`, `_REAL_ONLY`, `_CPLX_ONLY`) used by framework adapters
- **`adapters/doctest.h`** -- Bulk-registration bridge for doctest (`TCICT_DOCTEST_REGISTER_REAL/CPLX`)

## Requirements

- C++17
- TCI headers (`tci/tensor_traits.h` and backend-specific trait specialization)
