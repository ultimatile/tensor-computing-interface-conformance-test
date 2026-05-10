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

Use the provided doctest adapter to register all applicable conformance tests for each tensor type variant in a single macro call.

The backend's TCI header must be included **before** the adapter, because some test templates call non-dependent `tci::` functions (e.g., `tci::create_context` in the fixture constructor) that are resolved during the first parsing phase.

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

`TCICT_DOCTEST_REGISTER_REAL` registers every test that applies to real tensors (the full `ALL_TYPES` set plus `REAL_ONLY` tests such as `test_to_cplx_outofplace` / `test_to_cplx_inplace`).
`TCICT_DOCTEST_REGISTER_CPLX` registers every test that applies to complex tensors (`ALL_TYPES` plus `CPLX_ONLY` tests such as `test_to_cplx_complex_to_complex`).

Constraints:
- `tag` must be identifier-like; it is stringized into each `TEST_CASE` name to disambiguate type variants.
- `TenT` must be passable as a single macro argument. Types with unparenthesized commas (e.g., `std::map<K, V>`) cannot be passed directly — introduce a `using` alias first.

<details>
<summary>Per-test registration (advanced, rarely needed)</summary>

If you need fine-grained control — for example, registering an individual test during incremental backend development before bulk registration is viable — define a per-test bridge instead:

```cpp
#include <my_backend/tci.h>           // backend first
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
TCICT_DOCTEST_CASE("construction", test_eye,   MyTensor)
// ... register more tests
```

Prefer the bulk macros above unless you have a concrete reason to drop down to this form.

</details>

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

### 5. Skip tests for a specific precision (known precision-specific bugs)

For factorization / iterative APIs where a backend has a known **precision-specific runtime bug** (e.g., a single-precision `eig` that compiles fine but returns wrong values), define the corresponding `TCICT_SKIP_<API>_SINGLE_PRECISION` macro to skip those tests **only for single-precision instantiations** (TenT whose `tci::real_t<TenT>` is `float`, covering both `float` and `std::complex<float>`):

```cmake
target_compile_definitions(my_tests PRIVATE
    TCICT_SKIP_EIG_SINGLE_PRECISION
    TCICT_SKIP_EIGH_SINGLE_PRECISION
)
```

The corresponding double-precision instantiations continue to run.

Available flags (10 APIs covered, factorization + iterative classes per `fixture.h`):

- `TCICT_SKIP_EIG_SINGLE_PRECISION`, `TCICT_SKIP_EIGH_SINGLE_PRECISION`
- `TCICT_SKIP_EIGVALS_SINGLE_PRECISION`, `TCICT_SKIP_EIGVALSH_SINGLE_PRECISION`
- `TCICT_SKIP_SVD_SINGLE_PRECISION`, `TCICT_SKIP_TRUNC_SVD_SINGLE_PRECISION`
- `TCICT_SKIP_QR_SINGLE_PRECISION`, `TCICT_SKIP_LQ_SINGLE_PRECISION`
- `TCICT_SKIP_EXP_SINGLE_PRECISION`, `TCICT_SKIP_INVERSE_SINGLE_PRECISION`

**Different semantics from `TCICT_SKIP_<API>`.** Whole-API skip in section 4 excludes the test body **at preprocessing time** (`#ifndef`), so the API call disappears from the translation unit entirely. Precision skip is an **`if constexpr`-guarded runtime early return**: the body is still part of the function template and gets instantiated for every TenT, including the single-precision ones. API calls inside the body must therefore be valid for that precision at compile time.

This handles backends with runtime-buggy APIs (the call compiles but produces wrong values). For backends that intentionally make a (API × precision) combination compile-time-unavailable (via SFINAE / `static_assert` / `= delete`), use the whole-API `TCICT_SKIP_<API>` macro instead. A compile-time body-discard variant is planned (see [#50](https://github.com/ultimatile/tensor-computing-interface-conformance-test/issues/50)).

## Test Categories

| Category | Header | Functions exercised |
|---|---|---|
| Construction | `tests/construction.h` | `allocate`, `assign_from_range`, `clear`, `copy`, `eye`, `fill`, `move`, `random`, `zeros` |
| Read-only getters | `tests/read_only_getters.h` | `get_elem`, `order`, `set_elem`, `shape`, `size`, `size_bytes` |
| Tensor manipulation | `tests/tensor_manipulation.h` | `concatenate`, `cplx_conj`, `diag`, `expand`, `extract_sub`, `for_each`, `for_each_with_coors`, `imag`, `real`, `replace_sub`, `reshape`, `shrink`, `stack`, `to_cplx`, `transpose` |
| Linear algebra | `tests/linear_algebra.h` | `contract`, `eig`, `eigh`, `eigvals`, `eigvalsh`, `exp`, `inverse`, `linear_combine`, `lq`, `norm`, `normalize`, `qr`, `scale`, `svd`, `trace`, `trunc_svd` |
| I/O | `tests/io_operations.h` | `load`, `save` |
| Miscellaneous | `tests/miscellaneous.h` | `close`, `convert`, `show`, `to_range`, `version` |

## Requirements

- C++17
- TCI headers (`tci/tensor_traits.h` and backend-specific trait specialization)

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for the local integration check against the cytnx backend.
