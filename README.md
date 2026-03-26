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

### 3. Write a thin registration layer

Each backend provides a small bridge file that maps TCICT test functions to the backend's test framework.

```cpp
#include <tcict/tcict.h>
#include <my_backend/tci.h>

// Example with doctest
#include <doctest/doctest.h>

#define TCICT_DOCTEST_CASE(category, test_func, TenT)          \
  TEST_CASE("TCICT: " category " - " #test_func) {            \
    tcict::tci_test_fixture<TenT> fix;                         \
    tcict::tests::test_func<TenT>(fix);                        \
  }

using MyTensor = my_backend::Tensor<std::complex<double>>;

TCICT_DOCTEST_CASE("construction", test_zeros, MyTensor)
TCICT_DOCTEST_CASE("construction", test_eye, MyTensor)
// ... register more tests
```

### 4. Skip unimplemented functions

Define `TCICT_SKIP_*` macros to opt out of tests for functions not yet implemented:

```cmake
target_compile_definitions(my_tests PRIVATE
    TCICT_SKIP_QR
    TCICT_SKIP_LQ
)
```

See `include/tcict/skip.h` for the full list of available skip macros.

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

## Requirements

- C++17
- TCI headers (`tci/tensor_traits.h` and backend-specific trait specialization)
