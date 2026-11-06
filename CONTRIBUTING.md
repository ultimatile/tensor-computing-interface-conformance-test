# Contributing

## Commit messages

Commit subjects follow [Conventional Commits](https://www.conventionalcommits.org/).
This repository's deliverable is the test suite, so the type is judged against the suite, not against the backends it tests:

- `feat`: new coverage, such as a new test case or a newly covered API, or new suite infrastructure, such as adapters, tolerance categories or skip macros.
- `fix`: a test that asserts the wrong thing, including one that asserts behavior the spec does not define, or any other bug in the suite.
- `refactor`: a change to how tests are written that leaves what they assert unchanged.

`test` is not used: every change here touches tests, so the type would tell a reader nothing.

## Local integration check with the cytnx backend

`make integration-check` verifies the current tcapict HEAD against the [cytnx
backend](https://github.com/r-ccs-cms/tcapi-backend-cytnx)
without the manual "copy headers / build / `git checkout`" routine.

### Default behavior

```sh
make integration-check
```

This:

0. Refuses to run while tracked files of this repo have uncommitted changes (see "Caveats").
1. Clones cytnx (with submodules) into `./build/tcapi-backend-cytnx/` if it does not yet exist, then runs `git fetch`, `git reset --hard $CYTNX_REF`, and `git submodule update --init --recursive --force` on every invocation so the checkout always tracks `$CYTNX_REF` (default under "Environment variables" below).
2. Refuses to continue unless the cytnx working tree's `.gitmodules` registers a submodule at the path `external/tcapict` and that submodule is initialized, since a cytnx revision that vendors this suite under another path cannot take the override.
3. Replaces `external/tcapict/` content with this repo's tracked HEAD via `git archive HEAD | tar -xmf -`; `-m` stamps the extracted files with the current time, so the build recompiles against them.
4. Builds the cytnx test target (`TCAPITests`).

`build/` is gitignored, so the cloned backend and its cmake artefacts stay untracked.

### Use an existing cytnx checkout

```sh
make integration-check CYTNX_BACKEND_DIR=$HOME/path/to/tcapi-backend-cytnx
```

When `CYTNX_BACKEND_DIR` is set in the environment, the script does **not** clone or modify the outer cytnx working tree — it only overrides `external/tcapict/`. That checkout must already register and initialize `external/tcapict` as a submodule (step 2 above) and provide the signatures the suite calls; see `CYTNX_REF` under "Environment variables". To return to the auto-clone default, unset the variable rather than passing the default value explicitly.

### Run the tests

```sh
make integration-test
```

Builds, then runs `$CYTNX_BACKEND_DIR/build/test/TCAPITests` directly.

### Restore the cytnx submodule

```sh
make integration-restore
```

Resets `$CYTNX_BACKEND_DIR/external/tcapict` to cytnx's pinned submodule revision (`git submodule update --init --force external/tcapict`) and runs `git clean -ffd` inside that path. The clean step is required: the override adds files that do not exist at the submodule pin (Makefile, CONTRIBUTING.md, scripts/, .gitignore), and a plain `submodule update --force` would leave those files as untracked.

### Extra cmake configure args

Some platforms need more than `-DTCAPI_BUILD_TESTS=ON` to configure the cytnx build. Pass the additional flags via `INTEGRATION_CMAKE_ARGS`:

```sh
# macOS + Homebrew toolchain (matches cytnx's CMakePresets.json `brew-debug`)
INTEGRATION_CMAKE_ARGS='--preset brew-debug' make integration-check

# Or set the toolchain file directly
INTEGRATION_CMAKE_ARGS='-DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/macos-homebrew.cmake -DUSE_HPTT=ON' make integration-check
```

`INTEGRATION_CMAKE_ARGS` is appended verbatim to the `cmake -S ... -B ...` line.

### Caveats

- The override copies tcapict **HEAD** (last committed state), so `make integration-check` and `make integration-test` refuse to run while tracked files have uncommitted changes; commit or stash them first. Untracked files are not shipped either.
- The script never writes to `external/tcapict/.git*`, so `make integration-restore` always recovers the pinned state.
- The `TCAPICT_SKIP_*` skip-macro matrix (cross-build with various skip-macro combinations) is deferred to a follow-up issue.
- `make integration-restore` requires `CYTNX_BACKEND_DIR` to point at an actual git checkout whose working-tree `.gitmodules` registers a submodule at the path `external/tcapict`.

### Environment variables

| Variable | Default | Effect |
|---|---|---|
| `CYTNX_BACKEND_DIR` | `./build/tcapi-backend-cytnx` | cytnx checkout to override into; **set** disables auto-clone/sync |
| `CYTNX_REPO_URL` | `https://github.com/r-ccs-cms/tcapi-backend-cytnx.git` | URL used only when auto-cloning the default path |
| `CYTNX_REF` | `origin/main` | ref the default-path checkout is reset to on every run; it must provide the TCAPI V1 signatures this suite calls, such as `concatenate`, `stack` and `linear_combine` over `List<CRef<TenT>>` |
| `INTEGRATION_CMAKE_ARGS` | (empty) | extra args appended to the cmake configure command |

## Conventions for writing tests

These apply to every header under `include/tcapict/`; comments in the code refer to them by section name.

### Out-of-place inputs

TCAPI V1: "In-place variants mutate the supplied tensor operand(s), whereas out-of-place variants leave all inputs unmodified".
The tests that read their input back after an out-of-place call assert that "leave all inputs unmodified" for the elements they read.

### Inputs to multi-input operations

`concatenate`, `stack` and `linear_combine` take their inputs as `List<CRef<TenT>>`, and TCAPI V1's entry for each of them says: "In the C++ interface, `ins` is a list of non-owning const references to existing tensors; the referenced tensor objects are not copied, consumed, or modified and must remain valid for the duration of the call."
Each test reads every input back after its last call to the operation and asserts that its shape and elements are unchanged.
That asserts the "not ... modified"; the read-back cannot detect a copy, so "not copied" is not asserted.
An input whose elements all hold one value would pass that read-back even after being reordered in place, so each input element gets a value no other input element has.
Each input other than a zeroth-order one is built from a coordinate-to-value function, through `TCAPICT_SET_ELEMS` in `include/tcapict/elem_sweep.h`, and both the read-back and the check of every element of the result are computed from the same functions, through `TCAPICT_EXPECT_ELEMS`; `ramp_over` there gives distinct values to distinct coordinates.
A zeroth-order input has one element, which no reordering can move, so the test of `linear_combine` on zeroth-order inputs sets each input's element to a value the other input does not hold through `TCAPICT_SET_ZEROTH_ORDER`, and reads it back and checks the result through `TCAPICT_EXPECT_ZEROTH_ORDER`.

Each of the three operations, and each `linear_combine` overload, is called at least once through a named `tcapi::List<tcapi::CRef<TenT>>` variable, not only through a braced `{std::cref(a), ...}` list.
When `TenT` is copy-constructible, a braced list of `std::cref`s also initializes a `List<TenT>`, so a braced call alone compiles against a backend that still takes `List<TenT>`.
A named `List<CRef<TenT>>` does not convert to `List<TenT>`, so that call compiles only against V1's parameter type.
A call with a braced list names `TenT` explicitly, as in `tcapi::stack<TenT>(ctx, {std::cref(a), std::cref(b)}, 0)`, because no parameter of these functions deduces it from a braced list.

### Callbacks the backend invokes

Coordinate maps, generators and traversal callbacks run inside the backend's frames, which a conformance suite cannot assume are able to carry an exception.
The suite reads V1 as invoking them one at a time: V1's own `for_each` example accumulates into a plain variable, `[&tot](const float el) { tot += el; }`, and gives the result as `tot == 3.0`.
So they do not assert, do not allocate, and read a coordinate's components only after checking the coordinate: that it has as many components as the tensor has bonds, and that each is less than its bond's dimension.
Whatever a callback observes for the test, such as a flag for a malformed coordinate, a visit count or a running sum, it keeps in variables the test reads after the call returns, and the test asserts on those.
A callback that receives a second-order coordinate makes that check with `coors_2d` in `include/tcapict/elem_helper.h` before reading its components, and records a coordinate `coors_2d` rejects in a flag the test asserts is clear.
Tests call `to_range` and `assign_from_range` on second-order tensors through `to_range_2d` and `assign_from_range_2d` in `include/tcapict/range_2d.h`.

### Zeroth-order tensors

TCAPI V1, in the introduction to its core functions: "A zeroth-order tensor has empty shape, size 1, and a single empty coordinate sequence."
Where a test knows the element a zeroth-order `TenT` should hold, it checks the tensor through `TCAPICT_EXPECT_ZEROTH_ORDER` in `include/tcapict/zeroth_order.h`: order 0, shape `{}` and size 1, then the sole element read at `{}` through `expect_elems_at` in `include/tcapict/elem_sweep.h`, unless it splits that check as `test_random_zeroth_order` below does.
The results of `real` and `imag` of a complex tensor, and of `to_cplx` of a real one, have another tensor type; those results are checked through `TCAPICT_EXPECT_ZEROTH_ORDER_SHAPE`, which takes any tensor type, and `get_elem`.
`test_random_zeroth_order` checks its result through `TCAPICT_EXPECT_ZEROTH_ORDER_SHAPE` before it asserts the generator's call count, and then the sole element through `TCAPICT_EXPECT_ELEMS`, so that a backend ignoring the empty shape is reported as such.
A callback that receives a coordinate of a zeroth-order tensor expects the empty one and flags any other, under "Callbacks the backend invokes", through `to_range_0d`, `assign_from_range_0d` or `visit_record_0d` in the same header.

V1 goes on: "Operations whose definitions require selecting, reordering, inserting, removing, slicing, or partitioning existing bonds are not applicable to zeroth-order tensors unless explicitly specified by the operation."
It names no such operation. The suite reads `transpose`, `svd`, `qr`, `eig`, `expand`, `shrink`, `extract_sub` and `replace_sub` as such operations, and none of their entries specifies a zeroth-order case.
V1 defines no exception for them, so nothing about calling them on a zeroth-order tensor is assertable, and no test calls them on one.
No test calls the following on a zeroth-order tensor either, since the suite reads the same sentence as covering them, and their entries specify no zeroth-order case: `concatenate`, which selects a bond, `stack`, which inserts one, and the operations that matricize, which partition bonds into rows and columns (`trunc_svd`, `lq`, `eigh`, `eigvals`, `eigvalsh`, `exp`, `inverse`).
Nor does any call `diag`, whose input V1 requires to be first- or second-order.
The suite reads the operations its zeroth-order tests do call as requiring no existing bond of a zeroth-order operand; `contract` is among them, since such an operand takes the empty label list `""`.

The zeroth-order tests assert nothing about `TenT{}`, the state `clear` and `move` leave, since V1 does not say what queries return on it; the `order == 0` assertions the existing `clear` and `move` tests make on it are left as they are, pending the open question on [#51](https://github.com/ultimatile/tcapi-conformance-test/issues/51).
Of `size_bytes` they assert only that it returns, as the tests of the other ranks do.

### Diagonal tensors

TCAPI V1, in the introduction to its core functions: "TCAPI treats a diagonal tensor as an ordinary second-order tensor with shape {N,N} whose off-diagonal logical tensor elements are zero." and "Unless stated otherwise, TCAPI operations are defined by logical tensor elements, not by storage representation."
Its entries name these diagonal tensors: `eye` returns "the N×N identity tensor as a diagonal tensor", `diag` of a first-order input and the `sigma` of `trunc_svd` and `lambda_mat` of `eig` are each "a second-order diagonal tensor", and the `sigma` of `svd` and `lambda_mat` of `eigh` are each "a real second-order diagonal tensor".
`test_diag_mat_to_vec_from_diagonal` builds diagonal tensors with `diag`, and with `eye` when the backend has it, only as inputs to extract from, and checks the first through `TCAPICT_EXPECT_DIAGONAL`.
`test_eye` and the other tests of `diag`, `svd`, `trunc_svd`, `eig` and `eigh` check each diagonal tensor they produce as a result through `TCAPICT_EXPECT_DIAGONAL` in `include/tcapict/diagonal.h`: order 2, shape `{n, n}` and size `n * n`, checked before any element is read, then every element against the expected diagonal.
Off-diagonal elements must be exactly zero, since V1 defines a diagonal tensor as one "whose off-diagonal logical tensor elements are zero".
The check takes any tensor type, with the context its input came with, because `sigma` and `eigh`'s `lambda_mat` are `real_ten_t<TenT>` and `eig`'s `lambda_mat` is `cplx_ten_t<TenT>`.
V1 fixes the order of the diagonal for `svd` ("non-increasing order") and for `eigh` ("ascending order").
It states none for `eig`, nor for the `sigma` of `trunc_svd`, whose entry says that the routine "follows the same matricization and SVD convention as svd" but states no order for its own `sigma`; so the `trunc_svd` tests, and the `eig` tests wherever the eigenvalues are not all equal, compare the diagonal as a multiset, through `diag_order::any`.
`eigvals` and `eigvalsh` return first-order lists, and their tests assert order 1.
