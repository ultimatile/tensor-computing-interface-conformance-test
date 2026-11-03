# Contributing

## Commit messages

Commit subjects follow [Conventional Commits](https://www.conventionalcommits.org/).
This repository's deliverable is the test suite, so the type is judged against the suite, not against the backends it tests:

- `feat`: new coverage, such as a new test case or a newly covered API, or new suite infrastructure, such as adapters, tolerance categories or skip macros.
- `fix`: a test that asserts the wrong thing, including one that asserts behavior the spec does not define, or any other bug in the suite.
- `refactor`: a change to how tests are written that leaves what they assert unchanged.

`test` is not used: every change here touches tests, so the type would tell a reader nothing.

## Local integration check with the cytnx backend

`make integration-check` verifies the current tcict HEAD against the [cytnx
backend](https://github.com/r-ccs-cms/tensor-computing-interface-backend-cytnx)
without the manual "copy headers / build / `git checkout`" routine.

### Default behavior

```sh
make integration-check
```

This:

1. Clones cytnx (with submodules) into `./build/tensor-computing-interface-backend-cytnx/` if it does not yet exist, then runs `git fetch`, `git reset --hard $CYTNX_REF`, and `git submodule update --init --recursive --force` on every invocation so the checkout always tracks `$CYTNX_REF` (default under "Environment variables" below).
2. Replaces `external/tcict/` content with this repo's tracked HEAD via `git archive HEAD | tar -xmf -`; `-m` stamps the extracted files with the current time, so the build recompiles against them.
3. Builds the cytnx test target (`TCITests`).

`build/` is gitignored, so the cloned backend and its cmake artefacts stay untracked.

### Use an existing cytnx checkout

```sh
make integration-check CYTNX_BACKEND_DIR=$HOME/path/to/tensor-computing-interface-backend-cytnx
```

When `CYTNX_BACKEND_DIR` is set in the environment, the script does **not** clone or modify the outer cytnx working tree — it only overrides `external/tcict/`. To return to the auto-clone default, unset the variable rather than passing the default value explicitly.

### Run the tests

```sh
make integration-test
```

Builds, then runs `$CYTNX_BACKEND_DIR/build/test/TCITests` directly.

### Restore the cytnx submodule

```sh
make integration-restore
```

Resets `$CYTNX_BACKEND_DIR/external/tcict` to cytnx's pinned submodule revision (`git submodule update --init --force external/tcict`) and runs `git clean -ffd` inside that path. The clean step is required: the override adds files that do not exist at the submodule pin (Makefile, CONTRIBUTING.md, scripts/, .gitignore), and a plain `submodule update --force` would leave those files as untracked.

### Extra cmake configure args

Some platforms need more than `-DTCI_BUILD_TESTS=ON` to configure the cytnx build. Pass the additional flags via `INTEGRATION_CMAKE_ARGS`:

```sh
# macOS + Homebrew toolchain (matches cytnx's CMakePresets.json `brew-debug`)
INTEGRATION_CMAKE_ARGS='--preset brew-debug' make integration-check

# Or set the toolchain file directly
INTEGRATION_CMAKE_ARGS='-DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/macos-homebrew.cmake -DUSE_HPTT=ON' make integration-check
```

`INTEGRATION_CMAKE_ARGS` is appended verbatim to the `cmake -S ... -B ...` line.

### Caveats

- The override copies tcict **HEAD** (last committed state), so `make integration-check` and `make integration-test` refuse to run while tracked files have uncommitted changes; commit or stash them first. Untracked files are not shipped either.
- The script never writes to `external/tcict/.git*`, so `make integration-restore` always recovers the pinned state.
- The `TCICT_SKIP_*` skip-macro matrix (cross-build with various skip-macro combinations) is deferred to a follow-up issue.
- `make integration-restore` requires `CYTNX_BACKEND_DIR` to point at an actual git checkout (with `external/tcict` initialized as a submodule).

### Environment variables

| Variable | Default | Effect |
|---|---|---|
| `CYTNX_BACKEND_DIR` | `./build/tensor-computing-interface-backend-cytnx` | cytnx checkout to override into; **set** disables auto-clone/sync |
| `CYTNX_REPO_URL` | `https://github.com/r-ccs-cms/tensor-computing-interface-backend-cytnx.git` | URL used only when auto-cloning the default path |
| `CYTNX_REF` | `origin/dev/tcapi-v1` | ref the default-path checkout is reset to on every run; the default is the cytnx branch that carries the TCAPI V1 signatures this suite calls, which `origin/main` does not |
| `INTEGRATION_CMAKE_ARGS` | (empty) | extra args appended to the cmake configure command |

## Conventions for writing tests

These apply to every header under `include/tcict/`; comments in the code refer to them by section name.

### Out-of-place inputs

TCAPI V1: "Most mutating functions provide two overloads: (1) in-place and (2) out-of-place writing to a separate output."
The tests that read their input back after an out-of-place call assert a reading of that sentence: that the input is unchanged.

### Inputs to multi-input operations

`concatenate`, `stack` and `linear_combine` take their inputs as `List<CRef<TenT>>`, and TCAPI V1 says of them: "The referenced tensors are not copied, consumed, or modified and must stay alive for the call".
Each test reads every input back after its last call to the operation and asserts that it is unchanged; unlike "Out-of-place inputs", this asserts V1's own words, not a reading of them.

Each of the three operations, and each `linear_combine` overload, is called at least once through a named `tci::List<tci::CRef<TenT>>` variable, not only through a braced `{std::cref(a), ...}` list.
When `TenT` is copy-constructible, a braced list of `std::cref`s also initializes a `List<TenT>`, so a braced call alone compiles against a backend that still takes `List<TenT>`.
A named `List<CRef<TenT>>` does not convert to `List<TenT>`, so that call compiles only against V1's parameter type.
A call with a braced list names `TenT` explicitly, as in `tci::stack<TenT>(ctx, {std::cref(a), std::cref(b)}, 0)`, because no parameter of these functions deduces it from a braced list.

### Callbacks the backend invokes

Coordinate maps, generators and traversal callbacks run inside the backend's frames, which a conformance suite cannot assume are able to carry an exception.
The suite reads V1 as invoking them one at a time: V1's own `for_each` example accumulates into a plain variable, `[&tot](const float el) { tot += el; }`, and gives the result as `tot == 3.0`.
So they neither assert, nor allocate, nor read a coordinate's components before checking the coordinate: that it has as many components as the tensor has bonds, and that each is less than its bond's dimension.
Whatever a callback observes for the test, such as a flag for a malformed coordinate, a visit count or a running sum, it keeps in variables the test reads after the call returns, and the test asserts on those.
A callback that receives a second-order coordinate makes that check with `coors_2d` in `include/tcict/elem_helper.h` before reading its components, and records a coordinate `coors_2d` rejects in a flag the test asserts is clear.
Tests call `to_range` and `assign_from_range` on second-order tensors through `to_range_2d` and `assign_from_range_2d` in `include/tcict/range_2d.h`.
