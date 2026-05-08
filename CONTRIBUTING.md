# Contributing

## Local integration check with the cytnx backend

`make integration-check` verifies the current tcict HEAD against the [cytnx
backend](https://github.com/r-ccs-cms/tensor-computing-interface-backend-cytnx)
without the manual "copy headers / build / `git checkout`" routine.

### Default behavior

```sh
make integration-check
```

This:

1. Clones cytnx (with submodules) into `./build/tensor-computing-interface-backend-cytnx/` if it does not yet exist, then runs `git fetch`, `git reset --hard origin/main`, and `git submodule update --init --recursive --force` on every invocation so the checkout always tracks `$CYTNX_REF`.
2. Replaces `external/tcict/` content with this repo's tracked HEAD via `git archive HEAD | tar -x`.
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

- The override copies tcict **HEAD** (last committed state). Uncommitted changes are not included; commit first.
- The script never writes to `external/tcict/.git*`, so `make integration-restore` always recovers the pinned state.
- The `TCICT_SKIP_*` skip-macro matrix (cross-build with various skip-macro combinations) is deferred to a follow-up issue.
- `make integration-restore` requires `CYTNX_BACKEND_DIR` to point at an actual git checkout (with `external/tcict` initialized as a submodule).

### Environment variables

| Variable | Default | Effect |
|---|---|---|
| `CYTNX_BACKEND_DIR` | `./build/tensor-computing-interface-backend-cytnx` | cytnx checkout to override into; **set** disables auto-clone/sync |
| `CYTNX_REPO_URL` | `https://github.com/r-ccs-cms/tensor-computing-interface-backend-cytnx.git` | URL used only when auto-cloning the default path |
| `CYTNX_REF` | `origin/main` | ref the default-path checkout is reset to on every run |
| `INTEGRATION_CMAKE_ARGS` | (empty) | extra args appended to the cmake configure command |
