#!/usr/bin/env bash
# Local integration verification: build the cytnx backend against this repo's
# tracked HEAD without touching the cytnx submodule's pinned tcict revision
# permanently. See CONTRIBUTING.md for the developer-facing description.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TCICT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default cytnx repo coordinates. Both are overridable via env so this script
# can target forks or alternate refs without code edits.
CYTNX_REPO_URL="${CYTNX_REPO_URL:-https://github.com/r-ccs-cms/tensor-computing-interface-backend-cytnx.git}"
CYTNX_REF="${CYTNX_REF:-origin/main}"

# Extra cmake configure args (e.g. "--preset brew-debug" or
# "-DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/macos-homebrew.cmake -DUSE_HPTT=ON").
# Empty by default — set when the cytnx repo's documented setup needs more
# than -DTCI_BUILD_TESTS=ON.
INTEGRATION_CMAKE_ARGS="${INTEGRATION_CMAKE_ARGS:-}"

# Decide default-vs-user-supplied by env-var presence, not by string compare on
# the value. A value-compare would route paths like
# "./build/.../" or an absolute spelling of the same default through the
# user-supplied branch and silently skip clone/sync.
if [[ -z "${CYTNX_BACKEND_DIR-}" ]]; then
  CYTNX_BACKEND_DIR="$TCICT_ROOT/build/tensor-computing-interface-backend-cytnx"
  USER_SUPPLIED=0
else
  USER_SUPPLIED=1
fi

DEST="$CYTNX_BACKEND_DIR/external/tcict"
SENTINEL="$DEST/include/tcict/skip.h"

usage() {
  cat <<EOF
Usage: $0 <subcommand>

Subcommands:
  check     sync (default-path only) -> override external/tcict -> cmake build
  test      same as check, then run \$CYTNX_BACKEND_DIR/build/test/TCITests
  restore   restore external/tcict to the cytnx submodule pin

Environment:
  CYTNX_BACKEND_DIR        path to the cytnx checkout (default: $TCICT_ROOT/build/tensor-computing-interface-backend-cytnx)
  CYTNX_REPO_URL           cytnx repo URL used only when auto-cloning the default path (default: $CYTNX_REPO_URL)
  CYTNX_REF                ref to reset the default-path checkout to (default: $CYTNX_REF)
  INTEGRATION_CMAKE_ARGS   extra args appended to the cmake configure command
EOF
}

log() { printf '[integration-check] %s\n' "$*"; }
fail() { printf '[integration-check] error: %s\n' "$*" >&2; exit 1; }

sync_default_path() {
  if [[ ! -d "$CYTNX_BACKEND_DIR" ]]; then
    log "cloning cytnx into $CYTNX_BACKEND_DIR"
    mkdir -p "$(dirname "$CYTNX_BACKEND_DIR")"
    git clone --recurse-submodules "$CYTNX_REPO_URL" "$CYTNX_BACKEND_DIR"
  fi
  # An existing path may have been left behind by a partial clone or unrelated
  # work; reject it explicitly rather than letting `git fetch` print a raw
  # not-a-repo error.
  git -C "$CYTNX_BACKEND_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || fail "$CYTNX_BACKEND_DIR exists but is not a git checkout (remove it or set CYTNX_BACKEND_DIR to a real cytnx clone)"
  # Always fetch + reset + submodule update, including immediately after the
  # initial clone. `git clone` lands on the upstream default branch, so a
  # CYTNX_REF override (a tag, a feature branch) only takes effect once we
  # explicitly reset to it.
  log "syncing cytnx checkout at $CYTNX_BACKEND_DIR to $CYTNX_REF"
  git -C "$CYTNX_BACKEND_DIR" fetch
  git -C "$CYTNX_BACKEND_DIR" reset --hard "$CYTNX_REF"
  # --force is required: a previous run leaves external/tcict with the override's
  # working-tree content, and a non-forced submodule update refuses to checkout
  # a different pin over those files.
  git -C "$CYTNX_BACKEND_DIR" submodule update --init --recursive --force
}

require_sentinel() {
  [[ -f "$SENTINEL" ]] || fail "sentinel missing: $SENTINEL (CYTNX_BACKEND_DIR may be wrong)"
}

override_tcict() {
  log "overriding $DEST with tcict HEAD"
  # Wipe DEST top-level entries except the .git pointer so files removed in
  # tcict HEAD do not survive across runs. `git archive | tar` overwrites but
  # does not delete; the wipe is what makes the override idempotent.
  find "$DEST" -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf -- {} +
  # Ship only tracked files from HEAD; working-tree artefacts (build outputs,
  # editor backups) do not leak into the override. `tar -xf -` makes the stdin
  # archive source explicit for portability across tar implementations.
  git -C "$TCICT_ROOT" archive HEAD | tar -xf - -C "$DEST"
}

cmake_build() {
  log "configuring cytnx test build"
  # shellcheck disable=SC2086 # INTEGRATION_CMAKE_ARGS is intentionally word-split
  cmake -S "$CYTNX_BACKEND_DIR" -B "$CYTNX_BACKEND_DIR/build" -DTCI_BUILD_TESTS=ON $INTEGRATION_CMAKE_ARGS
  log "building TCITests"
  cmake --build "$CYTNX_BACKEND_DIR/build" --target TCITests
}

run_tests() {
  local bin="$CYTNX_BACKEND_DIR/build/test/TCITests"
  [[ -x "$bin" ]] || fail "test binary not found at $bin (build did not produce it)"
  log "running $bin"
  "$bin"
}

restore_tcict() {
  [[ -d "$CYTNX_BACKEND_DIR/.git" || -f "$CYTNX_BACKEND_DIR/.git" ]] \
    || fail "$CYTNX_BACKEND_DIR is not a git checkout"
  # Restore must work from a half-broken state too: a previous override may
  # have wiped DEST and failed before re-extracting, leaving no sentinel.
  # The cytnx-is-a-repo check above is enough; the submodule-update step
  # below repairs the working tree regardless.
  git -C "$CYTNX_BACKEND_DIR" config -f .gitmodules submodule.external/tcict.path >/dev/null 2>&1 \
    || fail "external/tcict is not a registered submodule of $CYTNX_BACKEND_DIR"
  log "restoring $DEST to cytnx submodule pin"
  git -C "$CYTNX_BACKEND_DIR" submodule update --init --force external/tcict
  # The submodule pin does not include files added by the override (Makefile,
  # CONTRIBUTING.md, scripts/, .gitignore). `submodule update --force` checks
  # out the pinned tree but leaves those as untracked; `clean -ffd` makes
  # restore a complete undo.
  git -C "$DEST" clean -ffd
}

main() {
  local cmd="${1:-}"
  case "$cmd" in
    check)
      [[ "$USER_SUPPLIED" == 1 ]] || sync_default_path
      [[ -d "$CYTNX_BACKEND_DIR" ]] || fail "$CYTNX_BACKEND_DIR does not exist"
      require_sentinel
      override_tcict
      cmake_build
      ;;
    test)
      [[ "$USER_SUPPLIED" == 1 ]] || sync_default_path
      [[ -d "$CYTNX_BACKEND_DIR" ]] || fail "$CYTNX_BACKEND_DIR does not exist"
      require_sentinel
      override_tcict
      cmake_build
      run_tests
      ;;
    restore)
      [[ -d "$CYTNX_BACKEND_DIR" ]] || fail "$CYTNX_BACKEND_DIR does not exist"
      restore_tcict
      ;;
    -h|--help|help|"")
      usage
      [[ -z "$cmd" ]] && exit 1 || exit 0
      ;;
    *)
      printf 'unknown subcommand: %s\n\n' "$cmd" >&2
      usage >&2
      exit 1
      ;;
  esac
}

main "$@"
