#!/usr/bin/env bash
# Local integration verification: build the cytnx backend against this repo's
# tracked HEAD without touching the cytnx submodule's pinned tcapict revision
# permanently. See CONTRIBUTING.md for the developer-facing description.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TCAPICT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default cytnx repo coordinates. Both are overridable via env so this script
# can target forks or alternate refs without code edits. The ref must provide
# the TCAPI V1 signatures this suite calls, such as concatenate, stack and
# linear_combine over List<CRef<TenT>>.
CYTNX_REPO_URL="${CYTNX_REPO_URL:-https://github.com/r-ccs-cms/tcapi-backend-cytnx.git}"
CYTNX_REF="${CYTNX_REF:-origin/main}"

# Extra cmake configure args (e.g. "--preset brew-debug" or
# "-DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/macos-homebrew.cmake -DUSE_HPTT=ON").
# Empty by default — set when the cytnx repo's documented setup needs more
# than -DTCAPI_BUILD_TESTS=ON.
INTEGRATION_CMAKE_ARGS="${INTEGRATION_CMAKE_ARGS:-}"

# Decide default-vs-user-supplied by env-var presence, not by string compare on
# the value. A value-compare would route paths like
# "./build/.../" or an absolute spelling of the same default through the
# user-supplied branch and silently skip clone/sync.
if [[ -z "${CYTNX_BACKEND_DIR-}" ]]; then
  CYTNX_BACKEND_DIR="$TCAPICT_ROOT/build/tcapi-backend-cytnx"
  USER_SUPPLIED=0
else
  USER_SUPPLIED=1
fi

DEST="$CYTNX_BACKEND_DIR/external/tcapict"

usage() {
  cat <<EOF
Usage: $0 <subcommand>

Subcommands:
  check     refuse if tracked tcapict files have uncommitted changes -> sync
            (default-path only) -> refuse unless .gitmodules registers an
            initialized submodule at external/tcapict -> override
            external/tcapict -> cmake build
  test      same as check, then run \$CYTNX_BACKEND_DIR/build/test/TCAPITests
  restore   restore external/tcapict to the cytnx submodule pin

Environment:
  CYTNX_BACKEND_DIR        path to the cytnx checkout (default: $TCAPICT_ROOT/build/tcapi-backend-cytnx)
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
  # --force is required: a previous run leaves external/tcapict with the override's
  # working-tree content, and a non-forced submodule update refuses to checkout
  # a different pin over those files.
  git -C "$CYTNX_BACKEND_DIR" submodule update --init --recursive --force
}

require_registered_submodule() {
  [[ -d "$CYTNX_BACKEND_DIR/.git" || -f "$CYTNX_BACKEND_DIR/.git" ]] \
    || fail "$CYTNX_BACKEND_DIR is not a git checkout"
  [[ -f "$CYTNX_BACKEND_DIR/.gitmodules" ]] \
    || fail "$CYTNX_BACKEND_DIR has no .gitmodules (CYTNX_BACKEND_DIR may not point at a cytnx checkout)"
  # Read from the working tree's .gitmodules, which a sync or a checkout sets
  # to the checked-out revision's. A file under DEST cannot answer this: the
  # pinned suite revision may have another layout, and a reset to an older
  # cytnx revision can leave the newer layout behind as untracked files.
  git -C "$CYTNX_BACKEND_DIR" config -f .gitmodules --get-regexp '^submodule\..*\.path$' '^external/tcapict$' >/dev/null 2>&1 \
    || fail "$CYTNX_BACKEND_DIR's .gitmodules registers no submodule at external/tcapict: the checked-out cytnx revision does not vendor this suite there"
}

require_vendored_suite() {
  require_registered_submodule
  # override_tcapict deletes everything in DEST but its .git, so DEST must be
  # the submodule's own checkout, which restore can repopulate. Its .git is
  # usually a file pointing into the superproject, sometimes a directory.
  [[ -e "$DEST/.git" ]] \
    || fail "$DEST is not an initialized submodule checkout; run 'git submodule update --init external/tcapict' in $CYTNX_BACKEND_DIR"
}

override_tcapict() {
  log "overriding $DEST with tcapict HEAD"
  # Wipe DEST top-level entries except the .git pointer so files removed in
  # tcapict HEAD do not survive across runs. `git archive | tar` overwrites but
  # does not delete; the wipe is what makes the override idempotent.
  find "$DEST" -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf -- {} +
  # Ship only tracked files from HEAD; working-tree artefacts (build outputs,
  # editor backups) do not leak into the override. `-f -` makes the stdin
  # archive source explicit for portability across tar implementations. `-m`
  # stamps the extracted files with the current time instead of the commit's:
  # an older stamp than the existing build objects would let the build skip
  # recompiling against the new headers.
  git -C "$TCAPICT_ROOT" archive HEAD | tar -xmf - -C "$DEST"
}

require_clean_tcapict() {
  # The override ships HEAD only, so an uncommitted edit to a tracked file
  # would silently go untested.
  local rc=0
  git -C "$TCAPICT_ROOT" diff --quiet HEAD -- || rc=$?
  case "$rc" in
    0) ;;
    1) fail "tracked files in $TCAPICT_ROOT have uncommitted changes, which the override (tcapict HEAD) would not include; commit or stash them first" ;;
    *) fail "git diff failed (exit $rc) while checking $TCAPICT_ROOT for uncommitted changes" ;;
  esac
}

cmake_build() {
  log "configuring cytnx test build"
  # shellcheck disable=SC2086 # INTEGRATION_CMAKE_ARGS is intentionally word-split
  cmake -S "$CYTNX_BACKEND_DIR" -B "$CYTNX_BACKEND_DIR/build" -DTCAPI_BUILD_TESTS=ON $INTEGRATION_CMAKE_ARGS
  log "building TCAPITests"
  cmake --build "$CYTNX_BACKEND_DIR/build" --target TCAPITests
}

run_tests() {
  local bin="$CYTNX_BACKEND_DIR/build/test/TCAPITests"
  [[ -x "$bin" ]] || fail "test binary not found at $bin (build did not produce it)"
  log "running $bin"
  "$bin"
}

restore_tcapict() {
  # Restore must work from a half-broken state too: a previous override may
  # have deleted DEST's contents and failed before re-extracting. The
  # registration check is enough; the submodule-update step below repairs the
  # working tree regardless.
  require_registered_submodule
  log "restoring $DEST to cytnx submodule pin"
  git -C "$CYTNX_BACKEND_DIR" submodule update --init --force external/tcapict
  # The submodule pin does not include files added by the override (Makefile,
  # CONTRIBUTING.md, scripts/, .gitignore). `submodule update --force` checks
  # out the pinned tree but leaves those as untracked; `clean -ffd` makes
  # restore a complete undo.
  git -C "$DEST" clean -ffd
}

prepare_and_build() {
  require_clean_tcapict
  [[ "$USER_SUPPLIED" == 1 ]] || sync_default_path
  [[ -d "$CYTNX_BACKEND_DIR" ]] || fail "$CYTNX_BACKEND_DIR does not exist"
  require_vendored_suite
  override_tcapict
  cmake_build
}

main() {
  local cmd="${1:-}"
  case "$cmd" in
    check)
      prepare_and_build
      ;;
    test)
      prepare_and_build
      run_tests
      ;;
    restore)
      [[ -d "$CYTNX_BACKEND_DIR" ]] || fail "$CYTNX_BACKEND_DIR does not exist"
      restore_tcapict
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
