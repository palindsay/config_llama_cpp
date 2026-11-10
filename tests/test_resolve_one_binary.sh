#!/usr/bin/env bash

set -euo pipefail

# Tests for _resolve_one_binary

# Get absolute path to the repo root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Source the target script (safe now because of BASH_SOURCE check)
source "${REPO_ROOT}/qwen35-122b-llamacpp-setup.sh"

# Assertion helpers
assert_eq() {
    local expected="$1"
    local actual="$2"
    local msg="${3:-}"
    if [[ "$expected" != "$actual" ]]; then
        echo "ASSERTION FAILED: ${msg}"
        echo "  Expected: '$expected'"
        echo "  Actual:   '$actual'"
        exit 1
    fi
}

# Run all tests
test_resolve_one_binary() {
    echo "Running test_resolve_one_binary..."

    local TEST_DIR
    TEST_DIR=$(mktemp -d)

    # Setup mock structure
    mkdir -p "$TEST_DIR/bin_dir"
    mkdir -p "$TEST_DIR/cpp_dir/build/bin"
    mkdir -p "$TEST_DIR/path_dir"

    # Add mock binaries
    touch "$TEST_DIR/bin_dir/test-bin"
    chmod +x "$TEST_DIR/bin_dir/test-bin"

    touch "$TEST_DIR/cpp_dir/build/bin/test-bin"
    chmod +x "$TEST_DIR/cpp_dir/build/bin/test-bin"

    touch "$TEST_DIR/path_dir/test-bin"
    chmod +x "$TEST_DIR/path_dir/test-bin"

    # 1. Test Explicit LLAMA_BIN_DIR priority
    LLAMA_BIN_DIR="$TEST_DIR/bin_dir"
    LLAMA_CPP_DIR="$TEST_DIR/cpp_dir"
    PATH="$TEST_DIR/path_dir:$PATH"

    local output
    output=$(_resolve_one_binary "test-bin")
    assert_eq "$TEST_DIR/bin_dir/test-bin" "$output" "Explicit LLAMA_BIN_DIR should have highest priority"

    # 2. Test Source-tree build priority (LLAMA_BIN_DIR empty)
    LLAMA_BIN_DIR=""
    output=$(_resolve_one_binary "test-bin")
    assert_eq "$TEST_DIR/cpp_dir/build/bin/test-bin" "$output" "Source-tree build should be fallback if LLAMA_BIN_DIR is empty"

    # 3. Test PATH fallback (LLAMA_BIN_DIR empty, LLAMA_CPP_DIR empty/invalid)
    LLAMA_CPP_DIR="$TEST_DIR/empty_cpp_dir"
    mkdir -p "$LLAMA_CPP_DIR"
    output=$(_resolve_one_binary "test-bin")
    assert_eq "$TEST_DIR/path_dir/test-bin" "$output" "PATH should be fallback if LLAMA_BIN_DIR and source-tree are invalid"

    # 4. Test Not found (expected to fail, so we capture error status)
    local status=0
    set +e
    output=$(_resolve_one_binary "missing-bin" 2>&1) || status=$?
    set -e

    if [[ $status -eq 0 ]]; then
        echo "ASSERTION FAILED: Should return non-zero exit status for missing binary"
        exit 1
    fi
    assert_eq "" "$output" "Should output nothing if binary is not found"

    # Cleanup
    rm -rf "$TEST_DIR"
    echo "test_resolve_one_binary passed!"
}

# Run tests
test_resolve_one_binary
