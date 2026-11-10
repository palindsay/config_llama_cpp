#!/usr/bin/env bash
set -euo pipefail

# Test setup
export TEST_DIR=$(mktemp -d)
trap 'rm -rf "${TEST_DIR}"' EXIT

export LLAMA_BIN_DIR="${TEST_DIR}/bin_override"
export LLAMA_CPP_DIR="${TEST_DIR}/src_tree"
export PATH="${TEST_DIR}/system_path:${PATH}"

# Source the script to be tested
source ./qwen35-122b-llamacpp-setup.sh

# Color helpers for test output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

test_passed=0
test_failed=0

run_test() {
    local test_name="$1"
    local func_to_run="$2"

    if $func_to_run; then
        echo -e "${GREEN}PASS${NC}: ${test_name}"
        ((test_passed++)) || true
    else
        echo -e "${RED}FAIL${NC}: ${test_name}"
        ((test_failed++)) || true
    fi
}

# --- Test Cases ---

test_resolve_via_bin_dir() {
    # Setup
    mkdir -p "${LLAMA_BIN_DIR}"
    touch "${LLAMA_BIN_DIR}/llama-server"
    chmod +x "${LLAMA_BIN_DIR}/llama-server"

    # Execution
    local result
    result=$(_resolve_one_binary "llama-server")
    local status=$?

    # Cleanup
    rm -rf "${LLAMA_BIN_DIR}"

    # Assertion
    [[ $status -eq 0 ]] && [[ "$result" == "${LLAMA_BIN_DIR}/llama-server" ]]
}

test_resolve_via_cpp_dir() {
    # Setup
    export LLAMA_BIN_DIR=""
    mkdir -p "${LLAMA_CPP_DIR}/build/bin"
    touch "${LLAMA_CPP_DIR}/build/bin/llama-cli"
    chmod +x "${LLAMA_CPP_DIR}/build/bin/llama-cli"

    # Execution
    local result
    result=$(_resolve_one_binary "llama-cli")
    local status=$?

    # Cleanup
    rm -rf "${LLAMA_CPP_DIR}"

    # Assertion
    [[ $status -eq 0 ]] && [[ "$result" == "${LLAMA_CPP_DIR}/build/bin/llama-cli" ]]
}

test_resolve_via_path() {
    # Setup
    export LLAMA_BIN_DIR=""
    export LLAMA_CPP_DIR=""
    mkdir -p "${TEST_DIR}/system_path"
    touch "${TEST_DIR}/system_path/llama-bench"
    chmod +x "${TEST_DIR}/system_path/llama-bench"

    # Execution
    local result
    result=$(_resolve_one_binary "llama-bench")
    local status=$?

    # Cleanup
    rm -rf "${TEST_DIR}/system_path"

    # Assertion
    [[ $status -eq 0 ]] && [[ "$result" == "${TEST_DIR}/system_path/llama-bench" ]]
}

test_resolve_not_found() {
    # Setup
    export LLAMA_BIN_DIR=""
    export LLAMA_CPP_DIR=""

    # Need to disable set -e for this specific block as we EXPECT failure
    set +e
    local result
    result=$(_resolve_one_binary "llama-nonexistent" 2>&1)
    local status=$?
    set -e

    # Assertion
    [[ $status -ne 0 ]] && [[ -z "$result" ]]
}


# --- Run Tests ---
echo "Running tests for _resolve_one_binary..."
run_test "Resolves via LLAMA_BIN_DIR explicitly" test_resolve_via_bin_dir
run_test "Resolves via LLAMA_CPP_DIR/build/bin" test_resolve_via_cpp_dir
run_test "Resolves via PATH" test_resolve_via_path
run_test "Fails gracefully when binary not found" test_resolve_not_found

echo ""
echo "Tests Passed: $test_passed"
echo "Tests Failed: $test_failed"

if [[ $test_failed -gt 0 ]]; then
    exit 1
fi
exit 0
