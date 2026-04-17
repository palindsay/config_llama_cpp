#!/usr/bin/env bash
# tests/test_logging.sh
set -euo pipefail

# Path to the script to test
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPT_UNDER_TEST="${ROOT_DIR}/setup-llamacpp-cuda-dgx-spark.sh"

# Source the script (it won't execute main because of the guard)
source "${SCRIPT_UNDER_TEST}"

# Helper for testing output and exit codes
# Usage: assert_output <stream> <expected_pattern> <command...>
# stream: 1 for stdout, 2 for stderr
assert_output() {
    local stream=$1
    local expected=$2
    shift 2
    local output
    local status=0

    # Capture output from the specified stream
    if [[ "$stream" -eq 1 ]]; then
        # Need to handle command arguments properly
        output=$( "$@" 2>/dev/null ) || status=$?
    else
        output=$( "$@" 2>&1 >/dev/null ) || status=$?
    fi

    # Check if the expected pattern is in the output
    # Using printf %q to handle ANSI escape codes in output if needed
    if [[ "$output" != *"$expected"* ]]; then
        printf "FAIL: Expected output to contain '%s', but got '%s'\n" "$expected" "$output"
        return 1
    fi
    return 0
}

# Usage: assert_exit_code <expected_code> <command...>
assert_exit_code() {
    local expected=$1
    shift
    local status=0
    ( "$@" ) >/dev/null 2>&1 || status=$?
    if [[ "$status" -ne "$expected" ]]; then
        echo "FAIL: Expected exit code $expected, but got $status"
        return 1
    fi
    return 0
}

test_info() {
    echo "Running test_info..."
    local green=$'\033[0;32m'
    local reset=$'\033[0m'
    assert_output 1 "${green}[INFO]${reset}" info "test message" || return 1
    assert_output 1 "test message" info "test message" || return 1
}

test_warn() {
    echo "Running test_warn..."
    local yellow=$'\033[0;33m'
    local reset=$'\033[0m'
    assert_output 1 "${yellow}[WARN]${reset}" warn "test message" || return 1
    assert_output 1 "test message" warn "test message" || return 1
}

test_error() {
    echo "Running test_error..."
    local red=$'\033[0;31m'
    local reset=$'\033[0m'
    assert_output 2 "${red}[ERROR]${reset}" error "test message" || return 1
    assert_output 2 "test message" error "test message" || return 1
}

test_die() {
    echo "Running test_die..."
    local red=$'\033[0;31m'
    local reset=$'\033[0m'
    # die calls error and then exits with 1
    assert_exit_code 1 die "test die message" || return 1
    # Run in a subshell for output capture because it exits
    assert_output 2 "${red}[ERROR]${reset}" die "test die message" || return 1
    assert_output 2 "test die message" die "test die message" || return 1
}

main() {
    local failed=0
    test_info || failed=1
    test_warn || failed=1
    test_error || failed=1
    test_die || failed=1

    if [[ $failed -eq 0 ]]; then
        echo "ALL TESTS PASSED"
    else
        echo "SOME TESTS FAILED"
        exit 1
    fi
}

main
