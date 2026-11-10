#!/usr/bin/env bats

setup() {
    # Source the script under test
    source ./qwen35-122b-llamacpp-setup.sh

    # Create a temporary directory for test binaries
    TEST_BIN_DIR=$(mktemp -d)

    # Create a non-executable file
    touch "${TEST_BIN_DIR}/non_executable"

    # Create an executable file
    touch "${TEST_BIN_DIR}/my_executable"
    chmod +x "${TEST_BIN_DIR}/my_executable"
}

teardown() {
    # Clean up the temporary directory
    rm -rf "${TEST_BIN_DIR}"
}

@test "require_binary fails when variable is empty" {
    TEST_VAR=""
    run require_binary TEST_VAR "empty_binary"
    [ "$status" -eq 1 ]
    [[ "${lines[0]}" == *"[ERROR]"*"'empty_binary' not found."* ]]
}

@test "require_binary fails when binary path does not exist" {
    TEST_VAR="/path/that/does/not/exist"
    run require_binary TEST_VAR "missing_binary"
    [ "$status" -eq 1 ]
    [[ "${lines[0]}" == *"[ERROR]"*"'missing_binary' not found."* ]]
}

@test "require_binary fails when binary path is not executable" {
    TEST_VAR="${TEST_BIN_DIR}/non_executable"
    run require_binary TEST_VAR "non_executable_binary"
    [ "$status" -eq 1 ]
    [[ "${lines[0]}" == *"[ERROR]"*"'non_executable_binary' not found."* ]]
}

@test "require_binary succeeds when binary path is executable" {
    TEST_VAR="${TEST_BIN_DIR}/my_executable"
    run require_binary TEST_VAR "my_executable"
    [ "$status" -eq 0 ]
    [ "${#lines[@]}" -eq 0 ]
}
