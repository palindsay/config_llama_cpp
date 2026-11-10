#!/usr/bin/env bats

setup() {
    export TEST_DIR=$(mktemp -d)

    # BATS_TEST_DIRNAME gives the directory of the bats file
    SCRIPT_PATH="${BATS_TEST_DIRNAME}/../setup-llamacpp-cuda-dgx-spark.sh"

    # Copy the script to test directory
    cp "$SCRIPT_PATH" "$TEST_DIR/setup-llamacpp-cuda-dgx-spark.sh"
    chmod +x "$TEST_DIR/setup-llamacpp-cuda-dgx-spark.sh"
}

teardown() {
    rm -rf "$TEST_DIR"
}

@test "setup script should fail with expected error when architecture is not aarch64" {
    # Mock uname
    uname() {
        if [[ "$1" == "-m" ]]; then
            echo "x86_64"
        else
            command uname "$@"
        fi
    }
    export -f uname

    # Run the script. It should fail (exit status 1)
    run "$TEST_DIR/setup-llamacpp-cuda-dgx-spark.sh"

    # Assert exit status is not 0
    [ "$status" -ne 0 ]

    # Assert output contains the expected error message with ANSI colors
    expected=$'\033[0;31m[ERROR]\033[0m Expected aarch64 (DGX Spark). Detected: x86_64'
    [[ "$output" == *"$expected"* ]]
}
