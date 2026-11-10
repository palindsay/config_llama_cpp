#!/usr/bin/env bats

setup() {
    # Create a mock bin directory
    export MOCK_BIN_DIR="$(mktemp -d)"
    export PATH="$MOCK_BIN_DIR:$PATH"

    # Create a mock uname command that outputs something other than aarch64
    cat << 'MOCK' > "$MOCK_BIN_DIR/uname"
#!/usr/bin/env bash
echo "x86_64"
MOCK
    chmod +x "$MOCK_BIN_DIR/uname"
}

teardown() {
    rm -rf "$MOCK_BIN_DIR"
}

@test "setup-llamacpp-cuda-dgx-spark.sh fails when OS architecture is not aarch64" {
    # Run the script from the root directory to avoid pathing bugs
    run bash "$BATS_TEST_DIRNAME/../setup-llamacpp-cuda-dgx-spark.sh"

    # Assert it fails
    [ "$status" -eq 1 ]

    # Assert the output matches the expected error
    expected_error=$'\033[0;31m[ERROR]\033[0m Expected aarch64 (DGX Spark). Detected: x86_64'
    [[ "$output" == *"$expected_error"* ]]
}
