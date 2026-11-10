#!/usr/bin/env bats

setup() {
    # Since bats runs from the directory the bats command was invoked (or the test's directory)
    # We should make sure we can source the main script correctly
    SCRIPT_PATH="$(dirname "$BATS_TEST_FILENAME")/../qwen35-122b-llamacpp-setup.sh"
    chmod +x "$SCRIPT_PATH"
}

@test "resolve_model_variant returns error on unknown variant" {
    SCRIPT_PATH="$(dirname "$BATS_TEST_FILENAME")/../qwen35-122b-llamacpp-setup.sh"

    # We use a subshell because resolve_model_variant calls exit 1 on error
    run bash -c "source \"$SCRIPT_PATH\" && resolve_model_variant 'unknown_variant'"

    # Assert exit status is 1
    [ "$status" -eq 1 ]

    # The output includes both stdout and stderr, with color codes
    # We'll assert that the strings are present in the output
    [[ "${output}" == *"Unknown model variant: 'unknown_variant'"* ]]
    [[ "${output}" == *"Valid variants: standard (std), uncensored (unc)"* ]]
}
