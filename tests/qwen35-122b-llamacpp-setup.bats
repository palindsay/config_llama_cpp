#!/usr/bin/env bats

setup() {
    export TEST_TEMP_DIR=$(mktemp -d)
    export LLAMA_CPP_DIR="${TEST_TEMP_DIR}/llama.cpp"
    export LLAMA_BIN_DIR="${TEST_TEMP_DIR}/bin"
    export MODELS_DIR="${TEST_TEMP_DIR}/models"

    mkdir -p "$LLAMA_CPP_DIR/build/bin"
    mkdir -p "$LLAMA_BIN_DIR"
    mkdir -p "$MODELS_DIR/standard"
    mkdir -p "$MODELS_DIR/uncensored"

    # Source the script, suppressing stdout/stderr on load if any
    source ./qwen35-122b-llamacpp-setup.sh >/dev/null 2>&1
}

teardown() {
    rm -rf "$TEST_TEMP_DIR"
}

@test "_resolve_one_binary finds binary in LLAMA_BIN_DIR" {
    # Setup test binary
    local test_bin="llama-test-bin"
    touch "${LLAMA_BIN_DIR}/${test_bin}"
    chmod +x "${LLAMA_BIN_DIR}/${test_bin}"

    run _resolve_one_binary "${test_bin}"
    [ "$status" -eq 0 ]
    [ "$output" = "${LLAMA_BIN_DIR}/${test_bin}" ]
}

@test "_resolve_one_binary finds binary in LLAMA_CPP_DIR/build/bin when LLAMA_BIN_DIR is empty" {
    local test_bin="llama-test-bin"
    # Unset LLAMA_BIN_DIR to fallback to LLAMA_CPP_DIR
    export LLAMA_BIN_DIR=""

    touch "${LLAMA_CPP_DIR}/build/bin/${test_bin}"
    chmod +x "${LLAMA_CPP_DIR}/build/bin/${test_bin}"

    run _resolve_one_binary "${test_bin}"
    [ "$status" -eq 0 ]
    [ "$output" = "${LLAMA_CPP_DIR}/build/bin/${test_bin}" ]
}

@test "_resolve_one_binary finds binary in PATH" {
    local test_bin="llama-test-bin"
    export LLAMA_BIN_DIR=""
    export LLAMA_CPP_DIR="/nonexistent"

    # Create bin in temp dir and add to PATH
    local temp_path="${TEST_TEMP_DIR}/path_bin"
    mkdir -p "${temp_path}"
    touch "${temp_path}/${test_bin}"
    chmod +x "${temp_path}/${test_bin}"

    export PATH="${temp_path}:${PATH}"

    run _resolve_one_binary "${test_bin}"
    [ "$status" -eq 0 ]
    [ "$output" = "${temp_path}/${test_bin}" ]
}

@test "_resolve_one_binary fails when binary is not found" {
    local test_bin="llama-test-bin-not-found"
    export LLAMA_BIN_DIR=""
    export LLAMA_CPP_DIR="/nonexistent"

    run _resolve_one_binary "${test_bin}"
    [ "$status" -eq 1 ]
    [ "$output" = "" ]
}

@test "find_model_file finds the single gguf file" {
    local test_dir="${TEST_TEMP_DIR}/models/single"
    mkdir -p "${test_dir}"
    local test_file="model-Q4_K.gguf"
    touch "${test_dir}/${test_file}"

    run find_model_file "${test_dir}" "*Q4_K*.gguf" "test_model"
    [ "$status" -eq 0 ]
    [ "$output" = "${test_dir}/${test_file}" ]
}

@test "find_model_file finds the correct shard of a multi-part gguf file" {
    local test_dir="${TEST_TEMP_DIR}/models/multi"
    mkdir -p "${test_dir}"

    # Create shards out of order to test sorting
    touch "${test_dir}/model-00002-of-00003.gguf"
    touch "${test_dir}/model-00003-of-00003.gguf"
    touch "${test_dir}/model-00001-of-00003.gguf"

    run find_model_file "${test_dir}" "model-*.gguf" "test_model"
    [ "$status" -eq 0 ]
    [ "$output" = "${test_dir}/model-00001-of-00003.gguf" ]
}

@test "find_model_file fails and outputs error to stderr when no file matches" {
    local test_dir="${TEST_TEMP_DIR}/models/empty"
    mkdir -p "${test_dir}"

    # Run the function in a subshell capturing stderr to stdout to test the error message
    run bash -c "source ./qwen35-122b-llamacpp-setup.sh && find_model_file \"${test_dir}\" \"*.gguf\" \"test_model\" 2>&1"
    [ "$status" -eq 1 ]
    # Check that output contains the error message (we don't need an exact string match for colored output, just partial)
    [[ "$output" == *"Could not find test_model model file matching '*.gguf' in ${test_dir}"* ]]
}

@test "find_mmproj_file finds the mmproj file" {
    local test_dir="${TEST_TEMP_DIR}/models/mmproj"
    mkdir -p "${test_dir}"
    local test_file="mmproj-model-f16.gguf"
    touch "${test_dir}/${test_file}"

    run find_mmproj_file "${test_dir}" "mmproj-*.gguf"
    [ "$status" -eq 0 ]
    [ "$output" = "${test_dir}/${test_file}" ]
}

@test "find_mmproj_file returns empty string when not found without failing" {
    local test_dir="${TEST_TEMP_DIR}/models/mmproj_empty"
    mkdir -p "${test_dir}"

    # We capture stdout and discard stderr here to check the actual returned value
    run bash -c "source ./qwen35-122b-llamacpp-setup.sh && find_mmproj_file \"${test_dir}\" \"mmproj-*.gguf\" 2>/dev/null"
    [ "$status" -eq 0 ]
    [ "$output" = "" ]
}

@test "resolve_model_variant defaults to standard variant" {
    touch "${MODELS_DIR}/standard/model-UD-Q4_K_XL.gguf"

    # We must run this in a context where the function can set variables
    # But since bats run encapsulates in a subshell we evaluate and echo
    run bash -c "source ./qwen35-122b-llamacpp-setup.sh && MODELS_DIR=\"${MODELS_DIR}\" resolve_model_variant && echo \"\$ACTIVE_VARIANT\""
    [ "$status" -eq 0 ]
    [[ "$output" == *"standard"* ]]
}

@test "resolve_model_variant resolves standard aliases" {
    touch "${MODELS_DIR}/standard/model-UD-Q4_K_XL.gguf"

    for alias in "std" "s" "-s" "--standard"; do
        run bash -c "source ./qwen35-122b-llamacpp-setup.sh && MODELS_DIR=\"${MODELS_DIR}\" resolve_model_variant \"${alias}\" && echo \"\$ACTIVE_VARIANT\""
        [ "$status" -eq 0 ]
        [[ "$output" == *"standard"* ]]
    done
}

@test "resolve_model_variant resolves uncensored aliases" {
    touch "${MODELS_DIR}/uncensored/model-Q4_K_P.gguf"

    for alias in "unc" "uncensored" "u" "-u" "--uncensored"; do
        run bash -c "source ./qwen35-122b-llamacpp-setup.sh && MODELS_DIR=\"${MODELS_DIR}\" resolve_model_variant \"${alias}\" && echo \"\$ACTIVE_VARIANT\""
        [ "$status" -eq 0 ]
        [[ "$output" == *"uncensored"* ]]
    done
}

@test "resolve_model_variant fails on unknown variant" {
    run bash -c "source ./qwen35-122b-llamacpp-setup.sh && MODELS_DIR=\"${MODELS_DIR}\" resolve_model_variant \"unknown_variant\" 2>&1"
    [ "$status" -eq 1 ]
    [[ "$output" == *"Unknown model variant: 'unknown_variant'"* ]]
}
