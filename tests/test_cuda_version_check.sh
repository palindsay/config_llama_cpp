#!/usr/bin/env bash

# Source the script to test. We want to be able to run this from tests/ or the root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../setup-llamacpp-cuda-dgx-spark.sh"

# Mock die function to avoid exiting the test suite
die() {
    echo "DIE: $*"
    exit 1
}

# Mock warn function to capture warnings
warn() {
    echo "WARN: $*"
}

# Mock info function to capture info messages
info() {
    echo "INFO: $*"
}

# Test 1: nvcc is missing
test_missing_nvcc() {
    echo "Running test_missing_nvcc..."

    # Temporarily remove nvcc from PATH for this test by mocking command
    command() {
        if [[ "$1" == "-v" && "$2" == "nvcc" ]]; then
            return 1
        fi
        builtin command "$@"
    }

    local output
    # Need to run in subshell since die calls exit 1 in original script, but we mocked it so we can just capture it
    output=$(check_cuda_version 2>&1)

    if [[ "$output" == *"DIE: nvcc not found"* ]]; then
        echo "PASS: Missing nvcc handled correctly."
    else
        echo "FAIL: Missing nvcc not handled. Output was: $output"
        return 1
    fi

    # Restore command
    unset -f command
}

# Test 2: CUDA version < 13
test_cuda_old() {
    echo "Running test_cuda_old..."

    # Mock nvcc to return version 12.1
    nvcc() {
        if [[ "$1" == "--version" ]]; then
            echo "Cuda compilation tools, release 12.1, V12.1.105"
        fi
    }

    # Ensure command -v nvcc returns true
    command() {
        if [[ "$1" == "-v" && "$2" == "nvcc" ]]; then
            return 0
        fi
        builtin command "$@"
    }

    local output
    output=$(check_cuda_version 2>&1)

    if [[ "$output" == *"WARN: Blackwell GB10 (sm_121) requires CUDA >= 13.0. Found 12.1"* ]]; then
        echo "PASS: Old CUDA version warns correctly."
    else
        echo "FAIL: Old CUDA version did not warn. Output was: $output"
        return 1
    fi

    # Restore
    unset -f nvcc
    unset -f command
}

# Test 3: CUDA version >= 13
test_cuda_new() {
    echo "Running test_cuda_new..."

    # Mock nvcc to return version 13.1
    nvcc() {
        if [[ "$1" == "--version" ]]; then
            echo "Cuda compilation tools, release 13.1, V13.1.88"
        fi
    }

    # Ensure command -v nvcc returns true
    command() {
        if [[ "$1" == "-v" && "$2" == "nvcc" ]]; then
            return 0
        fi
        builtin command "$@"
    }

    local output
    output=$(check_cuda_version 2>&1)

    if [[ "$output" == *"INFO: CUDA compiler: nvcc 13.1"* ]] && [[ "$output" != *"WARN"* ]]; then
        echo "PASS: New CUDA version handled correctly."
    else
        echo "FAIL: New CUDA version failed. Output was: $output"
        return 1
    fi

    # Restore
    unset -f nvcc
    unset -f command
}

main() {
    local failed=0

    test_missing_nvcc || failed=1
    test_cuda_old || failed=1
    test_cuda_new || failed=1

    if (( failed > 0 )); then
        echo "Some tests failed."
        exit 1
    else
        echo "All tests passed!"
        exit 0
    fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
