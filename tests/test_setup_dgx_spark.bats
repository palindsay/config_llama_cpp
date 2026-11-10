#!/usr/bin/env bats

setup() {
    # Source the script but don't run it due to the guard
    source ./setup-llamacpp-cuda-dgx-spark.sh
}

@test "info() prints correctly formatted green output" {
    run info "Test message"
    expected=$'\033[0;32m[INFO]\033[0m  Test message'
    [ "$status" -eq 0 ]
    [ "$output" = "$expected" ]
}

@test "warn() prints correctly formatted yellow output" {
    run warn "Test warning"
    expected=$'\033[0;33m[WARN]\033[0m  Test warning'
    [ "$status" -eq 0 ]
    [ "$output" = "$expected" ]
}

@test "error() prints correctly formatted red output to stderr" {
    # Bats 'run' captures both stdout and stderr by default
    run error "Test error"
    expected=$'\033[0;31m[ERROR]\033[0m Test error'
    [ "$status" -eq 0 ]
    [ "$output" = "$expected" ]
}

@test "die() prints error message and exits with status 1" {
    # Need to run die in a subshell or run command will catch the exit
    run bash -c "source ./setup-llamacpp-cuda-dgx-spark.sh && die 'Fatal error'"
    expected=$'\033[0;31m[ERROR]\033[0m Fatal error'
    [ "$status" -eq 1 ]
    [ "$output" = "$expected" ]
}
