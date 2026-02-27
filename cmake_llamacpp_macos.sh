#!/usr/bin/env bash
set -euo pipefail

# Install Homebrew dependencies
brew install cmake ninja ccache libomp pkg-config openssl

# Resolve libomp prefix so CMake can find Apple's missing OpenMP runtime
LIBOMP_PREFIX="$(brew --prefix libomp)"

cmake -B build \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_METAL=ON \
    -DGGML_METAL_EMBED_LIBRARY=ON \
    -DGGML_METAL_NDEBUG=ON \
    -DGGML_ACCELERATE=ON \
    -DGGML_BLAS=ON \
    -DGGML_BLAS_VENDOR=Apple \
    -DGGML_NATIVE=ON \
    -DGGML_CPU_KLEIDIAI=ON \
    -DGGML_OPENMP=ON \
    -DGGML_LTO=ON \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include" \
    -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include" \
    -DOpenMP_C_LIB_NAMES="omp" \
    -DOpenMP_CXX_LIB_NAMES="omp" \
    -DOpenMP_omp_LIBRARY="${LIBOMP_PREFIX}/lib/libomp.dylib"

cmake --build build --config Release -j "$(sysctl -n hw.logicalcpu)"

echo ""
echo "Build complete. To run a model with full GPU offload:"
echo "  ./build/bin/llama-cli -m <model.gguf> -ngl 99 -p \"Hello\""
