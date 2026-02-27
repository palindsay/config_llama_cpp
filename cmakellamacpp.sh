#!/bin/bash
#
# Optimized llama.cpp build for multi-GPU setup:
#   GPU 0: RTX 5090 (sm_120a, Blackwell, 32GB) - FP4/MXFP4 tensor cores
#   GPU 1: RTX 4090D (sm_89, Ada Lovelace, 48GB)
#   GPU 2: RTX 4090 (sm_89, Ada Lovelace, 24GB) - if installed
#   CPU:   AMD Ryzen 9 7950X (16C/32T, Zen 4, full AVX-512)
#   RAM:   128GB
#   CUDA:  13.1
#   OS:    Pop!_OS / Ubuntu 24.04
#
# CUDA ARCHITECTURE NOTES:
#   89-real   = Ada Lovelace (RTX 4090/4090D) - native SASS
#   120a-real = Blackwell (RTX 5090) - native SASS with FP4/MXFP4 tensor cores
#   The "a" suffix (architecture-specific) enables FP4 tensor cores and is
#   compatible with CMake 3.28+ (unlike the older "f" suffix which needs 3.31.8+)
#
# GPU KERNEL SELECTION:
#   GGML_CUDA_FORCE_MMQ=ON  - Use custom MMQ kernels (optimized for int8 tensor cores)
#   GGML_CUDA_FORCE_CUBLAS=ON - Use cuBLAS FP16 (may be faster for some models/batch sizes)
#   Both OFF (default) - Auto-select based on quantization type (recommended)
#
# MULTI-GPU NOTES:
#   PEER_MAX_BATCH_SIZE=256 - Enable peer-to-peer GPU transfers for batches up to 256 tokens
#   NO_PEER_COPY=OFF - Allow direct GPU-to-GPU memory copies (faster multi-GPU)
#
# CPU NOTES:
#   GGML_NATIVE=ON uses -march=native which auto-enables all Zen 4 ISA extensions:
#   AVX-512F/BW/VL/DQ/CD/VNNI/BF16/VBMI/VBMI2/IFMA/BITALG/VPOPCNTDQ, FMA, F16C, BMI2
#

set -euo pipefail

# ─── Install build dependencies ─────────────────────────────────────────────

echo "==> Checking and installing build dependencies..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    ccache \
    gcc-14 \
    g++-14 \
    pkg-config \
    git \
    curl \
    libcurl4-gnutls-dev \
    libssl-dev \
    libomp-dev

# ─── Clean stale build artifacts ────────────────────────────────────────────

echo "==> Cleaning previous build..."
rm -rf build

# ─── Configure ──────────────────────────────────────────────────────────────

echo "==> Running CMake configuration..."
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="89-real;120a-real" \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_CUDA_GRAPHS=ON \
    -DGGML_CUDA_PEER_MAX_BATCH_SIZE=256 \
    -DGGML_CUDA_NO_PEER_COPY=OFF \
    -DGGML_CUDA_COMPRESSION_MODE=size \
    -DGGML_NATIVE=ON \
    -DGGML_OPENMP=ON \
    -DGGML_LTO=ON \
    -DGGML_CCACHE=ON \
    -DLLAMA_CURL=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc-14 \
    -DCMAKE_CXX_COMPILER=g++-14 \
    -G Ninja

# ─── Build ──────────────────────────────────────────────────────────────────

echo ""
echo "==> Building with $(nproc) parallel jobs..."
cmake --build build -j$(nproc)

echo ""
echo "==> Build complete. Binaries are in build/bin/"
