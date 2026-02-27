#!/usr/bin/env bash
# =============================================================================
# setup-llamacpp-cuda-dgx-spark.sh
#
# Installs build dependencies and clones/configures llama.cpp for a
# CUDA-optimized build on NVIDIA DGX Spark (GB10 Grace Blackwell, aarch64).
#
# Target: DGX Spark — Ubuntu 24.04, CUDA 13.x, Blackwell sm_121, ARM Grace
#
# References:
#   [1] https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
#   [2] https://learn.arm.com/learning-paths/laptops-and-desktops/dgx_spark_llamacpp/2_gb10_llamacpp_gpu/
#   [3] https://github.com/ggml-org/llama.cpp/issues/15269  (perf flags)
#
# Usage:
#   chmod +x setup-llamacpp-cuda-dgx-spark.sh
#   sudo ./setup-llamacpp-cuda-dgx-spark.sh
# =============================================================================
set -euo pipefail

LLAMA_DIR="${LLAMA_DIR:-${HOME}/llama.cpp}"
BUILD_DIR="${LLAMA_DIR}/build"
NPROC="$(nproc)"

info()  { printf '\033[0;32m[INFO]\033[0m  %s\n' "$*"; }
warn()  { printf '\033[0;33m[WARN]\033[0m  %s\n' "$*"; }
error() { printf '\033[0;31m[ERROR]\033[0m %s\n' "$*" >&2; }
die()   { error "$@"; exit 1; }

# ── Pre-flight ───────────────────────────────────────────────────────────────
[[ "$(uname -m)" == "aarch64" ]] || die "Expected aarch64 (DGX Spark). Detected: $(uname -m)"
[[ $EUID -eq 0 ]] || die "Run with sudo."

# ── 1. Install build dependencies ───────────────────────────────────────────
# Ref [2]: "sudo apt install -y git cmake build-essential nvtop htop"
# Ref [1]: OpenSSL for TLS in llama-server; ccache for faster rebuilds
info "Installing build dependencies..."
apt-get update -y
apt-get install -y \
    build-essential \
    cmake \
    git \
    ccache \
    pkg-config \
    libssl-dev \
    libcurl4-openssl-dev \
    python3 \
    python3-pip \
    python3-venv \
    nvtop \
    htop

# ── 2. Verify CUDA toolkit ──────────────────────────────────────────────────
# DGX Spark ships CUDA 13.x pre-installed via DGX OS.
# Ref [2]: "nvcc --version" → release 13.0, V13.0.88
if ! command -v nvcc &>/dev/null; then
    die "nvcc not found. CUDA toolkit must be installed (DGX OS ships it pre-installed)."
fi
NVCC_VER="$(nvcc --version | grep -oP 'release \K[\d.]+')"
info "CUDA compiler: nvcc ${NVCC_VER}"

NVCC_MAJOR="${NVCC_VER%%.*}"
if (( NVCC_MAJOR < 13 )); then
    warn "Blackwell GB10 (sm_121) requires CUDA >= 13.0. Found ${NVCC_VER}."
    warn "Run: sudo apt update && sudo apt upgrade"
fi

if ! command -v nvidia-smi &>/dev/null; then
    warn "nvidia-smi not found — cannot verify GPU driver."
else
    info "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    info "Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
fi

# ── 3. Clone llama.cpp ──────────────────────────────────────────────────────
# Ref [1]: "git clone https://github.com/ggml-org/llama.cpp.git"
if [[ -d "${LLAMA_DIR}/.git" ]]; then
    info "llama.cpp already cloned at ${LLAMA_DIR} — pulling latest..."
    git -C "${LLAMA_DIR}" pull --ff-only
else
    info "Cloning llama.cpp into ${LLAMA_DIR}..."
    git clone https://github.com/ggml-org/llama.cpp.git "${LLAMA_DIR}"
fi

# ── 4. CMake configure — CUDA-optimized for DGX Spark ───────────────────────
#
# Flag justification (all verified against refs [1], [2], [3]):
#
#   -DGGML_CUDA=ON                    CUDA backend                    [1][2]
#   -DGGML_CUDA_F16=ON                Native FP16 half-precision      [2][3]
#   -DGGML_CUDA_FA_ALL_QUANTS=ON      Flash Attention all quant types  [1][3]
#   -DGGML_CUDA_FORCE_CUBLAS=ON       cuBLAS matmul (large batches)    [3]
#   -DCMAKE_CUDA_ARCHITECTURES=121    Blackwell GB10 = sm_121          [2]
#   -DGGML_NATIVE=ON                  ARM-native (SVE2/BF16/I8MM)      [1][2]
#   -DCMAKE_BUILD_TYPE=Release        Compiler optimizations           [1]
#   -DLLAMA_CURL=ON                   curl support (HF downloads)      [1]
#
info "Configuring CMake build..."
cmake -B "${BUILD_DIR}" -S "${LLAMA_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_F16=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_CUDA_FORCE_CUBLAS=ON \
    -DCMAKE_CUDA_ARCHITECTURES=121 \
    -DGGML_NATIVE=ON \
    -DLLAMA_CURL=ON \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_CUDA_COMPILER=nvcc

# ── 5. Build ─────────────────────────────────────────────────────────────────
info "Building with ${NPROC} parallel jobs..."
cmake --build "${BUILD_DIR}" --config Release -j"${NPROC}"

# ── 6. Verify ────────────────────────────────────────────────────────────────
info "Verifying CUDA linkage..."
if ldd "${BUILD_DIR}/bin/llama-cli" 2>/dev/null | grep -q libcudart; then
    info "llama-cli is linked against CUDA runtime ✓"
else
    warn "libcudart not found in ldd output — check the build log."
fi

info "Listing detected devices..."
"${BUILD_DIR}/bin/llama-cli" --list-devices 2>&1 || true

# ── Done ─────────────────────────────────────────────────────────────────────
cat <<EOF

══════════════════════════════════════════════════════════════
  Build complete: ${BUILD_DIR}/bin/
══════════════════════════════════════════════════════════════

  Quick test:
    ${BUILD_DIR}/bin/llama-cli -hf ggml-org/gemma-3-1b-it-GGUF -ngl 99 --flash-attn

  Start server:
    ${BUILD_DIR}/bin/llama-server -hf ggml-org/gemma-3-1b-it-GGUF -ngl 99 --flash-attn

  Unified memory (recommended for DGX Spark NVLink-C2C):
    export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1

EOF
