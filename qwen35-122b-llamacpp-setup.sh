#!/usr/bin/env bash
###############################################################################
# qwen35_122b_setup.sh — Qwen3.5-122B-A10B Multi-GPU Setup for llama.cpp
#
# Target Hardware:
#   GPU0: NVIDIA RTX 5090  — 32 GB VRAM (sm_120, Blackwell)
#   GPU1: NVIDIA RTX 4090D — 48 GB VRAM (sm_89,  Ada Lovelace)
#   GPU2: NVIDIA RTX 4090  — 24 GB VRAM (sm_89,  Ada Lovelace)
#   Total VRAM: ~104 GB
#
# Host OS: Pop!_OS / Ubuntu 24.04 (Linux 6.x kernel)
#
# Models downloaded:
#   1. unsloth/Qwen3.5-122B-A10B-GGUF (UD-Q4_K_XL) — best performing dynamic quant
#   2. HauhauCS/Qwen3.5-122B-A10B-Uncensored-HauhauCS-Aggressive (Q4_K_P) — abliterated
#
# Architecture notes (Qwen3.5-122B-A10B):
#   - MoE: 122B total params, only 10B active per token
#   - 256 experts, 8 routed + 1 shared per token
#   - 48 layers: 12 × (3 × GatedDeltaNet-MoE + 1 × Attention-MoE)
#   - Hybrid attention: Gated Delta Networks + sparse MoE
#   - 262K native context (extendable to ~1M via YaRN)
#   - 201 languages, multimodal (text + vision)
#   - Thinking mode enabled by default
#
# References:
#   - https://github.com/ggml-org/llama.cpp
#   - https://unsloth.ai/docs/models/qwen3.5
#   - https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF
#   - https://huggingface.co/HauhauCS/Qwen3.5-122B-A10B-Uncensored-HauhauCS-Aggressive
#   - https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide
#   - https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html
#   - https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks
#
# Author: Phil Lindsay / Swartisan LLC
# Date:   2026-04-11
# License: MIT
###############################################################################
set -euo pipefail
IFS=$'\n\t'

# ===========================================================================
# Configuration — adjust these paths to your preference
#
# All settings can be overridden via environment variables, e.g.:
#   LLAMA_BIN_DIR=/opt/llama.cpp/build/bin ./qwen35_122b_setup.sh server
# ===========================================================================

# ---------------------------------------------------------------------------
# llama.cpp binary resolution (most important setting for existing installs)
# ---------------------------------------------------------------------------
# Option 1: Point to an existing build's bin/ directory.
#   LLAMA_BIN_DIR=/path/to/llama.cpp/build/bin
#
# Option 2: Leave empty — the script searches in this order:
#   a) ${LLAMA_CPP_DIR}/build/bin/  (the source-build default)
#   b) $PATH                        (system-installed llama.cpp)
#
# The 'build' command always builds into ${LLAMA_CPP_DIR}/build/bin/.
# ---------------------------------------------------------------------------
LLAMA_BIN_DIR="${LLAMA_BIN_DIR:-}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-${HOME}/llama.cpp}"  # source tree for 'build' cmd
MODELS_DIR="${MODELS_DIR:-${HOME}/models/qwen35-122b}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"

# Model repos
STANDARD_REPO="unsloth/Qwen3.5-122B-A10B-GGUF"
UNCENSORED_REPO="HauhauCS/Qwen3.5-122B-A10B-Uncensored-HauhauCS-Aggressive"

# Quant selections
# UD-Q4_K_XL: Unsloth Dynamic 2.0 — SOTA accuracy at 4-bit, outperforms
# standard Q4_K_M while ~8GB smaller per Unsloth KLD benchmarks (March 2026).
STANDARD_QUANT_PATTERN="*UD-Q4_K_XL*"
STANDARD_QUANT_LABEL="UD-Q4_K_XL"

# Q4_K_P: HauhauCS custom "Perfect" quant — model-specific analysis preserves
# quality at critical tensors, effectively 1-2 quant levels above standard Q4_K_M
# at only ~5-15% larger file size. 0/465 refusals (aggressive abliteration).
UNCENSORED_QUANT_FILE="Qwen3.5-122B-A10B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf"
UNCENSORED_MMPROJ_FILE="mmproj-Qwen3.5-122B-A10B-Uncensored-HauhauCS-Aggressive-f16.gguf"

# GPU configuration
# RTX 5090=32GB, RTX 4090D=48GB, RTX 4090=24GB (per nvidia-smi)
# tensor-split proportions (VRAM ratio, leaving ~2GB overhead per GPU)
TENSOR_SPLIT="30,46,22"
MAIN_GPU=1  # RTX 4090D has the most VRAM — use as main GPU for KV cache

# CUDA architectures: sm_89 (Ada/4090) + sm_120 (Blackwell/5090)
CUDA_ARCHITECTURES="89;120"

# Context and inference defaults
CTX_SIZE=32768         # 32K default — increase if you have headroom
MAX_PREDICT=16384      # max tokens per generation
THREADS="$(( $(nproc) / 2 ))"  # use half of CPU threads
BATCH_SIZE=4096
UBATCH_SIZE=4096

# Server port
SERVER_PORT=8080

# ===========================================================================
# Color helpers
# ===========================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "\n${CYAN}${BOLD}═══ $* ═══${NC}\n"; }

# ---------------------------------------------------------------------------
# Print a command array as a copy-pasteable command line, then execute it.
# Usage: log_cmd_and_exec cmd_array[@]
# ---------------------------------------------------------------------------
log_cmd_and_exec() {
    local -n _cmd_ref=$1

    echo ""
    echo -e "${CYAN}${BOLD}──── Command ────${NC}"
    # Print binary on first line, each subsequent arg indented with backslash
    local i=0
    local last_idx=$(( ${#_cmd_ref[@]} - 1 ))
    for arg in "${_cmd_ref[@]}"; do
        if [[ $i -eq 0 ]]; then
            if [[ $i -eq $last_idx ]]; then
                printf '%s\n' "$arg"
            else
                printf '%s \\\n' "$arg"
            fi
        elif [[ $i -eq $last_idx ]]; then
            printf '    %s\n' "$arg"
        else
            printf '    %s \\\n' "$arg"
        fi
        i=$(( i + 1 ))
    done
    echo -e "${CYAN}${BOLD}─────────────────${NC}"
    echo ""

    "${_cmd_ref[@]}"
}

# ===========================================================================
# Binary resolution — find llama-server, llama-cli, llama-bench
#
# Search order:
#   1. LLAMA_BIN_DIR (explicit override)
#   2. ${LLAMA_CPP_DIR}/build/bin/ (source-tree build)
#   3. $PATH (system-installed)
#
# Called lazily by commands that need binaries (server, chat, bench).
# The 'build' command always builds into ${LLAMA_CPP_DIR}/build/bin/.
# ===========================================================================
LLAMA_SERVER=""
LLAMA_CLI=""
LLAMA_BENCH=""
LLAMA_GGUF_SPLIT=""

_resolve_one_binary() {
    local name="$1"

    # 1. Explicit LLAMA_BIN_DIR
    if [[ -n "${LLAMA_BIN_DIR}" && -x "${LLAMA_BIN_DIR}/${name}" ]]; then
        echo "${LLAMA_BIN_DIR}/${name}"
        return 0
    fi

    # 2. Source-tree build
    if [[ -x "${LLAMA_CPP_DIR}/build/bin/${name}" ]]; then
        echo "${LLAMA_CPP_DIR}/build/bin/${name}"
        return 0
    fi

    # 3. $PATH
    if command -v "${name}" &>/dev/null; then
        command -v "${name}"
        return 0
    fi

    return 1
}

resolve_binaries() {
    LLAMA_SERVER=$(_resolve_one_binary llama-server) || true
    LLAMA_CLI=$(_resolve_one_binary llama-cli) || true
    LLAMA_BENCH=$(_resolve_one_binary llama-bench) || true
    LLAMA_GGUF_SPLIT=$(_resolve_one_binary llama-gguf-split) || true

    if [[ -n "${LLAMA_SERVER}" ]]; then
        local resolved_dir
        resolved_dir="$(dirname "${LLAMA_SERVER}")"
        log_info "llama.cpp binaries: ${resolved_dir}/"
        if [[ -n "${LLAMA_BIN_DIR}" ]]; then
            log_info "  (via LLAMA_BIN_DIR override)"
        fi
    fi
    # Print version from whichever binary we found
    # NOTE: capture to var first — piping --version through head causes SIGPIPE
    # under set -eo pipefail because --version prints multi-line CUDA device info.
    local version_output=""
    if [[ -n "${LLAMA_CLI}" ]]; then
        version_output=$("${LLAMA_CLI}" --version 2>&1) || true
    elif [[ -n "${LLAMA_SERVER}" ]]; then
        version_output=$("${LLAMA_SERVER}" --version 2>&1) || true
    fi
    if [[ -n "$version_output" ]]; then
        echo "$version_output" | head -3 | while read -r line; do
            log_info "  ${line}"
        done || true
    fi
}

require_binary() {
    local varname="$1"
    local binary_name="$2"
    local binary_path="${!varname}"

    if [[ -z "${binary_path}" || ! -x "${binary_path}" ]]; then
        log_error "'${binary_name}' not found."
        log_error ""
        log_error "Resolution search order:"
        log_error "  1. LLAMA_BIN_DIR=${LLAMA_BIN_DIR:-<not set>}"
        log_error "  2. ${LLAMA_CPP_DIR}/build/bin/"
        log_error "  3. \$PATH"
        log_error ""
        log_error "Options:"
        log_error "  a) Set LLAMA_BIN_DIR to your existing build:"
        log_error "     export LLAMA_BIN_DIR=/path/to/llama.cpp/build/bin"
        log_error "  b) Build from source:"
        log_error "     $0 build"
        log_error "  c) Ensure '${binary_name}' is on your PATH"
        exit 1
    fi
}

# ===========================================================================
# Step 0: Preflight checks
# ===========================================================================
preflight_checks() {
    log_step "STEP 0: Preflight Checks"

    # Check OS
    if [[ ! -f /etc/os-release ]]; then
        log_error "Cannot determine OS. Expected Ubuntu/Pop!_OS 24.04."
        exit 1
    fi
    source /etc/os-release
    log_info "OS: ${PRETTY_NAME:-unknown}"

    # Check NVIDIA driver + GPUs
    if ! command -v nvidia-smi &>/dev/null; then
        log_error "nvidia-smi not found. Install NVIDIA drivers first."
        exit 1
    fi
    log_info "NVIDIA Driver:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while read -r line; do
        log_info "  GPU $line"
    done

    local gpu_info
    gpu_info=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>&1) || true
    GPU_COUNT=$(echo "$gpu_info" | head -1)
    if [[ "$GPU_COUNT" -lt 3 ]]; then
        log_warn "Expected 3 GPUs, found ${GPU_COUNT}. Adjust TENSOR_SPLIT accordingly."
    fi

    # Check CUDA toolkit
    if ! command -v nvcc &>/dev/null; then
        log_warn "nvcc not found. Will attempt to install CUDA toolkit."
    else
        local cuda_ver
        cuda_ver=$(nvcc --version 2>&1) || true
        log_info "CUDA: $(echo "$cuda_ver" | grep 'release' | awk '{print $6}')"
    fi

    # Check disk space (need ~200GB for models + build)
    AVAIL_GB=$(df -BG "${HOME}" | awk 'NR==2{print $4}' | tr -d 'G')
    if [[ "$AVAIL_GB" -lt 200 ]]; then
        log_warn "Only ${AVAIL_GB}GB free in ${HOME}. Recommend 200GB+ for models + build."
    fi
    log_info "Available disk space: ${AVAIL_GB}GB"

    # Check RAM
    TOTAL_RAM_GB=$(free -g | awk '/Mem:/{print $2}')
    log_info "System RAM: ${TOTAL_RAM_GB}GB"
    if [[ "$TOTAL_RAM_GB" -lt 32 ]]; then
        log_warn "Recommend 64GB+ RAM for optimal MoE inference with CPU fallback."
    fi
}

# ===========================================================================
# Step 1: Install system dependencies
# ===========================================================================
install_dependencies() {
    log_step "STEP 1: Installing System Dependencies"

    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        build-essential \
        cmake \
        ninja-build \
        git \
        curl \
        wget \
        pciutils \
        libcurl4-openssl-dev \
        python3 \
        python3-pip \
        python3-venv \
        pkg-config

    # Install/update HuggingFace CLI tools
    # NOTE: as of 2026, `huggingface-cli` is deprecated — the new CLI is `hf`
    log_info "Installing HuggingFace Hub CLI (hf) + accelerators..."
    pip3 install --break-system-packages -q -U \
        huggingface_hub \
        hf_transfer \
        hf-xet 2>/dev/null || \
    pip3 install -q -U \
        huggingface_hub \
        hf_transfer \
        hf-xet

    # Enable fast HF downloads
    export HF_HUB_ENABLE_HF_TRANSFER=1

    # Verify hf CLI is available
    if command -v hf &>/dev/null; then
        local hf_ver
        hf_ver=$(hf --version 2>&1) || true
        log_info "HF CLI: $(echo "$hf_ver" | head -1)"
    else
        log_error "'hf' command not found after install. Check PATH or pip install."
        exit 1
    fi

    log_info "Dependencies installed."
}

# ---------------------------------------------------------------------------
# HF download wrapper — uses `hf download` (the current CLI as of 2026)
# Falls back to `huggingface-cli download` for older installs.
# ---------------------------------------------------------------------------
hf_download() {
    if command -v hf &>/dev/null; then
        hf download "$@"
    elif command -v huggingface-cli &>/dev/null; then
        huggingface-cli download "$@"
    else
        log_error "No HuggingFace CLI found. Run: $0 install"
        exit 1
    fi
}

# ===========================================================================
# Step 2: Build llama.cpp from source with multi-arch CUDA
# ===========================================================================
build_llama_cpp() {
    log_step "STEP 2: Building llama.cpp from Source (into ${LLAMA_CPP_DIR})"

    if [[ -d "${LLAMA_CPP_DIR}" ]]; then
        log_info "Updating existing llama.cpp source tree..."
        cd "${LLAMA_CPP_DIR}"
        git fetch --all --prune
        git reset --hard origin/master
        git pull origin master
    else
        log_info "Cloning llama.cpp..."
        git clone https://github.com/ggml-org/llama.cpp "${LLAMA_CPP_DIR}"
        cd "${LLAMA_CPP_DIR}"
    fi

    local COMMIT_HASH
    COMMIT_HASH=$(git rev-parse --short HEAD)
    log_info "Building commit: ${COMMIT_HASH}"

    # Clean previous build
    rm -rf build

    # Configure with CMake
    # Key flags:
    #   GGML_CUDA=ON                    — enable CUDA backend
    #   GGML_CUDA_FA_ALL_QUANTS=ON      — flash attention for ALL quant types
    #   GGML_NATIVE=ON                  — native CPU optimizations (AVX2/AVX-512)
    #   CMAKE_CUDA_ARCHITECTURES        — sm_89 (Ada) + sm_120 (Blackwell)
    #   BUILD_SHARED_LIBS=OFF           — static linking for portability
    log_info "Configuring CMake with CUDA architectures: ${CUDA_ARCHITECTURES}"

    cmake -G Ninja -B build \
        -DGGML_CUDA=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DGGML_NATIVE=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF

    # Build all key targets
    log_info "Building with ${BUILD_JOBS} parallel jobs..."
    cmake --build build --config Release -j "${BUILD_JOBS}" \
        --target llama-cli llama-server llama-bench llama-gguf-split

    # Verify build
    if [[ ! -x build/bin/llama-server ]]; then
        log_error "Build failed — llama-server not found."
        exit 1
    fi

    log_info "Build successful: ${LLAMA_CPP_DIR}/build/bin/"
    local build_ver
    build_ver=$(./build/bin/llama-cli --version 2>&1) || true
    echo "$build_ver" | head -5

    # Update runtime resolution to use this fresh build
    LLAMA_BIN_DIR="${LLAMA_CPP_DIR}/build/bin"
    resolve_binaries
}

# ===========================================================================
# Step 3: Download models
# ===========================================================================
download_models() {
    log_step "STEP 3: Downloading Models"

    mkdir -p "${MODELS_DIR}/standard"
    mkdir -p "${MODELS_DIR}/uncensored"

    export HF_HUB_ENABLE_HF_TRANSFER=1

    # -----------------------------------------------------------------------
    # 3a. Standard model: unsloth/Qwen3.5-122B-A10B-GGUF (UD-Q4_K_XL)
    # -----------------------------------------------------------------------
    log_info "Downloading standard model: ${STANDARD_REPO} (${STANDARD_QUANT_LABEL})..."
    log_info "This is Unsloth's Dynamic 2.0 quantization — SOTA accuracy at 4-bit."
    log_info "Updated March 5, 2026 with improved imatrix + quant algorithm."

    hf_download "${STANDARD_REPO}" \
        --local-dir "${MODELS_DIR}/standard" \
        --include "${STANDARD_QUANT_PATTERN}" \
        --include "*mmproj-F16*" || {
            log_warn "Bulk download failed, trying individual patterns..."
            hf_download "${STANDARD_REPO}" \
                --local-dir "${MODELS_DIR}/standard" \
                --include "${STANDARD_QUANT_PATTERN}"
            hf_download "${STANDARD_REPO}" \
                --local-dir "${MODELS_DIR}/standard" \
                --include "*mmproj-F16*"
        }

    # -----------------------------------------------------------------------
    # 3b. Uncensored model: HauhauCS Aggressive abliterated (Q4_K_P)
    # -----------------------------------------------------------------------
    log_info "Downloading uncensored model: ${UNCENSORED_REPO} (Q4_K_P)..."
    log_info "HauhauCS Aggressive: 0/465 refusals, custom K_P quant."
    log_info "K_P quants preserve critical tensors at higher precision (+1-2 levels)."

    hf_download "${UNCENSORED_REPO}" \
        --local-dir "${MODELS_DIR}/uncensored" \
        --include "${UNCENSORED_QUANT_FILE}" \
        --include "${UNCENSORED_MMPROJ_FILE}" || {
            log_warn "Specific file download failed, trying full repo download..."
            hf_download "${UNCENSORED_REPO}" \
                --local-dir "${MODELS_DIR}/uncensored"
        }

    # -----------------------------------------------------------------------
    # 3c. Verify downloads
    # -----------------------------------------------------------------------
    log_info "Downloaded files:"
    find "${MODELS_DIR}" -name "*.gguf" -exec ls -lh {} \; 2>/dev/null | \
        awk '{print "  " $5 "\t" $9}'

    log_info "Model downloads complete."
}

# ===========================================================================
# Step 4: Locate model files (handle multi-part GGUFs)
# ===========================================================================
find_model_file() {
    local search_dir="$1"
    local pattern="$2"
    local label="$3"

    # For multi-part GGUFs, we need the first shard (-00001-of-NNNNN.gguf)
    local found
    found=$(find "${search_dir}" -name "${pattern}" -type f 2>/dev/null | sort | head -1)

    if [[ -z "$found" ]]; then
        # Send diagnostics to stderr — this function is called inside $()
        # and stdout must only contain the file path or nothing
        log_error "Could not find ${label} model file matching '${pattern}' in ${search_dir}" >&2
        log_info "Available files:" >&2
        find "${search_dir}" -name "*.gguf" -type f 2>/dev/null >&2
        return 1
    fi

    echo "$found"
}

find_mmproj_file() {
    local search_dir="$1"
    local pattern="$2"

    local found
    found=$(find "${search_dir}" -name "${pattern}" -type f 2>/dev/null | head -1)

    if [[ -z "$found" ]]; then
        log_warn "mmproj file not found (vision disabled). Pattern: ${pattern}" >&2
        echo ""
        return 0
    fi

    echo "$found"
}

# ===========================================================================
# Step 5: Model variant resolution
#
# Resolves the GGUF model file for a given variant.
# Supported variants:
#   standard   (default) — unsloth/Qwen3.5-122B-A10B-GGUF UD-Q4_K_XL
#   uncensored / unc     — HauhauCS Aggressive Q4_K_P (abliterated)
# ===========================================================================
resolve_model_variant() {
    local variant="${1:-standard}"

    # Normalize short aliases
    case "$variant" in
        unc|uncensored|u|-u|--uncensored)
            variant="uncensored"
            ;;
        std|standard|s|-s|--standard|"")
            variant="standard"
            ;;
        *)
            log_error "Unknown model variant: '${variant}'"
            log_error "Valid variants: standard (std), uncensored (unc)"
            exit 1
            ;;
    esac

    ACTIVE_VARIANT="$variant"

    case "$variant" in
        standard)
            VARIANT_LABEL="Qwen3.5-122B-A10B UD-Q4_K_XL (standard)"
            MODEL_FILE=$(find_model_file \
                "${MODELS_DIR}/standard" \
                "*UD-Q4_K_XL*00001*.gguf" \
                "standard UD-Q4_K_XL") || \
            MODEL_FILE=$(find_model_file \
                "${MODELS_DIR}/standard" \
                "*UD-Q4_K_XL*.gguf" \
                "standard UD-Q4_K_XL") || exit 1
            ;;
        uncensored)
            VARIANT_LABEL="HauhauCS Aggressive Q4_K_P (uncensored/abliterated)"
            MODEL_FILE=$(find_model_file \
                "${MODELS_DIR}/uncensored" \
                "*Aggressive*Q4_K_P*.gguf" \
                "uncensored Q4_K_P") || \
            MODEL_FILE=$(find_model_file \
                "${MODELS_DIR}/uncensored" \
                "*Q4_K_P*.gguf" \
                "uncensored Q4_K_P") || exit 1
            ;;
    esac
}

# ===========================================================================
# Step 6: Run llama-server with optimized multi-GPU settings
# ===========================================================================
run_server() {
    local variant="${1:-standard}"
    resolve_model_variant "$variant"

    log_step "Launching Server — ${VARIANT_LABEL}"

    resolve_binaries
    require_binary LLAMA_SERVER llama-server

    # Note: mmproj vision is currently buggy on CUDA (issue #21268, Apr 2026)
    # Text-only inference works reliably. Uncomment --mmproj when fixed.

    log_info "Variant:      ${ACTIVE_VARIANT}"
    log_info "Model:        ${MODEL_FILE}"
    log_info "Tensor split: ${TENSOR_SPLIT} (GPU0:GPU1:GPU2)"
    log_info "Main GPU:     ${MAIN_GPU} (RTX 4090D — largest VRAM)"
    log_info "Context:      ${CTX_SIZE} tokens"
    log_info "Threads:      ${THREADS}"
    log_info "Port:         ${SERVER_PORT}"

    # -----------------------------------------------------------------------
    # Optimized llama-server launch
    #
    # Key flags explained:
    #   -ngl 999            All layers offloaded to GPU (MoE model fits in 104GB)
    #   -ts 30,46,22        VRAM proportional split (5090:4090D:4090 minus overhead)
    #   -mg 1               Main GPU = RTX 4090D (largest VRAM for KV cache)
    #   -sm layer           Split mode: layer-based (default, best for PCIe setups)
    #   -fa on              Flash Attention ON — critical for long-context performance
    #   -ctk q8_0           KV cache key quantization — saves ~50% KV VRAM
    #   -ctv q8_0           KV cache value quantization
    #   --jinja             Use embedded Jinja chat template (required for Qwen3.5)
    #   --no-mmap           Disable mmap — recommended for multi-GPU stability
    #   --reasoning-format deepseek   Parse <think>...</think> blocks properly
    #   --temp 0.6          Qwen3.5 recommended for coding/precise tasks
    #   --top-k 20          Official Qwen3.5 sampling parameter
    #   --top-p 0.95        Official Qwen3.5 sampling parameter
    #   --min-p 0.0         Official Qwen3.5 sampling parameter
    #   --no-context-shift  Prevents silent context truncation
    #   -b / -ub            Batch sizes tuned for multi-GPU MoE throughput
    # -----------------------------------------------------------------------

    local cmd=(
        "${LLAMA_SERVER}"
        --model "${MODEL_FILE}"
        --port "${SERVER_PORT}"
        --host 0.0.0.0
        --gpu-layers 999
        --tensor-split "${TENSOR_SPLIT}"
        --main-gpu "${MAIN_GPU}"
        --split-mode layer
        --flash-attn on
        --cache-type-k q8_0
        --cache-type-v q8_0
        --jinja
        --no-mmap
        --reasoning-format deepseek
        --ctx-size "${CTX_SIZE}"
        --n-predict "${MAX_PREDICT}"
        --batch-size "${BATCH_SIZE}"
        --ubatch-size "${UBATCH_SIZE}"
        --threads "${THREADS}"
        --threads-batch "${THREADS}"
        --temp 0.6
        --top-k 20
        --top-p 0.95
        --min-p 0.0
        --no-context-shift
    )

    log_cmd_and_exec cmd
}

# ===========================================================================
# Step 7: Interactive chat mode (llama-cli)
# ===========================================================================
run_chat() {
    local variant="${1:-standard}"
    resolve_model_variant "$variant"

    log_step "Launching Interactive Chat — ${VARIANT_LABEL}"

    resolve_binaries
    require_binary LLAMA_CLI llama-cli

    local cmd=(
        "${LLAMA_CLI}"
        --model "${MODEL_FILE}"
        --gpu-layers 999
        --tensor-split "${TENSOR_SPLIT}"
        --main-gpu "${MAIN_GPU}"
        --split-mode layer
        --flash-attn on
        --cache-type-k q8_0
        --cache-type-v q8_0
        --jinja
        --no-mmap
        --ctx-size "${CTX_SIZE}"
        --threads "${THREADS}"
        --temp 0.6
        --top-k 20
        --top-p 0.95
        --min-p 0.0
        --color
    )

    log_cmd_and_exec cmd
}

# ===========================================================================
# Step 7: Benchmark
# ===========================================================================
run_benchmark() {
    local variant="${1:-standard}"
    resolve_model_variant "$variant"

    log_step "Running llama-bench — ${VARIANT_LABEL}"

    resolve_binaries
    require_binary LLAMA_BENCH llama-bench

    local bench_ver
    bench_ver=$("${LLAMA_BENCH}" --version 2>&1) || true
    if [[ -n "$bench_ver" ]]; then
        echo "$bench_ver" | head -1 | while read -r line; do
            log_info "  ${line}"
        done || true
    fi

    local cmd=(
        "${LLAMA_BENCH}"
        --model "${MODEL_FILE}"
        --n-gpu-layers 999
        --flash-attn 1
        --threads "${THREADS}"
        --repetitions 3
    )

    log_cmd_and_exec cmd
}

# ===========================================================================
# Usage / Help
# ===========================================================================
print_usage() {
    cat <<'USAGE'

╔══════════════════════════════════════════════════════════════════════════════╗
║              Qwen3.5-122B-A10B Multi-GPU Setup Script                      ║
║                                                                            ║
║  GPU0: RTX 5090 (32GB)  GPU1: RTX 4090D (48GB)  GPU2: RTX 4090 (24GB)    ║
║  Total VRAM: ~104 GB  |  Active params: 10B per token (MoE)               ║
╚══════════════════════════════════════════════════════════════════════════════╝

USAGE
    echo -e "${BOLD}Usage:${NC} $0 <command>"
    echo ""
    echo -e "${BOLD}Setup commands (run in order for first-time setup):${NC}"
    echo "  install       Install all system dependencies"
    echo "  build         Clone/update and build llama.cpp from source"
    echo "  download      Download both model variants (standard + uncensored)"
    echo "  setup         Run all of the above in sequence"
    echo ""
    echo -e "${BOLD}Run commands:${NC}"
    echo "  server [VARIANT]   Launch OpenAI-compatible API server"
    echo "  chat   [VARIANT]   Interactive CLI chat"
    echo "  bench  [VARIANT]   Run llama-bench performance benchmark"
    echo ""
    echo -e "${BOLD}Model variants (optional argument to server/chat/bench):${NC}"
    echo "  standard           Unsloth UD-Q4_K_XL (default if omitted)"
    echo "  uncensored / unc   HauhauCS Aggressive Q4_K_P (abliterated)"
    echo ""
    echo -e "${BOLD}Legacy aliases (backwards-compatible):${NC}"
    echo "  server-unc         Same as: server uncensored"
    echo "  chat-unc           Same as: chat uncensored"
    echo ""
    echo -e "${BOLD}Utilities:${NC}"
    echo "  check         Run preflight hardware/software checks only"
    echo "  help          Show this help message"
    echo ""
    echo -e "${BOLD}Examples:${NC}"
    echo "  $0 setup                    # Full first-time setup (build + download)"
    echo "  $0 server                   # Standard model on port ${SERVER_PORT}"
    echo "  $0 server unc               # Uncensored model on port ${SERVER_PORT}"
    echo "  $0 server uncensored        # Same as above (long form)"
    echo "  $0 chat                     # Interactive chat (standard)"
    echo "  $0 chat unc                 # Interactive chat (uncensored)"
    echo "  $0 bench unc                # Benchmark the uncensored model"
    echo ""
    echo -e "${BOLD}Using an existing llama.cpp build:${NC}"
    echo "  export LLAMA_BIN_DIR=/path/to/llama.cpp/build/bin"
    echo "  $0 download           # Download models only (skip build)"
    echo "  $0 server             # Uses your existing binaries"
    echo ""
    echo "  # Or inline:"
    echo "  LLAMA_BIN_DIR=/opt/llama.cpp/build/bin $0 server"
    echo ""
    echo -e "${BOLD}Configuration (edit at top of script, or override via env):${NC}"
    echo "  LLAMA_BIN_DIR=${LLAMA_BIN_DIR:-<auto-detect>}"
    echo "  LLAMA_CPP_DIR=${LLAMA_CPP_DIR}  (source tree for 'build' cmd)"
    echo "  MODELS_DIR=${MODELS_DIR}"
    echo "  TENSOR_SPLIT=${TENSOR_SPLIT}"
    echo "  CTX_SIZE=${CTX_SIZE}"
    echo "  SERVER_PORT=${SERVER_PORT}"
    echo ""
    echo -e "${BOLD}API usage after 'server' command:${NC}"
    echo '  curl http://localhost:8080/v1/chat/completions \'
    echo '    -H "Content-Type: application/json" \'
    echo '    -d '"'"'{"model":"qwen3.5","messages":[{"role":"user","content":"Hello!"}]}'"'"
    echo ""
    echo -e "${BOLD}Disable thinking mode (non-thinking for faster direct responses):${NC}"
    echo "  Add to server command: --chat-template-kwargs '{\"enable_thinking\":false}'"
    echo ""
    echo -e "${BOLD}Notes:${NC}"
    echo "  • Vision (mmproj) has a known CUDA bug as of Apr 2026 (issue #21268)."
    echo "    Text inference works perfectly. Vision may crash — use --no-mmproj."
    echo "  • For repetition issues, add: --presence-penalty 1.5"
    echo "  • Increase context: --ctx-size 65536 (requires more KV cache VRAM)"
    echo "  • The --cpu-moe flag has known corruption issues (issue #20140)."
    echo "    With 104GB VRAM, full GPU offload at Q4 is preferred."
    echo ""
}

# ===========================================================================
# Main dispatch
# ===========================================================================
main() {
    if [[ $# -lt 1 ]]; then
        print_usage
        exit 0
    fi

    local cmd="$1"
    shift

    case "$cmd" in
        check)
            preflight_checks
            ;;
        install)
            preflight_checks
            install_dependencies
            ;;
        build)
            preflight_checks
            build_llama_cpp
            ;;
        download)
            preflight_checks
            download_models
            ;;
        setup)
            preflight_checks
            install_dependencies
            build_llama_cpp
            download_models
            log_step "SETUP COMPLETE"
            log_info "Standard model:    ${MODELS_DIR}/standard/"
            log_info "Uncensored model:  ${MODELS_DIR}/uncensored/"
            log_info ""
            log_info "Next steps:"
            log_info "  $0 server           # Launch standard model server"
            log_info "  $0 server unc       # Launch uncensored model server"
            log_info "  $0 chat             # Interactive chat"
            ;;
        server)
            run_server "${1:-standard}"
            ;;
        server-unc)
            run_server uncensored
            ;;
        chat)
            run_chat "${1:-standard}"
            ;;
        chat-unc)
            run_chat uncensored
            ;;
        bench)
            run_benchmark "${1:-standard}"
            ;;
        help|--help|-h)
            print_usage
            ;;
        *)
            log_error "Unknown command: ${cmd}"
            print_usage
            exit 1
            ;;
    esac
}

main "$@"
