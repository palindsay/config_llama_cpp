# llama.cpp Configuration Scripts

## Introduction

This repository contains a collection of highly optimized, platform-specific build and configuration scripts for [llama.cpp](https://github.com/ggml-org/llama.cpp). The goal is to provide developers, researchers, and AI enthusiasts with out-of-the-box scripts that automatically configure `llama.cpp` to squeeze the maximum performance out of specific hardware setups.

## Value Proposition

Building `llama.cpp` with the optimal flags for specific high-end or specialized hardware can be complex and time-consuming. These scripts solve that problem by:

- **Saving Time:** Eliminating the need to research and test various compilation flags.
- **Maximizing Performance:** Leveraging platform-specific features like Apple Metal, NVIDIA CUDA (Ada Lovelace & Blackwell architectures), and ARM-specific instructions (SVE2).
- **Automating Dependency Management:** Handling the installation of necessary tools, libraries, and compiler toolchains on supported operating systems.

## Included Scripts

### 1. Apple Silicon (macOS)

**File:** `cmake_llamacpp_macos.sh`

- **Target:** Apple Silicon (M1/M2/M3/M4) Macs.
- **Features:** Installs dependencies via Homebrew, resolves missing OpenMP runtimes for Apple Clang, and configures CMake for optimal Metal execution (`GGML_METAL`).
- **Optimizations:** Enables Apple Accelerate (`GGML_ACCELERATE`), native CPU instructions (`GGML_NATIVE`), and OpenMP for multi-threading.

### 2. Multi-GPU High-End PC (Linux)

**File:** `cmakellamacpp.sh`

- **Target:** Multi-GPU Linux setups (e.g., Pop!\_OS / Ubuntu 24.04). Specifically tailored for mixed architectures like RTX 5090 (Blackwell, sm_120a) and RTX 4090 (Ada Lovelace, sm_89) combined with high-end CPUs (e.g., AMD Ryzen 9 7950X).
- **Features:** Automates apt dependency installation and CUDA configuration (CUDA 13.1+).
- **Optimizations:** Enables Flash Attention, CUDA graphs, custom MMQ kernels for int8 tensor cores, peer-to-peer GPU memory transfers, and AVX-512 extensions for Zen 4 CPUs.

### 3. NVIDIA DGX Spark (Grace Blackwell)

**File:** `setup-llamacpp-cuda-dgx-spark.sh`

- **Target:** NVIDIA DGX Spark (GB10 Grace Blackwell, aarch64) running Ubuntu 24.04.
- **Features:** Clones and builds `llama.cpp` directly from source, handling apt dependencies and verifying CUDA 13.x toolchains.
- **Optimizations:** Configured for Blackwell `sm_121`, native FP16 (`GGML_CUDA_F16`), ARM-native instructions (`GGML_NATIVE`), Flash Attention, and cuBLAS matmul for large batches.

## How to Use

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```
2. **Make the scripts executable:**
   ```bash
   chmod +x *.sh
   ```
3. **Run the script appropriate for your hardware:**

   ```bash
   # For macOS
   ./cmake_llamacpp_macos.sh

   # For Linux Multi-GPU setup
   ./cmakellamacpp.sh

   # For DGX Spark
   sudo ./setup-llamacpp-cuda-dgx-spark.sh
   ```

_Keywords for search visibility:_ llama.cpp optimized build, RTX 5090 llama.cpp, DGX Spark LLM inference, Apple Silicon llama.cpp, CUDA 13.1, Blackwell sm_120a, sm_121, multi-gpu LLM setup.

## License

This project is licensed under the [MIT License](LICENSE).
