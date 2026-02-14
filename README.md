# Ollama Intel Arc (Battlemage) Fix

This repository provides a **custom native SYCL backend** for [Ollama](https://ollama.com/) (via `llama.cpp`) optimized for **Intel Arc B-Series (Battlemage)** GPUs.

It addresses performance issues and driver quirks by bypasssing the generic Vulkan path and using **Intel OneAPI** directly with custom XMX kernels.

## Features
-   🚀 **Intel XMX Acceleration**: Uses `intel::joint_matrix` for high-performance tensor operations (F16).
-   ⚡ **Low Latency**: Implements `immediate_command_list` queues to reduce submission overhead.
-   🛠️ **Dockerized Build**: No need to install OneAPI on your host system.

## Performance
Tested on **Dual Intel Arc B580**:
-   **Baseline (Vulkan):** ~9 tokens/sec
-   **This Backend (SYCL/XMX):** ~18-20 tokens/sec (**2x Speedup**)

## Prerequisites
-   Linux with Intel GPU Drivers installed.
-   Docker.

## Quick Start

### 1. Build the Custom Backend
Run the included build script. It will create a Docker image with the OneAPI compilers and build `llama-server`.

```bash
./build.sh
```

### 2. Run the Optimized Server
Use the provided `run_ollama_xmx.sh` (or `docker run` command) to launch the backend.

```bash
# Example: Launch with a GGUF model mapped
docker run -d --name ollama-xmx \
  --device /dev/dri:/dev/dri \
  -v ./llama.cpp-source/build/bin:/app \
  -v /path/to/your/models:/models \
  -p 8081:8080 \
  llama-oneapi \
  /app/llama-server -m /models/phi4.gguf -c 8192 --host 0.0.0.0 --port 8080 --n-gpu-layers 99
```

## Legacy Fixes
If you prefer to stick with the default Vulkan backend but are experiencing hangs with DeepSeek models, check `scripts/legacy/fix_deepseek.sh`.
