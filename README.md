# Ollama-Intel-Fix for Arc Battlemage (Dual B580, B50, B60 / Xe2)

This repository contains the **custom backend and proxy** required to run Ollama with hardware acceleration on Intel Arc Battlemage (Xe2) GPUs.

Standard Ollama builds do not yet support the Xe2 architecture or multi-GPU splitting correctly. This solution provides:

1.  **Custom Compilation**: A `llama-server` binary built from source with Intel OneAPI 2025.3 and XMX optimization patches.
2.  **Ollama Proxy**: A smart Python wrapper that handles model loading, environment variables (`setvars.sh`), and memory safety.
3.  **Auto-Split (Multi-GPU)**: Automatically shards large models (>11GB) across dual GPUs while keeping small models fast on a single card.

## Prerequisites

*   **Hardware**: Intel Arc Battlemage (B580/B570) - Single or Dual.
*   **Drivers**: Intel Compute Runtime (Level Zero) installed on host.
*   **Docker**: With `/dev/dri` access.

## Quick Start (Pre-Built Image)

If you have built the image `ollama-custom-xmx:latest` (instructions below), run:

```bash
docker run -d \
  --restart always \
  -p 11434:8080 \
  -v ollama_data:/ollama_data \
  --device /dev/dri \
  --name ollama-custom-xmx \
  ollama-custom-xmx
```

Then connect via any Ollama client (OpenWebUI, Curl, etc) at `http://localhost:11434`.

## Features

### 1. Smart Auto-Split (Dual GPU Support)
The Proxy checks the model size before loading:
*   **< 11GB (e.g., Llama3-8B, Phi-4)**: Runs on **Single GPU**. Fastest inference, zero overhead.
*   **> 11GB (e.g., Qwen-14B, Command-R)**: Automatically enables `--split-mode layer`. Uses both B580s as a single 24GB VRAM pool.

**Manual Override**: Force split mode for small models by adding `-e SPLIT_MODE=true` to the `docker run` command.

### 2. Memory Safety
*   **Context Limit**: Default context is capped at **2048 tokens** to prevent OOM crashes on 12GB cards with larger models.
*   **Environment Injection**: Automatically sources Intel OneAPI environment variables which are often missing in standard containers.

## Build From Source

To build the `ollama-custom-xmx` image yourself:

1.  **Clone this repo**:
    ```bash
    git clone https://github.com/qbnasasn/Ollama-Intel-Fix
    cd Ollama-Intel-Fix
    ```

2.  **Build the Builder Image**:
    ```bash
    docker build -t llama-oneapi -f docker/Dockerfile.builder docker/
    ```

3.  **Compile the Backend**:
    *Requires `llama.cpp` source code mounted to `/llama-source`.*
    ```bash
    git clone https://github.com/ggml-org/llama.cpp llama.cpp-source
    
    docker run --rm -v $(pwd)/llama.cpp-source:/llama-source -w /llama-source llama-oneapi bash -c \
      "source /opt/intel/oneapi/setvars.sh --force ; \
       rm -rf build && \
       cmake -B build -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL \
       -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
       -DBUILD_SHARED_LIBS=OFF -DGGML_STATIC=ON -DCMAKE_EXE_LINKER_FLAGS='-static-intel' && \
       cmake --build build --config Release -j \$(nproc)"
    ```

4.  **Build the Runtime Image**:
    ```bash
    docker build --no-cache -t ollama-custom-xmx -f docker/Dockerfile.runtime .
    ```

## Project Structure

*   `docker/`: Dockerfiles for the build and runtime stages.
*   `scripts/`: Python proxy (`ollama_proxy.py`) and legacy helpers.
*   `patches/`: XMX optimization patches (applied during build).

