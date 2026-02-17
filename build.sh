#!/bin/bash
set -e

echo "=== Intel Arc Battlemage (Xe2) Custom Backend Builder ==="
echo "Building optimized llama-server for B580/B570 GPUs..."

# 1. Build the OneAPI Docker image
echo "[1/4] Building Builder Image (This may take a while)..."
docker build -t llama-oneapi -f docker/Dockerfile.builder docker/

# 2. Check for source code or clone it
if [ ! -d "llama.cpp-source" ]; then
    echo "[2/4] Cloning llama.cpp..."
    git clone https://github.com/ggml-org/llama.cpp llama.cpp-source
else
    echo "[2/4] Using existing llama.cpp-source..."
fi

# 3. Apply XMX Patches
echo "[3/4] Applying XMX & Queue Latency Patches..."
cp patches/xmx_kernels_fixed.hpp llama.cpp-source/ggml/src/ggml-sycl/xmx_kernels.hpp
# (Add logic here to inject the include if not present - simplified for now)
if ! grep -q "xmx_kernels.hpp" llama.cpp-source/ggml/src/ggml-sycl/ggml-sycl.cpp; then
    sed -i '/#include "ggml-sycl\/sycl_hw.hpp"/a #include "ggml-sycl/xmx_kernels.hpp"' llama.cpp-source/ggml/src/ggml-sycl/ggml-sycl.cpp
    echo "Header injected."
fi

# 4. Build inside Container
echo "[4/4] Compiling inside Container..."
docker run --rm \
    -v $(pwd)/llama.cpp-source:/llama-source \
    -w /llama-source \
    llama-oneapi \
    bash -c "source /opt/intel/oneapi/setvars.sh && \
             cmake -B build -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx && \
             cmake --build build --config Release -j \$(nproc)"

echo "=== Build Complete! ==="
echo "Artifacts are in llama.cpp-source/build/bin/"
