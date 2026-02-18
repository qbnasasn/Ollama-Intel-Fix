import os
import json
import time
import signal
import sys
import subprocess
import threading
import logging
from flask import Flask, request, jsonify, Response, stream_with_context
import requests

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("OllamaProxy")

app = Flask(__name__)

# Configuration
OLLAMA_HOST = "0.0.0.0"
OLLAMA_PORT = 8080
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8081
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
MODELS_DIR = "/ollama_data/models"
BLOBS_DIR = os.path.join(MODELS_DIR, "blobs")
MANIFESTS_DIR = os.path.join(MODELS_DIR, "manifests")
SERVER_BINARY = "/app/llama-server"

# State
current_process = None
current_model_path = None
model_map = {}
lock = threading.Lock()

def load_model_map():
    """Scans manifests to build map of model_name -> blob_path"""
    global model_map
    new_map = {}
    logger.info(f"Scanning manifests in {MANIFESTS_DIR}...")
    
    # Expected structure: .../manifests/registry.ollama.ai/library/name/tag
    for root, dirs, files in os.walk(MANIFESTS_DIR):
        for file in files:
            manifest_path = os.path.join(root, file)
            try:
                rel_path = os.path.relpath(manifest_path, MANIFESTS_DIR)
                parts = rel_path.split(os.sep)
                # Example: ['registry.ollama.ai', 'library', 'phi4', 'latest']
                if len(parts) >= 3: 
                     # We assume standard registry layout
                     # registry = parts[0]
                     namespace = parts[-3] if len(parts) > 2 else "library"
                     name = parts[-2]
                     tag = parts[-1]
                     
                     # Construct colloquial names
                     full_name = f"{name}:{tag}"
                     if namespace != "library":
                         full_name = f"{namespace}/{full_name}"
                     
                     # Add 'latest' alias if applicable
                     if tag == "latest":
                         short_name = name
                         if namespace != "library":
                             short_name = f"{namespace}/{name}"
                         new_map[short_name] = manifest_path

                     new_map[full_name] = manifest_path

            except Exception as e:
                logger.error(f"Error parsing path {manifest_path}: {e}")

    # Parse manifests to find actual blobs
    final_map = {}
    short_name_counts = {}
    
    # First pass: Count short names to check uniqueness
    for name, path in new_map.items():
        # name is like "huihui_ai/phi4-abliterated:latest" or "phi4:latest"
        if "/" in name and ":" in name:
            short_model_name = name.split("/")[-1] # "phi4-abliterated:latest"
            base_name = short_model_name.split(":")[0] # "phi4-abliterated"
            short_name_counts[base_name] = short_name_counts.get(base_name, 0) + 1

    for name, path in new_map.items():
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                # Look for the 'model' layer (vnd.ollama.image.model)
                for layer in data.get('layers', []):
                    if layer.get('mediaType') == 'application/vnd.ollama.image.model':
                        digest = layer.get('digest')
                        if digest:
                            # Convert sha256:abc... -> sha256-abc...
                            blob_hash = digest.replace(":", "-")
                            blob_path = os.path.join(BLOBS_DIR, blob_hash)
                            if os.path.exists(blob_path):
                                final_map[name] = blob_path
                                
                                # Smart Alias: If unique non-library model, allow short name access
                                if "/" in name and ":" in name:
                                     short_model_name = name.split("/")[-1] # "phi4-abliterated:latest"
                                     base_name = short_model_name.split(":")[0]
                                     
                                     # If this base name is unique (count == 1), add alias
                                     if short_name_counts.get(base_name) == 1:
                                         final_map[short_model_name] = blob_path
                                         # Also alias the raw base name for resolution
                                         final_map[base_name] = blob_path
                                         
                            else:
                                logger.warning(f"Blob missing for {name}: {blob_path}")
                            break
        except Exception as e:
            logger.error(f"Error reading manifest {path}: {e}")
    
    with lock:
        model_map = final_map
    logger.info(f"Loaded {len(model_map)} models: {list(model_map.keys())}")

def kill_backend():
    global current_process
    if current_process:
        logger.info(f"Stopping backend PID {current_process.pid}...")
        current_process.terminate()
        try:
            current_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            current_process.kill()
        current_process = None

def start_backend(blob_path):
    global current_process, current_model_path
    
    with lock:
        if current_process and current_process.poll() is None:
            if current_model_path == blob_path:
                return True # Already running correct model
            kill_backend()
        
        logger.info(f"Starting backend with model: {blob_path}")
        
        # 1. Capture OneAPI Environment
        # We run a shell command to source setvars.sh and dump the environment
        env_cmd = "source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1 && env"
        try:
            # Run bash to get env vars
            output = subprocess.check_output(["bash", "-c", env_cmd], text=True)
            oneapi_env = {}
            for line in output.splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    oneapi_env[key] = value
            
            # Merge with current env
            final_env = os.environ.copy()
            final_env.update(oneapi_env)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to load OneAPI environment: {e}")
            return False

        # 2. Start llama-server directly
        # Note: Reduced context to 2048 to prevent OOM on 12GB VRAM
        cmd = [
            SERVER_BINARY,
            "-m", blob_path,
            "-c", "2048",
            "--host", BACKEND_HOST,
            "--port", str(BACKEND_PORT),
            "--n-gpu-layers", "99",
            "--ctx-size", "2048",
            "--batch-size", "256"
        ]
        
        # Auto-Split Logic:
        # 1. Explicit Env Var: SPLIT_MODE=true
        # 2. File Size > 11GB (Safety margin for 12GB VRAM)
        enable_split = False
        if os.environ.get("SPLIT_MODE", "false").lower() == "true":
            enable_split = True
            logger.info("Split Mode enabled via environment variable.")
        else:
            try:
                blob_size = os.path.getsize(blob_path)
                size_gb = blob_size / (1024**3)
                if size_gb > 11.0:
                    enable_split = True
                    logger.info(f"Model size {size_gb:.2f}GB > 11GB. Auto-enabling Split Mode.")
            except Exception as e:
                logger.warning(f"Could not check model size: {e}")

        if enable_split:
             cmd.extend(["--split-mode", "layer"])
        
        logger.info(f"Executing: {' '.join(cmd)}")
        
        # Start directly with the captured environment
        current_process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, env=final_env)
        current_model_path = blob_path
        
        # Wait for health check
        logger.info("Waiting for backend health...")
        for _ in range(300): # 300 seconds timeout
            try:
                requests.get(f"{BACKEND_URL}/health", timeout=0.5)
                logger.info("Backend is healthy.")
                return True
            except:
                time.sleep(1)
                if current_process.poll() is not None:
                    logger.error("Backend process exited prematurely.")
                    return False
        return False

@app.route('/api/tags', methods=['GET'])
def api_tags():
    load_model_map()
    models = []
    
    # 1. Collect all candidates (must have tag for Ollama compatibility)
    candidates = [k for k in model_map.keys() if ":" in k]
    
    # 2. Identify aliases to deduplicate
    short_names = set()
    for name in candidates:
        if "/" not in name:
            short_names.add(name)
            
    final_list = []
    for name in candidates:
        # If name has namespace
        if "/" in name:
            # Check if a corresponding short alias exists
            suffix = name.split("/")[-1]
            if suffix in short_names:
                continue # Skip this namespaced duplicate
        final_list.append(name)
    
    for name in final_list:
        models.append({
            "name": name,
            "modified_at": "2026-01-01T00:00:00Z",
            "size": 0,
            "digest": "sha256:custom",
            "details": {"format": "gguf", "family": "llama", "families": ["llama"], "parameter_size": "7B", "quantization_level": "Q4_0"}
        })
    return jsonify({"models": models})

@app.route('/api/version', methods=['GET'])
def api_version():
    return jsonify({"version": "0.5.4"})

def resolve_model(model_name):
    if not model_name: return None
    if model_name in model_map: return model_map[model_name]
    if f"{model_name}:latest" in model_map: return model_map[f"{model_name}:latest"]
    return None

def proxy_request(path):
    # Check if backend is running
    with lock:
        if not current_process or current_process.poll() is not None:
            # Try to start default model (Phi-4) if nothing running
            # Resolving phi4
            phi4_path = resolve_model("phi4") or resolve_model("phi4:latest")
            if phi4_path:
                start_backend(phi4_path)
    
    if not current_process:
        return jsonify({"error": "No model loaded and no default found"}), 503

    resp = requests.request(
        method=request.method,
        url=f"{BACKEND_URL}{path}",
        headers={k:v for k,v in request.headers if k.lower() != 'host'},
        data=request.get_data(),
        cookies=request.cookies,
        allow_redirects=False,
        stream=True
    )
    
    headers = [(k,v) for k,v in resp.headers.items()]
    return Response(stream_with_context(resp.iter_content(chunk_size=4096)), status=resp.status_code, headers=headers)

@app.route('/v1/chat/completions', methods=['POST'])
def v1_chat():
    data = request.json
    model_name = data.get('model')
    blob_path = resolve_model(model_name)
    
    if blob_path:
        if not start_backend(blob_path):
             return jsonify({"error": f"Failed to load model {model_name}"}), 500
    elif model_name:
        return jsonify({"error": f"Model {model_name} not found"}), 404

    return proxy_request("/v1/chat/completions")

@app.route('/v1/models', methods=['GET'])
def v1_models():
    # OpenAI compatible models list
    load_model_map()
    # Filter using same logic as api_tags? For consistency, probably.
    # But for now just dump everything or apply similar filter.
    # Let's apply similar filter for consistency.
    
    candidates = [k for k in model_map.keys() if ":" in k]
    short_names = set(n for n in candidates if "/" not in n)
    final_list = []
    for name in candidates:
        if "/" in name:
            suffix = name.split("/")[-1]
            if suffix in short_names:
                continue
        final_list.append(name)

    data = [{"id": name, "object": "model", "created": 1677610602, "owned_by": "ollama"} for name in final_list]
    return jsonify({"object": "list", "data": data})

@app.route('/health', methods=['GET'])
def health():
    return proxy_request("/health")

if __name__ == '__main__':
    load_model_map()
    # Pre-load Phi-4 if available
    phi4 = resolve_model("phi4") or resolve_model("phi4:latest")
    if phi4:
        start_backend(phi4)
    app.run(host=OLLAMA_HOST, port=OLLAMA_PORT, threaded=True)
