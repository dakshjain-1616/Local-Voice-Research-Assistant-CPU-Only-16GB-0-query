#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Download models for the speech-to-speech pipeline
# NVIDIA Parakeet TDT 0.6B (ONNX INT8) + Phi-4-mini-instruct 3.8B (GGUF Q4_K_M)
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"

echo "=== Speech-to-Speech Pipeline: Model Download ==="
echo "Project: $PROJECT_DIR"
echo "Models:  $MODELS_DIR"
echo ""

mkdir -p "$MODELS_DIR"

# ── Helper: check if huggingface-cli is available ─────────
check_hf_cli() {
    if ! command -v huggingface-cli &>/dev/null; then
        echo "ERROR: huggingface-cli not found."
        echo "Install with: pip install huggingface-hub"
        exit 1
    fi
}

# ── 1. Parakeet TDT 0.6B (via onnx-asr cache) ────────────
download_parakeet() {
    echo ""
    echo "--- Step 1: Parakeet TDT 0.6B (ONNX INT8) ---"
    echo ""
    echo "Parakeet TDT is managed by the onnx-asr library."
    echo "It will be downloaded automatically on first use when you run:"
    echo ""
    echo "  /usr/bin/python3 -c \"from onnx_asr import load_model; load_model('nemo-parakeet-tdt-0.6b-v3')\""
    echo ""
    echo "To pre-download it now:"
    echo "  huggingface-cli download nvidia/parakeet-tdt-0.6b-v3 --local-dir \"$MODELS_DIR/parakeet-tdt-0.6b-v3\""
    echo ""
    echo "Or let onnx-asr handle it automatically at runtime."
}

# ── 2. Phi-4-mini-instruct GGUF ───────────────────────────
download_phi4() {
    echo ""
    echo "--- Step 2: Phi-4-mini-instruct 3.8B (GGUF Q4_K_M) ---"
    echo ""
    local GGUF_REPO="bartowski/Phi-4-mini-instruct-GGUF"
    local GGUF_FILE="Phi-4-mini-instruct-Q4_K_M.gguf"
    local GGUF_PATH="$MODELS_DIR/$GGUF_FILE"

    if [ -f "$GGUF_PATH" ]; then
        echo "Model already exists at: $GGUF_PATH"
        ls -lh "$GGUF_PATH"
        return
    fi

    echo "Downloading $GGUF_FILE from $GGUF_REPO ..."
    echo "Size: ~2.3 GB"
    echo ""

    check_hf_cli

    huggingface-cli download \
        "$GGUF_REPO" \
        "$GGUF_FILE" \
        --local-dir "$MODELS_DIR" \
        --local-dir-use-symlinks False

    echo ""
    echo "Downloaded: $GGUF_PATH"
    ls -lh "$GGUF_PATH"
}

# ── 3. llama-server instructions ──────────────────────────
print_llama_server_instructions() {
    echo ""
    echo "--- Step 3: Running llama-server ---"
    echo ""
    echo "After downloading the GGUF model, start llama-server:"
    echo ""
    echo "  # Build llama.cpp (if not already done):"
    echo "  git clone https://github.com/ggml-org/llama.cpp"
    echo "  cd llama.cpp"
    echo "  mkdir build && cd build"
    echo "  cmake .. -DLLAMA_CUBLAS=OFF  # CPU-only"
    echo "  cmake --build . --config Release -j 4"
    echo ""
    echo "  # Start server (OpenAI-compatible endpoint):"
    echo "  ./bin/llama-server \\"
    echo "    -m \"$MODELS_DIR/$GGUF_FILE\" \\"
    echo "    --host 127.0.0.1 \\"
    echo "    --port 8081 \\"
    echo "    -ngl 0 \\"
    echo "    -t 2 \\"
    echo "    -c 4096 \\"
    echo "    --no-mmap"
    echo ""
    echo "The pipeline connects to http://127.0.0.1:8081/v1"
}

# ── Main ──────────────────────────────────────────────────
main() {
    download_parakeet
    download_phi4
    print_llama_server_instructions

    echo ""
    echo "=== Download complete ==="
    echo ""
    echo "Model files location: $MODELS_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Install Python dependencies: pip install -r requirements.txt"
    echo "  2. Install espeak-ng: sudo apt install espeak-ng"
    echo "  3. Build and start llama-server (see instructions above)"
    echo "  4. Run the pipeline: /usr/bin/python3 main.py chat --text-only"
}

main "$@"
