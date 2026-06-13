# Local Voice Research Assistant — CPU-Only, 16GB, $0/query

A production-grade speech-to-speech pipeline that runs entirely on a consumer laptop. **No GPU. No cloud. No per-query cost.**

**The stack** (chosen by NEO MCP after benchmarking 6 models):
`Parakeet TDT 0.6B (INT8)` → `Phi-4-mini-instruct (Q4_K_M)` → `Kokoro 82M`

| Metric | Result |
|---|---|
| Transcription WER | 1.69% (clean) |
| Time-to-first-audio | ~1–4s (streaming) |
| RAM footprint | ~6 GB / 16 GB |
| Cost per query | $0 |
| Tests | 39/39 passing |

**Why not the "obvious" stack?** The performance favorite (Qwen3-4B w/ *thinking mode*) adds **1–8 min/query** on CPU. We chose latency realism over leaderboard scores. → **[Read the full case study](./speech_pipeline_benchmark_report.md)**

## Quick start
```bash
./scripts/download_models.sh
pip install -r requirements.txt
python main.py chat --text-only   # no audio hardware needed
```

## Documentation & case study

| Document | What it is |
|---|---|
| [`speech_pipeline_benchmark_report.md`](./speech_pipeline_benchmark_report.md) | The full 24-section benchmark & deployment case study (produced by NEO MCP) |
| [`BLOG.md`](./BLOG.md) | Engineering blog version — *"We Tried to Run a Production Voice Agent on a 16GB CPU. The Obvious Choice Lost."* |

> **Built by NEO MCP** — autonomous research → recommendation → self-revision → sensitivity analysis → tested code. See §22.5 of the report for the full process.

---

CPU-only voice assistant pipeline designed for **16GB RAM, 4-core CPU** machines. Based on the benchmark report: `speech_pipeline_benchmark_report.md` (Pipeline C Hybrid — the recommended winning configuration).

## Architecture

```
User Speech ──▶ VAD (Silero) ──▶ STT (Parakeet TDT 0.6B ONNX INT8)
                                            │
                                            ▼
                                     ┌──────────┐
                                     │  State   │
                                     │ Machine  │
                                     └──────────┘
                                            │
                                            ▼
                                   LLM (Phi-4-mini-instruct 3.8B)
                                            │
                                            ▼
                                  TTS (Kokoro 82M) ──▶ Audio Out
```

**Pipeline C Hybrid** was selected as the winning configuration because it delivers the best balance of:
- **STT accuracy**: Parakeet TDT's 1.69% WER (LibriSpeech clean) vs Moonshine's ~4.5%
- **LLM speed**: Phi-4-mini-instruct's ~10-12 tok/s (vs Qwen3's ~6-8 tok/s)
- **TTS quality**: Kokoro's UTMOS 4.49 (#1 on HF TTS Arena)

**Total estimated latency**: ~9-28s short query, ~18-45s long query (with streaming TTFA of ~1-4s).

## Prerequisites

### System Dependencies

```bash
# Kokoro requires espeak-ng for phonemization
sudo apt update
sudo apt install -y espeak-ng build-essential cmake git
```

### Python

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

See [requirements.txt](requirements.txt) for full dependency list.

### llama.cpp (LLM Server)

The Phi-4-mini-instruct model runs via llama.cpp's `llama-server` in OpenAI-compatible mode.

```bash
# Clone and build llama.cpp
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake .. -DLLAMA_CUBLAS=OFF  # CPU-only (no GPU)
cmake --build . --config Release -j 4
```

## Model Download

Run the download script, or download manually:

```bash
# Option 1: Automated script
bash scripts/download_models.sh

# Option 2: Download Phi-4-mini GGUF manually
huggingface-cli download bartowski/Phi-4-mini-instruct-GGUF \
    Phi-4-mini-instruct-Q4_K_M.gguf \
    --local-dir ./models \
    --local-dir-use-symlinks False

# Parakeet TDT will be downloaded automatically by onnx-asr on first use
# Or pre-download:
huggingface-cli download nvidia/parakeet-tdt-0.6b-v3 \
    --local-dir ./models/parakeet-tdt-0.6b-v3
```

### Model Files

| Model | File | Size | Source |
|-------|------|------|--------|
| Parakeet TDT 0.6B ONNX INT8 | (onnx-asr cache) | ~640 MB | nvidia/parakeet-tdt-0.6b-v3 |
| Phi-4-mini-instruct GGUF Q4_K_M | `models/Phi-4-mini-instruct-Q4_K_M.gguf` | ~2.3 GB | bartowski/Phi-4-mini-instruct-GGUF |
| Kokoro 82M | (PyPI package) | ~400-500 MB RAM | `pip install kokoro` |

**Total disk space needed**: ~3 GB
**Peak RAM usage**: ~4-5 GB (LLM ~2.8-3.2 GB + STT ~1.2-1.5 GB + TTS ~400-500 MB)

## Starting llama-server

After building llama.cpp and downloading the GGUF model:

```bash
cd /path/to/llama.cpp/build

./bin/llama-server \
    -m /path/to/tts_cpu/models/Phi-4-mini-instruct-Q4_K_M.gguf \
    --host 127.0.0.1 \
    --port 8081 \
    -ngl 0 \
    -t 2 \
    -c 4096 \
    --no-mmap
```

- **`-t 2`**: Uses 2 CPU threads (matches the 4-core allocation strategy)
- **`-c 4096`**: 4K context window (balance of memory vs. capacity)
- **`--no-mmap`**: Ensures stable memory usage
- The server listens on `http://127.0.0.1:8081/v1` (OpenAI-compatible endpoint)

## CLI Usage

### Chat (Live Microphone Loop)

```bash
# Audio chat (requires microphone and speakers)
python main.py chat

# Text-only chat (no audio hardware needed)
python main.py chat --text-only

# Chat with specific audio devices
python main.py chat --mic-device 0 --output-device 1
```

### Transcribe (WAV file → Text)

```bash
python main.py transcribe recording.wav
```

### Say (Text → WAV file)

```bash
# Default output: output.wav
python main.py say "Hello, how can I help you today?"

# Specify output file
python main.py say "Research shows that..." -o response.wav
```

### Testing with Mock Components

```bash
# Use mock implementations for all components (no models needed)
python main.py --mock chat --text-only
python main.py --mock transcribe test.wav
python main.py --mock say "Hello world" -o output.wav
```

### Options

| Flag | Description |
|------|-------------|
| `--text-only` | Skip audio playback (useful for testing without audio hardware) |
| `--mock` | Use mock implementations for all components |
| `--config PATH` | Path to custom config YAML |
| `--debug` | Enable debug logging |

## Configuration

Edit [config.yaml](config.yaml) to customize:

```yaml
# Thread allocation (4 cores total)
threads:
  llm: 2    # 2 cores for Phi-4-mini (llama.cpp)
  stt: 1    # 1 core for Parakeet TDT (ONNX runtime)
  tts: 1    # 1 core for Kokoro (TTS)

# Voice selection
tts:
  voice: "af_heart"    # American female, heart voice
  lang_code: "a"       # American English

# Sample rates
audio:
  stt_sample_rate: 16000   # Parakeet expects 16 kHz
  tts_sample_rate: 24000   # Kokoro outputs 24 kHz

# LLM defaults
llama_server:
  url: "http://127.0.0.1:8081/v1"
llm:
  max_tokens: 256
  temperature: 0.1
```

## Thread Allocation Strategy

Derived from the benchmark report (Section 17):

| Component | Threads | Rationale |
|-----------|---------|-----------|
| **LLM (llama.cpp)** | 2 | Most computationally intensive; benefits from parallel token generation |
| **STT (ONNX Runtime)** | 1 | Parakeet TDT INT8 is efficient (~0.03-0.05 RTF); single thread sufficient |
| **TTS (Kokoro)** | 1 | Kokoro has ~0.47 RTF on 4-core CPU; single thread handles real-time |
| **System/Other** | 0 | Reserved for OS and overhead |

**Total:** 4 threads, matching the 4-core CPU constraint.

## Hardware Notes

### Minimum Requirements

- **CPU**: 4 cores (x86_64)
- **RAM**: 16 GB (minimum; 8 GB may work but with significant swapping)
- **Disk**: 3 GB free for models
- **Audio**: Microphone and speakers/headphones for full-duplex chat

### Performance Expectations

| Metric | Estimate |
|--------|----------|
| Time to first audio (TTFA) | ~1-4 seconds (streaming) |
| Total short query (~3s speech) | ~9-28 seconds |
| Total long query (~15s speech) | ~18-45 seconds |
| STT latency | ~0.1-0.8 seconds |
| LLM generation speed | ~10-12 tokens/second |
| TTS synthesis | ~2x real-time (0.47 RTF) |

**Note**: True near-real-time (< 5s) is not achievable on CPU-only hardware. The LLM generation itself takes 8-30s minimum. However, TTS streaming makes the interaction feel conversational by starting audio playback after just the first sentence is synthesized.

### Memory Budget

| Component | RAM Usage | When Loaded |
|-----------|-----------|-------------|
| Parakeet TDT (INT8) | ~1.2-1.5 GB | During STT |
| Phi-4-mini (Q4_K_M) | ~2.8-3.2 GB | Always (llama-server) |
| Kokoro 82M | ~400-500 MB | During TTS |
| Silero VAD | ~50-100 MB | Always |
| Audio buffers + overhead | ~200-500 MB | During conversation |
| **Peak total** | **~5-6 GB** | During barge-in (STT + TTS overlap) |

## Text-Only Mode

Use `--text-only` when you want to test the pipeline without audio hardware:

```bash
# Chat in text-only mode
python main.py chat --text-only

# Use mock components (no models needed at all)
python main.py --mock chat --text-only
```

In text-only mode:
- No microphone capture
- No audio playback
- The LLM response is displayed as text
- Useful for development, testing, and environments without audio hardware

## Testing

```bash
# Run the full test suite
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/ -v -k "state_machine"
python -m pytest tests/ -v -k "barge"
python -m pytest tests/ -v -k "cli"
python -m pytest tests/ -v -k "mock"
```

The test suite uses mock components (no real models required).

## Project Structure

```
tts_cpu/
├── config.yaml              # Pipeline configuration
├── requirements.txt         # Python dependencies
├── main.py                  # CLI entry point
├── README.md                # This file
├── speech_pipeline_benchmark_report.md  # Full case study (24 sections)
├── BLOG.md                  # Engineering blog version
├── assets/blog-header.svg   # Blog header infographic
├── .gitignore               # Ignores venv/, models/, audio artifacts
├── models/                  # Downloaded model files (gitignored)
├── stt/
│   ├── __init__.py          # TranscribeResult namedtuple
│   └── transcriber.py       # ParakeetTDT class
├── llm/
│   ├── __init__.py
│   └── client.py            # Phi4MiniClient class
├── tts/
│   ├── __init__.py
│   └── synthesizer.py       # KokoroSynthesizer class
├── pipeline/
│   ├── __init__.py          # Exports key classes
│   ├── state_machine.py     # TurnTakingStateMachine
│   └── orchestrator.py      # PipelineOrchestrator
├── scripts/
│   └── download_models.sh   # Model download script
└── tests/
    ├── __init__.py
    └── test_pipeline.py     # Pytest test suite
```

## License

This project uses the following models with their respective licenses:
- **NVIDIA Parakeet TDT 0.6B**: CC-BY-4.0
- **Phi-4-mini-instruct**: MIT
- **Kokoro 82M**: Apache 2.0