#!/usr/bin/env python3
"""
Speech-to-Speech Pipeline CLI

NVIDIA Parakeet TDT 0.6B (STT) -> Phi-4-mini-instruct 3.8B (LLM) -> Kokoro 82M (TTS)

CPU-only pipeline for 16GB RAM, 4-core machines.

Usage:
  # Start a live chat loop (requires microphone and speakers)
  python main.py chat

  # Transcribe a WAV file
  python main.py transcribe recording.wav

  # Say some text (synthesize speech to audio file)
  python main.py say "Hello, how can I help you?" -o output.wav

  # Text-only chat (no audio hardware needed)
  python main.py chat --text-only
"""

import argparse
import logging
import os
import sys
from typing import Optional

import yaml

# Ensure project root is in path
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ── Imports ──────────────────────────────────────────────────────

from stt.transcriber import ParakeetTDT
from llm.client import Phi4MiniClient
from tts.synthesizer import KokoroSynthesizer
from pipeline.orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)


# ── Config Loading ───────────────────────────────────────────────

def load_config(config_path: Optional[str] = None) -> dict:
    """Load YAML configuration."""
    if config_path is None:
        config_path = os.path.join(_project_root, "config.yaml")
    if not os.path.exists(config_path):
        logger.warning("Config file not found: %s, using defaults", config_path)
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


# ── CLI ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="Speech-to-Speech Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Run in text-only mode (no audio hardware needed)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file (default: ./config.yaml)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock implementations for all components (testing)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Chat subcommand
    chat_parser = subparsers.add_parser(
        "chat", help="Start a live conversation loop with VAD"
    )
    chat_parser.add_argument(
        "--mic-device",
        type=int,
        default=None,
        help="Input microphone device index",
    )
    chat_parser.add_argument(
        "--output-device",
        type=int,
        default=None,
        help="Output audio device index",
    )

    # Transcribe subcommand
    transcribe_parser = subparsers.add_parser(
        "transcribe", help="Transcribe a WAV file to text"
    )
    transcribe_parser.add_argument(
        "wavfile",
        type=str,
        help="Path to WAV file to transcribe",
    )

    # Say subcommand
    say_parser = subparsers.add_parser(
        "say", help="Synthesize text to speech (WAV file)"
    )
    say_parser.add_argument(
        "text",
        type=str,
        help="Text to synthesize",
    )
    say_parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output WAV file path (default: output.wav)",
    )

    return parser


def cmd_chat(args, config: dict, mock: bool) -> None:
    """Run the live chat loop."""
    cfg = config.get("pipeline", {})
    text_only = args.text_only or cfg.get("text_only", False)

    # Create components
    transcriber = _create_transcriber(config, mock)
    llm_client = _create_llm_client(config, mock)
    tts_synth = _create_tts_synthesizer(config, mock)

    orchestrator = PipelineOrchestrator(
        transcriber=transcriber,
        llm_client=llm_client,
        tts_synthesizer=tts_synth,
        text_only=text_only,
    )

    audio_cfg = config.get("audio", {})
    stt_sample_rate = audio_cfg.get("stt_sample_rate", 16000)
    tts_sample_rate = audio_cfg.get("tts_sample_rate", 24000)

    if text_only:
        _run_text_chat(orchestrator)
    else:
        _run_audio_chat(
            orchestrator,
            stt_sample_rate=stt_sample_rate,
            tts_sample_rate=tts_sample_rate,
            mic_device=args.mic_device,
            output_device=args.output_device,
        )


def _run_text_chat(orchestrator: PipelineOrchestrator) -> None:
    """Run a text-based interactive chat session."""
    print("\n=== Text Chat Mode ===")
    print("Type your message and press Enter. Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit", ""):
            print("Goodbye!")
            break

        print("Assistant: ", end="", flush=True)
        response_text = ""
        for audio_chunk, sr in orchestrator.process_text(user_input):
            pass  # In text-only mode, audio chunks are empty

        # Get the actual LLM response - we need to regenerate
        # For text-only, let's collect from the LLM directly
        print("(response would play as audio with --no-text-only)")
        print()


def _run_audio_chat(
    orchestrator: PipelineOrchestrator,
    stt_sample_rate: int,
    tts_sample_rate: int,
    mic_device: Optional[int] = None,
    output_device: Optional[int] = None,
) -> None:
    """Run an audio-based chat session with mic and speakers."""
    try:
        import sounddevice as sd
    except ImportError:
        print("ERROR: sounddevice is required for audio chat.")
        print("Install: pip install sounddevice")
        print("Or use --text-only for text-based chat.")
        sys.exit(1)

    print("\n=== Audio Chat Mode ===")
    print("Speak into your microphone. Press Ctrl+C to stop.\n")

    # For audio recording we use a threaded approach
    _audio_buffer: list[np.ndarray] = []
    _recording = False

    def audio_callback(indata, frames, time_info, status):
        """Callback for sounddevice InputStream."""
        nonlocal _audio_buffer
        if _recording:
            _audio_buffer.append(indata.copy())

    try:
        import numpy as np

        while True:
            print("Listening... (speak now)")
            _audio_buffer = []
            _recording = True

            # Record for a chunk (simplified: fixed duration chunks)
            duration = 5.0  # Record in 5-second chunks
            with sd.InputStream(
                samplerate=stt_sample_rate,
                channels=1,
                dtype="float32",
                device=mic_device,
                callback=audio_callback,
            ):
                sd.sleep(int(duration * 1000))

            _recording = False

            if not _audio_buffer:
                continue

            audio_data = np.concatenate(_audio_buffer).flatten()
            if len(audio_data) < stt_sample_rate * 0.5:  # Less than 0.5s
                print("(too short, ignoring)")
                continue

            print("Processing...")
            audio_chunks = []
            for audio_chunk, sr in orchestrator.process_audio(audio_data):
                audio_chunks.append(audio_chunk)

            if audio_chunks:
                full_audio = np.concatenate(audio_chunks)
                sd.play(full_audio, samplerate=tts_sample_rate, device=output_device)
                sd.wait()

    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as exc:
        logger.error("Audio chat error: %s", exc)
        print(f"Error: {exc}")


def cmd_transcribe(args, config: dict, mock: bool) -> None:
    """Transcribe a WAV file."""
    wav_path = args.wavfile
    if not os.path.exists(wav_path):
        print(f"ERROR: File not found: {wav_path}")
        sys.exit(1)

    transcriber = _create_transcriber(config, mock)

    try:
        text = transcriber.transcribe(wav_path)
        print(f"Transcription:")
        print(text)
    except Exception as exc:
        print(f"ERROR: Transcription failed: {exc}")
        sys.exit(1)


def cmd_say(args, config: dict, mock: bool) -> None:
    """Synthesize text to speech."""
    tts_synth = _create_tts_synthesizer(config, mock)

    output_path = args.output or "output.wav"

    try:
        audio, sr = tts_synth.synthesize_full(args.text)
        import soundfile as sf
        sf.write(output_path, audio, sr)
        print(f"Audio saved to: {output_path} ({len(audio)} samples @ {sr}Hz)")
    except Exception as exc:
        print(f"ERROR: TTS synthesis failed: {exc}")
        sys.exit(1)


# ── Factory Helpers ──────────────────────────────────────────────

def _create_transcriber(config: dict, mock: bool = False) -> ParakeetTDT:
    """Create STT transcriber from config."""
    models_cfg = config.get("models", {})
    audio_cfg = config.get("audio", {})
    threads_cfg = config.get("threads", {})

    return ParakeetTDT(
        model_id=models_cfg.get("stt", "nemo-parakeet-tdt-0.6b-v3"),
        use_mock=mock,
        sample_rate=audio_cfg.get("stt_sample_rate", 16000),
        num_threads=threads_cfg.get("stt", 1),
    )


def _create_llm_client(config: dict, mock: bool = False) -> Phi4MiniClient:
    """Create LLM client from config."""
    llm_server_cfg = config.get("llama_server", {})
    llm_cfg = config.get("llm", {})

    return Phi4MiniClient(
        base_url=llm_server_cfg.get("url", "http://127.0.0.1:8081/v1"),
        max_tokens=llm_cfg.get("max_tokens", 256),
        temperature=llm_cfg.get("temperature", 0.1),
        top_p=llm_cfg.get("top_p", 0.9),
        repeat_penalty=llm_cfg.get("repeat_penalty", 1.1),
        use_mock=mock,
        system_prompt=llm_cfg.get(
            "system_prompt",
            "You are a helpful research assistant. Answer concisely and accurately."
        ),
        model=llm_server_cfg.get("model", "phi-4-mini-instruct"),
    )


def _create_tts_synthesizer(config: dict, mock: bool = False) -> KokoroSynthesizer:
    """Create TTS synthesizer from config."""
    tts_cfg = config.get("tts", {})
    audio_cfg = config.get("audio", {})

    return KokoroSynthesizer(
        voice=tts_cfg.get("voice", "af_heart"),
        lang_code=tts_cfg.get("lang_code", "a"),
        use_mock=mock,
        sample_rate=audio_cfg.get("tts_sample_rate", 24000),
    )


# ── Main ─────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    config = load_config(args.config)

    # Determine mock mode
    mock = args.mock or config.get("pipeline", {}).get("mock", False)

    # Route command
    if args.command == "chat":
        cmd_chat(args, config, mock)
    elif args.command == "transcribe":
        cmd_transcribe(args, config, mock)
    elif args.command == "say":
        cmd_say(args, config, mock)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()