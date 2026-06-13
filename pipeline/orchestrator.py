"""
Pipeline orchestrator that wires STT + LLM + TTS + VAD together.

Manages the full speech-to-speech pipeline:
  1. VAD endpointing -> detect speech boundaries
  2. STT -> transcribe audio to text
  3. LLM -> generate response text (streaming)
  4. TTS -> sentence-chunked audio synthesis (streaming)
  5. Barge-in support (SPEAKING -> LISTENING)
"""

import logging
import queue
import threading
import time
from typing import Callable, Generator, Optional, Tuple

import numpy as np

from pipeline.state_machine import TurnState, TurnTakingStateMachine
from stt.transcriber import ParakeetTDT
from llm.client import Phi4MiniClient
from tts.synthesizer import KokoroSynthesizer

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Orchestrates the STT -> LLM -> TTS pipeline with VAD and turn-taking.

    Supports:
      - Full pipeline: audio input -> transcribed text -> LLM response -> TTS audio
      - Text-only mode: skip audio I/O (for testing without mic/speakers)
      - Barge-in: user speech during playback triggers re-listening
    """

    def __init__(
        self,
        transcriber: ParakeetTDT,
        llm_client: Phi4MiniClient,
        tts_synthesizer: KokoroSynthesizer,
        vad_threshold: float = 0.5,
        min_silence_ms: int = 500,
        sample_rate_stt: int = 16000,
        sample_rate_tts: int = 24000,
        text_only: bool = False,
        on_speech_during_playback: Optional[Callable] = None,
    ):
        """
        Args:
            transcriber: ParakeetTDT instance.
            llm_client: Phi4MiniClient instance.
            tts_synthesizer: KokoroSynthesizer instance.
            vad_threshold: Silero VAD confidence threshold.
            min_silence_ms: Silence duration (ms) to trigger endpointing.
            sample_rate_stt: STT model input sample rate.
            sample_rate_tts: TTS model output sample rate.
            text_only: If True, skip TTS and return text instead of audio.
            on_speech_during_playback: Callback invoked when barge-in occurs.
        """
        self.transcriber = transcriber
        self.llm_client = llm_client
        self.tts_synthesizer = tts_synthesizer
        self.vad_threshold = vad_threshold
        self.min_silence_ms = min_silence_ms
        self.sample_rate_stt = sample_rate_stt
        self.sample_rate_tts = sample_rate_tts
        self.text_only = text_only
        self.on_speech_during_playback = on_speech_during_playback

        self.state_machine = TurnTakingStateMachine()
        self._barge_in_requested = False
        self._stop_tts = threading.Event()

        # Register barge-in handler
        self.state_machine.on_enter(TurnState.LISTENING, self._on_listening_started)

    def _on_listening_started(self, from_state: TurnState, to_state: TurnState) -> None:
        """Called when entering LISTENING state (including via barge-in)."""
        self._stop_tts.set()
        if from_state == TurnState.SPEAKING:
            logger.info("Barge-in detected: interrupting TTS playback")
            if self.on_speech_during_playback:
                self.on_speech_during_playback()

    def request_barge_in(self) -> None:
        """
        Request a barge-in (interrupt TTS and start listening).

        Called externally when VAD detects user speech during playback.
        """
        if self.state_machine.is_speaking():
            logger.debug("Barge-in requested during SPEAKING state")
            self._stop_tts.set()
            try:
                self.state_machine.transition(TurnState.LISTENING)
            except ValueError:
                logger.warning("Cannot barge-in from state: %s", self.state_machine.state)

    def process_audio(
        self,
        audio_data: np.ndarray,
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """
        Process audio through the full STT -> LLM -> TTS pipeline.

        Args:
            audio_data: Input audio as float32 numpy array (16kHz mono).

        Yields:
            (audio_chunk, sample_rate) tuples for streaming TTS output.
            In text-only mode, yields (empty_audio, sample_rate) and the
            caller should use process_text() instead for text output.

        Raises:
            RuntimeError: If pipeline processing fails at any stage.
        """
        self.state_machine.transition(TurnState.LISTENING)

        try:
            # ---- STAGE 1: VAD + STT ----
            text = self._transcribe_audio(audio_data)
            if not text or text == "[no speech detected]":
                logger.info("No speech detected in audio")
                self.state_machine.transition(TurnState.IDLE)
                return

            # ---- STAGE 2: LLM ----
            self.state_machine.transition(TurnState.PROCESSING)
            llm_text = self._generate_response(text)
            if not llm_text:
                logger.info("Empty LLM response")
                self.state_machine.transition(TurnState.IDLE)
                return

            # ---- STAGE 3: TTS (streaming) ----
            if self.text_only:
                # In text-only mode, yield empty audio but make text available
                # via the process_text() method instead
                logger.info("Text-only mode: skipping TTS. Response: %s", llm_text[:100])
                self.state_machine.transition(TurnState.IDLE)
                return

            self.state_machine.transition(TurnState.SPEAKING)
            yield from self._synthesize_speech_streaming(llm_text)

            self.state_machine.transition(TurnState.IDLE)

        except Exception as exc:
            logger.error("Pipeline processing failed: %s", exc)
            self.state_machine.reset()
            raise RuntimeError(f"Pipeline processing failed: {exc}") from exc

    def process_text(
        self,
        text: str,
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """
        Process text directly through LLM -> TTS (skip STT).

        Useful for text-only mode or when text input is provided directly.

        Args:
            text: Input text to process.

        Yields:
            (audio_chunk, sample_rate) tuples for streaming output.
            In text-only mode, yields a single empty chunk.
        """
        self.state_machine.transition(TurnState.PROCESSING)

        try:
            llm_text = self._generate_response(text)
            if not llm_text:
                logger.info("Empty LLM response")
                self.state_machine.transition(TurnState.IDLE)
                return

            if self.text_only:
                logger.info("Text-only mode: Response: %s", llm_text[:100])
                yield (np.zeros(0, dtype=np.float32), self.sample_rate_tts)
                self.state_machine.transition(TurnState.IDLE)
                return

            self.state_machine.transition(TurnState.SPEAKING)
            yield from self._synthesize_speech_streaming(llm_text)
            self.state_machine.transition(TurnState.IDLE)

        except Exception as exc:
            logger.error("Text processing failed: %s", exc)
            self.state_machine.reset()
            raise RuntimeError(f"Text processing failed: {exc}") from exc

    def _transcribe_audio(self, audio_data: np.ndarray) -> str:
        """Run STT on audio data."""
        logger.debug("Transcribing audio (%d samples)", len(audio_data))
        text = self.transcriber.transcribe(audio_data)
        logger.info("STT result: %s", text[:100] if text else "(empty)")
        return text

    def _generate_response(self, text: str) -> str:
        """Generate LLM response for given text."""
        messages = [{"role": "user", "content": text}]
        logger.debug("Generating LLM response...")
        chunks = list(self.llm_client.chat(messages, stream=True))
        response = "".join(chunks).strip()
        logger.info("LLM response (%d chars): %s", len(response), response[:100])
        return response

    def _synthesize_speech_streaming(
        self,
        text: str,
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """
        Synthesize speech with barge-in support.

        Stops early if barge-in is requested.
        """
        self._stop_tts.clear()

        for audio_chunk, sr in self.tts_synthesizer.synthesize_streaming(text):
            if self._stop_tts.is_set():
                logger.debug("TTS stopped due to barge-in")
                break
            yield (audio_chunk, sr)

    def get_conversation_state(self) -> str:
        """Return a human-readable description of the current state."""
        return str(self.state_machine.state)

    def reset(self) -> None:
        """Reset the pipeline to IDLE state."""
        self.state_machine.reset()
        self._stop_tts.clear()
        self._barge_in_requested = False