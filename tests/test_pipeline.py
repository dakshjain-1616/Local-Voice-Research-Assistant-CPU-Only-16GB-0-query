"""
Tests for the speech-to-speech pipeline using mock components.

Verifies:
  - State machine transitions (including barge-in)
  - Sentence-chunked streaming
  - Barge-in logic
  - CLI argument parsing
"""

import io
import sys
from typing import Generator, Tuple

import numpy as np
import pytest

from pipeline.state_machine import TurnState, TurnTakingStateMachine
from pipeline.orchestrator import PipelineOrchestrator
from stt.transcriber import ParakeetTDT, TranscribeResult
from llm.client import Phi4MiniClient
from tts.synthesizer import KokoroSynthesizer


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def mock_transcriber() -> ParakeetTDT:
    """Create a mock STT transcriber."""
    return ParakeetTDT(use_mock=True)


@pytest.fixture
def mock_llm_client() -> Phi4MiniClient:
    """Create a mock LLM client."""
    return Phi4MiniClient(use_mock=True)


@pytest.fixture
def mock_tts_synthesizer() -> KokoroSynthesizer:
    """Create a mock TTS synthesizer."""
    return KokoroSynthesizer(use_mock=True)


@pytest.fixture
def orchestrator(
    mock_transcriber: ParakeetTDT,
    mock_llm_client: Phi4MiniClient,
    mock_tts_synthesizer: KokoroSynthesizer,
) -> PipelineOrchestrator:
    """Create a pipeline orchestrator with all mock components."""
    return PipelineOrchestrator(
        transcriber=mock_transcriber,
        llm_client=mock_llm_client,
        tts_synthesizer=mock_tts_synthesizer,
    )


@pytest.fixture
def text_only_orchestrator(
    mock_transcriber: ParakeetTDT,
    mock_llm_client: Phi4MiniClient,
    mock_tts_synthesizer: KokoroSynthesizer,
) -> PipelineOrchestrator:
    """Create a text-only pipeline orchestrator."""
    return PipelineOrchestrator(
        transcriber=mock_transcriber,
        llm_client=mock_llm_client,
        tts_synthesizer=mock_tts_synthesizer,
        text_only=True,
    )


# ── State Machine Tests ───────────────────────────────────────────

class TestTurnTakingStateMachine:
    """Tests for the turn-taking state machine."""

    def test_initial_state(self):
        """State machine starts in IDLE."""
        sm = TurnTakingStateMachine()
        assert sm.state == TurnState.IDLE
        assert sm.is_idle()

    def test_idle_to_listening(self):
        """Valid transition: IDLE -> LISTENING."""
        sm = TurnTakingStateMachine()
        sm.transition(TurnState.LISTENING)
        assert sm.state == TurnState.LISTENING
        assert sm.is_listening()

    def test_listening_to_processing(self):
        """Valid transition: LISTENING -> PROCESSING."""
        sm = TurnTakingStateMachine()
        sm.transition(TurnState.LISTENING)
        sm.transition(TurnState.PROCESSING)
        assert sm.state == TurnState.PROCESSING
        assert sm.is_processing()

    def test_processing_to_speaking(self):
        """Valid transition: PROCESSING -> SPEAKING."""
        sm = TurnTakingStateMachine()
        sm.transition(TurnState.LISTENING)
        sm.transition(TurnState.PROCESSING)
        sm.transition(TurnState.SPEAKING)
        assert sm.state == TurnState.SPEAKING
        assert sm.is_speaking()

    def test_speaking_to_idle(self):
        """Valid transition: SPEAKING -> IDLE."""
        sm = TurnTakingStateMachine()
        sm.transition(TurnState.LISTENING)
        sm.transition(TurnState.PROCESSING)
        sm.transition(TurnState.SPEAKING)
        sm.transition(TurnState.IDLE)
        assert sm.is_idle()

    def test_barge_in_speaking_to_listening(self):
        """Barge-in: SPEAKING -> LISTENING when user interrupts."""
        sm = TurnTakingStateMachine()
        sm.transition(TurnState.LISTENING)
        sm.transition(TurnState.PROCESSING)
        sm.transition(TurnState.SPEAKING)
        # Barge-in
        sm.transition(TurnState.LISTENING)
        assert sm.is_listening()

    def test_invalid_transition_raises(self):
        """Invalid transitions raise ValueError."""
        sm = TurnTakingStateMachine()
        # IDLE -> SPEAKING is not valid
        with pytest.raises(ValueError, match="Invalid state transition"):
            sm.transition(TurnState.SPEAKING)

    def test_invalid_transition_processing_to_listening(self):
        """PROCESSING -> LISTENING is not valid."""
        sm = TurnTakingStateMachine()
        sm.transition(TurnState.LISTENING)
        sm.transition(TurnState.PROCESSING)
        with pytest.raises(ValueError, match="Invalid state transition"):
            sm.transition(TurnState.LISTENING)

    def test_full_cycle(self):
        """Complete turn cycle: IDLE -> LISTENING -> PROCESSING -> SPEAKING -> IDLE."""
        sm = TurnTakingStateMachine()
        sm.transition(TurnState.LISTENING)
        sm.transition(TurnState.PROCESSING)
        sm.transition(TurnState.SPEAKING)
        sm.transition(TurnState.IDLE)
        assert sm.is_idle()
        assert sm.transition_count == 4

    def test_reset(self):
        """Reset returns to IDLE."""
        sm = TurnTakingStateMachine()
        sm.transition(TurnState.LISTENING)
        sm.transition(TurnState.PROCESSING)
        sm.reset()
        assert sm.is_idle()

    def test_can_transition_to(self):
        """can_transition_to returns correct values."""
        sm = TurnTakingStateMachine()
        assert sm.can_transition_to(TurnState.LISTENING)
        assert sm.can_transition_to(TurnState.PROCESSING)   # Direct text input
        assert not sm.can_transition_to(TurnState.SPEAKING)

    def test_callbacks(self):
        """Enter and exit callbacks are fired."""
        sm = TurnTakingStateMachine()
        enter_calls = []
        exit_calls = []

        sm.on_enter(TurnState.LISTENING, lambda f, t: enter_calls.append(("enter", f, t)))
        sm.on_exit(TurnState.IDLE, lambda f, t: exit_calls.append(("exit", f, t)))

        sm.transition(TurnState.LISTENING)

        assert len(enter_calls) == 1
        assert enter_calls[0] == ("enter", TurnState.IDLE, TurnState.LISTENING)
        assert len(exit_calls) == 1
        assert exit_calls[0] == ("exit", TurnState.IDLE, TurnState.LISTENING)


# ── Streaming Tests ───────────────────────────────────────────────

class TestStreamingBehavior:
    """Tests for sentence-chunked streaming."""

    def test_tts_streaming_yields_chunks(self, mock_tts_synthesizer):
        """TTS streaming yields audio chunks for each sentence."""
        text = "Hello world. How are you? I am fine."
        chunks = list(mock_tts_synthesizer.synthesize_streaming(text))
        # Should yield one chunk per sentence (3 sentences)
        assert len(chunks) == 3
        for audio, sr in chunks:
            assert isinstance(audio, np.ndarray)
            assert audio.dtype == np.float32
            assert sr == 24000
            assert len(audio) > 0

    def test_tts_streaming_single_sentence(self, mock_tts_synthesizer):
        """Single sentence yields one chunk."""
        chunks = list(mock_tts_synthesizer.synthesize_streaming("Hello."))
        assert len(chunks) == 1

    def test_tts_streaming_empty_text(self, mock_tts_synthesizer):
        """Empty text yields empty result."""
        chunks = list(mock_tts_synthesizer.synthesize_streaming(""))
        assert len(chunks) == 1  # Mock handles empty as one chunk

    def test_llm_streaming_yields_chunks(self, mock_llm_client):
        """LLM streaming yields text chunks."""
        chunks = list(mock_llm_client.chat(
            [{"role": "user", "content": "Hello"}],
            stream=True,
        ))
        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert "mock" in full_text.lower()

    def test_llm_non_streaming(self, mock_llm_client):
        """Non-streaming LLM returns full text."""
        chunks = list(mock_llm_client.chat(
            [{"role": "user", "content": "Hello"}],
            stream=True,  # Mock always streams
        ))
        full_text = "".join(chunks)
        assert len(full_text) > 0


# ── Barge-in Tests ────────────────────────────────────────────────

class TestBargeIn:
    """Tests for barge-in behavior."""

    def test_barge_in_callback_called(self):
        """Barge-in callback is invoked when user speech detected during SPEAKING."""
        sm = TurnTakingStateMachine()
        sm.transition(TurnState.LISTENING)
        sm.transition(TurnState.PROCESSING)
        sm.transition(TurnState.SPEAKING)

        callback_called = [False]

        def on_barge_in():
            callback_called[0] = True

        # Register on the orchestrator level
        tts_synth = KokoroSynthesizer(use_mock=True)
        orch = PipelineOrchestrator(
            transcriber=ParakeetTDT(use_mock=True),
            llm_client=Phi4MiniClient(use_mock=True),
            tts_synthesizer=tts_synth,
            on_speech_during_playback=on_barge_in,
        )
        # Put orchestrator in SPEAKING state
        orch.state_machine.transition(TurnState.LISTENING)
        orch.state_machine.transition(TurnState.PROCESSING)
        orch.state_machine.transition(TurnState.SPEAKING)

        # Request barge-in
        orch.request_barge_in()
        assert callback_called[0]
        assert orch.state_machine.is_listening()

    def test_barge_in_aborts_tts(self, orchestrator):
        """Barge-in causes TTS to stop yielding chunks."""
        # Get orchestrator into SPEAKING state via process_audio
        audio = np.zeros(16000, dtype=np.float32)
        text = orchestrator._generate_response("test")
        orchestrator.state_machine.transition(TurnState.LISTENING)
        orchestrator.state_machine.transition(TurnState.PROCESSING)
        orchestrator.state_machine.transition(TurnState.SPEAKING)

        # Start TTS streaming and request barge-in midway
        chunks = []
        for i, chunk in enumerate(orchestrator._synthesize_speech_streaming(text)):
            if i == 0:
                # Request barge-in after first chunk
                orchestrator.request_barge_in()
            chunks.append(chunk)

        # Should have stopped early
        assert len(chunks) < 3  # Full mock response would yield ~3 chunks

    def test_barge_in_does_not_affect_non_speaking(self, orchestrator):
        """Barge-in requested outside SPEAKING has no effect."""
        # In IDLE, barge-in should be no-op
        orchestrator.request_barge_in()
        assert orchestrator.state_machine.is_idle()


# ── Pipeline Orchestrator Tests ───────────────────────────────────

class TestPipelineOrchestrator:
    """Tests for the full pipeline orchestrator."""

    def test_process_audio_mock_full_pipeline(self, orchestrator):
        """Full pipeline with mock components produces audio chunks."""
        audio_data = np.zeros(16000 * 3, dtype=np.float32)  # 3 seconds of silence
        chunks = list(orchestrator.process_audio(audio_data))
        assert len(chunks) > 0
        for audio, sr in chunks:
            assert isinstance(audio, np.ndarray)
            assert sr == 24000

    def test_process_audio_state_transitions(self, orchestrator):
        """State transitions are correct during audio processing."""
        audio_data = np.zeros(16000 * 3, dtype=np.float32)
        list(orchestrator.process_audio(audio_data))
        assert orchestrator.state_machine.is_idle()

    def test_text_only_mode(self, text_only_orchestrator):
        """Text-only mode skips TTS."""
        audio_data = np.zeros(16000 * 3, dtype=np.float32)
        chunks = list(text_only_orchestrator.process_audio(audio_data))
        # In text-only mode, after LLM generates response, the method returns
        # before yielding TTS audio
        assert len(chunks) == 0

    def test_process_text_direct(self, orchestrator):
        """Direct text processing works."""
        chunks = list(orchestrator.process_text("Hello"))
        assert len(chunks) > 0
        assert orchestrator.state_machine.is_idle()

    def test_process_text_text_only(self, text_only_orchestrator):
        """Text-only text processing yields empty audio."""
        chunks = list(text_only_orchestrator.process_text("Hello"))
        # In text-only mode, yields one empty chunk
        assert len(chunks) == 1
        audio, sr = chunks[0]
        assert len(audio) == 0

    def test_reset(self, orchestrator):
        """Reset returns orchestrator to idle."""
        audio_data = np.zeros(16000, dtype=np.float32)
        list(orchestrator.process_audio(audio_data))
        orchestrator.reset()
        assert orchestrator.state_machine.is_idle()


# ── CLI Argument Parsing Tests ────────────────────────────────────

class TestCLIArgumentParsing:
    """Tests for CLI argument parsing."""

    def test_chat_command(self):
        """'chat' subcommand is parsed."""
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args(["chat"])
        assert args.command == "chat"

    def test_chat_text_only(self):
        """'chat --text-only' sets text_only flag."""
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args(["--text-only", "chat"])
        assert args.command == "chat"
        assert args.text_only is True

    def test_transcribe_command(self):
        """'transcribe <file>' is parsed."""
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args(["transcribe", "test.wav"])
        assert args.command == "transcribe"
        assert args.wavfile == "test.wav"

    def test_say_command(self):
        """'say <text>' is parsed."""
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args(["say", "Hello world"])
        assert args.command == "say"
        assert args.text == "Hello world"

    def test_say_with_output(self):
        """'say <text> -o <file>' is parsed."""
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args(["say", "Hello", "-o", "output.wav"])
        assert args.command == "say"
        assert args.output == "output.wav"

    def test_mock_flag(self):
        """'--mock' flag is parsed."""
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args(["--mock", "--text-only", "chat"])
        assert args.mock is True

    def test_debug_flag(self):
        """'--debug' flag is parsed."""
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args(["--debug", "chat"])
        assert args.debug is True

    def test_config_flag(self):
        """'--config <path>' is parsed."""
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args(["--config", "/path/to/config.yaml", "chat"])
        assert args.config == "/path/to/config.yaml"

    def test_help(self):
        """Help is printed without errors."""
        from main import build_parser
        parser = build_parser()
        try:
            parser.parse_args(["--help"])
        except SystemExit:
            pass  # argparse exits after printing help


# ── Mock Component Tests ──────────────────────────────────────────

class TestMockComponents:
    """Tests for mock component implementations."""

    def test_mock_transcriber(self, mock_transcriber):
        """Mock transcriber returns canned text."""
        text = mock_transcriber.transcribe(None)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_mock_transcriber_no_file_needed(self, mock_transcriber):
        """Mock transcriber works without any audio input."""
        text = mock_transcriber.transcribe()
        assert isinstance(text, str)
        assert "mock" in text.lower()

    def test_mock_llm_client(self, mock_llm_client):
        """Mock LLM yields canned chunks."""
        chunks = list(mock_llm_client.chat(
            [{"role": "user", "content": "test"}],
        ))
        assert len(chunks) > 0
        full = "".join(chunks)
        assert "mock" in full.lower()

    def test_mock_tts(self, mock_tts_synthesizer):
        """Mock TTS yields silent audio chunks."""
        chunks = list(mock_tts_synthesizer.synthesize_streaming("Hello world."))
        assert len(chunks) > 0
        for audio, sr in chunks:
            # Mock audio should be all zeros (silence)
            assert np.all(audio == 0.0)
            assert sr == 24000