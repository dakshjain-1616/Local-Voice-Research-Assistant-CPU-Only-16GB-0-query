"""
Pipeline orchestration module for the speech-to-speech pipeline.

Wires together STT (Parakeet TDT), LLM (Phi-4-mini-instruct),
TTS (Kokoro 82M), and VAD with a turn-taking state machine.
"""

from pipeline.state_machine import TurnState, TurnTakingStateMachine
from pipeline.orchestrator import PipelineOrchestrator

__all__ = [
    "TurnState",
    "TurnTakingStateMachine",
    "PipelineOrchestrator",
]