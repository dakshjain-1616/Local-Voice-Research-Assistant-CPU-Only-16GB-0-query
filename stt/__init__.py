"""
Speech-to-Text (STT) module using NVIDIA Parakeet TDT 0.6B (ONNX INT8).
"""

from typing import NamedTuple


class TranscribeResult(NamedTuple):
    """Result from a transcription call."""
    text: str
    confidence: float = 0.0
    duration_seconds: float = 0.0


__all__ = ["TranscribeResult"]