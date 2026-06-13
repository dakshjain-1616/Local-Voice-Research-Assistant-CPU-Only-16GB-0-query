"""
NVIDIA Parakeet TDT 0.6B STT transcriber.

Uses onnx-asr for ONNX INT8 CPU inference.
"""

import logging
import os
from typing import Optional, Union

import numpy as np

from stt import TranscribeResult

logger = logging.getLogger(__name__)

# Fake audio array for mock mode (1 second of silence at 16kHz)
_MOCK_AUDIO: np.ndarray = np.zeros(16000, dtype=np.float32)


class ParakeetTDT:
    """
    STT transcriber using NVIDIA Parakeet TDT 0.6B ONNX INT8 model.

    Two modes:
      - Real: uses onnx_asr.load_model() for actual inference.
      - Mock: returns canned transcription text for testing.
    """

    def __init__(
        self,
        model_id: str = "nemo-parakeet-tdt-0.6b-v3",
        use_mock: bool = False,
        sample_rate: int = 16000,
        num_threads: int = 1,
    ):
        """
        Args:
            model_id: HuggingFace/onnx-asr model identifier.
            use_mock: If True, return canned text instead of real inference.
            sample_rate: Expected input audio sample rate (Hz).
            num_threads: Number of CPU threads for ONNX runtime.
        """
        self.model_id = model_id
        self.use_mock = use_mock
        self.sample_rate = sample_rate
        self.num_threads = num_threads
        self._model = None

        if not use_mock:
            self._load_model()

    def _load_model(self) -> None:
        """Load the Parakeet TDT ONNX model via onnx-asr."""
        try:
            from onnx_asr import load_model
            logger.info("Loading Parakeet TDT model: %s", self.model_id)
            self._model = load_model(self.model_id)
            logger.info("STT model loaded successfully.")
        except ImportError:
            logger.warning(
                "onnx-asr not installed. Install with: pip install onnx-asr[cpu,hub]"
            )
            self._model = None
        except Exception as exc:
            logger.error("Failed to load STT model '%s': %s", self.model_id, exc)
            self._model = None

    def transcribe(
        self,
        audio_path_or_array: Optional[Union[str, np.ndarray]] = None,
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            audio_path_or_array: Path to a WAV file, or a numpy audio array
                                 (float32, 1D, 16kHz). Can be None in mock mode.

        Returns:
            Transcribed text string.

        Raises:
            RuntimeError: If real mode is requested but model could not be loaded.
            FileNotFoundError: If audio_path is a string path that doesn't exist.
        """
        if self.use_mock:
            return self._mock_transcribe()

        if self._model is None:
            raise RuntimeError(
                "STT model is not loaded. Check that onnx-asr is installed and "
                f"model '{self.model_id}' is available."
            )

        return self._real_transcribe(audio_path_or_array)

    def _mock_transcribe(self) -> str:
        """Return canned transcription for testing."""
        return "This is a mock transcription for testing purposes."

    def _real_transcribe(
        self,
        audio_path_or_array: Optional[Union[str, np.ndarray]],
    ) -> str:
        """
        Run real ONNX inference.
        """
        import soundfile as sf

        # Load audio
        if isinstance(audio_path_or_array, str):
            if not os.path.exists(audio_path_or_array):
                raise FileNotFoundError(f"Audio file not found: {audio_path_or_array}")
            audio, sr = sf.read(audio_path_or_array, dtype="float32")
            # Resample if needed
            if sr != self.sample_rate:
                audio = self._resample(audio, sr, self.sample_rate)
        elif isinstance(audio_path_or_array, np.ndarray):
            audio = audio_path_or_array
        else:
            raise TypeError(
                "audio_path_or_array must be a file path (str) or numpy array, "
                f"got {type(audio_path_or_array)}"
            )

        # Ensure 1D mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Run inference
        try:
            result = self._model(audio)
            text = result.text.strip() if hasattr(result, "text") else str(result).strip()
            return text if text else "[no speech detected]"
        except Exception as exc:
            logger.error("STT inference failed: %s", exc)
            raise RuntimeError(f"STT inference failed: {exc}") from exc

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple linear resampling."""
        if orig_sr == target_sr:
            return audio
        duration = len(audio) / orig_sr
        target_len = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    @property
    def is_loaded(self) -> bool:
        """Check if the real model is loaded."""
        return self._model is not None and not self.use_mock