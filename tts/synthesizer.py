"""
Kokoro 82M TTS synthesizer with sentence-chunked streaming.

Uses the `kokoro` PyPI package (KPipeline) for fast CPU-friendly TTS.
Supports sentence-level streaming: yields audio for each sentence as soon
as it is synthesized, rather than waiting for the full text.
"""

import logging
import re
import time
from typing import Generator, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Regex to split text into sentences (handles common English sentence endings)
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


class KokoroSynthesizer:
    """
    TTS synthesizer using Kokoro 82M (via `kokoro` PyPI package).

    Two modes:
      - Real: loads KPipeline and runs actual synthesis.
      - Mock: yields short silent audio chunks for testing.

    Streaming: sentences are synthesized and yielded one at a time
    to minimize Time-To-First-Audio (TTFA).
    """

    def __init__(
        self,
        voice: str = "af_heart",
        lang_code: str = "a",
        use_mock: bool = False,
        sample_rate: int = 24000,
    ):
        """
        Args:
            voice: Kokoro voice identifier (e.g., "af_heart", "am_midwest").
            lang_code: Language code ("a" = American English, "b" = British, etc.).
            use_mock: If True, yield silent audio instead of real synthesis.
            sample_rate: Output sample rate in Hz (Kokoro defaults to 24000).
        """
        self.voice = voice
        self.lang_code = lang_code
        self.use_mock = use_mock
        self.sample_rate = sample_rate
        self._pipeline = None

        if not use_mock:
            self._load_pipeline()

    def _load_pipeline(self) -> None:
        """Initialize the Kokoro KPipeline."""
        try:
            from kokoro import KPipeline
            logger.info(
                "Initializing Kokoro KPipeline (voice=%s, lang=%s)",
                self.voice,
                self.lang_code,
            )
            self._pipeline = KPipeline(lang_code=self.lang_code)
            logger.info("Kokoro KPipeline loaded successfully.")
        except ImportError:
            logger.warning(
                "kokoro package not installed. Install with: pip install kokoro>=0.9.4"
            )
            self._pipeline = None
        except Exception as exc:
            logger.error("Failed to load Kokoro pipeline: %s", exc)
            self._pipeline = None

    def synthesize_streaming(
        self,
        text: str,
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """
        Synthesize text to audio, yielding sentence-sized audio chunks.

        Each yielded tuple is (audio_chunk: np.ndarray, sample_rate: int).
        The first sentence's audio is yielded as soon as it is ready,
        enabling streaming playback while the rest of the text is still
        being synthesized.

        Args:
            text: Input text to synthesize.

        Yields:
            (audio_chunk, sample_rate) tuples where audio_chunk is a
            float32 numpy array normalized to [-1.0, 1.0].

        Raises:
            RuntimeError: If real mode is requested but pipeline not loaded.
        """
        if self.use_mock:
            yield from self._mock_synthesize(text)
            return

        if self._pipeline is None:
            raise RuntimeError(
                "Kokoro pipeline is not loaded. Ensure the kokoro package "
                "is installed and a valid voice is available."
            )

        yield from self._real_synthesize_streaming(text)

    def _mock_synthesize(
        self,
        text: str,
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """Yield short silent audio chunks for testing."""
        # Split into "sentences" by punctuation for realistic streaming
        sentences = _SENTENCE_SPLIT_RE.split(text.strip())
        if not sentences:
            sentences = [text]

        # Generate ~0.3s of silence per sentence chunk
        chunk_samples = int(0.3 * self.sample_rate)
        for _ in sentences:
            time.sleep(0.02)  # Simulate processing time
            yield (np.zeros(chunk_samples, dtype=np.float32), self.sample_rate)

    def _real_synthesize_streaming(
        self,
        text: str,
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """Run real Kokoro synthesis with sentence-level streaming."""
        try:
            # The KPipeline itself handles sentence splitting internally;
            # it yields (gs, ps, audio) tuples per sentence.
            for result in self._pipeline(text, voice=self.voice):
                # result is typically a namedtuple or tuple with audio
                if hasattr(result, "audio"):
                    audio = result.audio
                elif isinstance(result, (list, tuple)) and len(result) >= 3:
                    audio = result[-1]  # Last element is usually audio
                else:
                    continue

                if audio is not None and len(audio) > 0:
                    # Ensure float32
                    audio = np.asarray(audio, dtype=np.float32)
                    yield (audio, self.sample_rate)

        except Exception as exc:
            logger.error("Kokoro synthesis failed: %s", exc)
            raise RuntimeError(f"TTS synthesis failed: {exc}") from exc

    def synthesize_full(
        self,
        text: str,
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize full text to audio (non-streaming).

        Args:
            text: Input text to synthesize.

        Returns:
            (audio, sample_rate) where audio is the concatenated waveform.
        """
        chunks = list(self.synthesize_streaming(text))
        if not chunks:
            return (np.zeros(0, dtype=np.float32), self.sample_rate)
        audio = np.concatenate([c[0] for c in chunks])
        return (audio, self.sample_rate)

    @property
    def is_loaded(self) -> bool:
        """Check if the real pipeline is loaded."""
        return self._pipeline is not None and not self.use_mock