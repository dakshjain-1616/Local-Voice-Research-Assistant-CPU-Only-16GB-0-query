"""
OpenAI-compatible client for Phi-4-mini-instruct via llama.cpp server.

Connects to llama-server running in OpenAI-compatible mode at a configurable
base URL. Supports streaming text generation via Server-Sent Events (SSE).
"""

import json
import logging
import time
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# Default system prompt for the research assistant
_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful research assistant. Answer concisely and accurately."
)


class Phi4MiniClient:
    """
    Client for the Phi-4-mini-instruct LLM via llama.cpp OpenAI-compatible endpoint.

    Two modes:
      - Real: sends requests to llama-server at the configured base_url.
      - Mock: yields canned text chunks for testing.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8081/v1",
        max_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        use_mock: bool = False,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        model: Optional[str] = None,
    ):
        """
        Args:
            base_url: llama-server OpenAI-compatible endpoint URL.
            max_tokens: Maximum tokens to generate per response.
            temperature: Sampling temperature (0.0 = deterministic).
            top_p: Nucleus sampling parameter.
            repeat_penalty: Repetition penalty factor.
            use_mock: If True, yield canned text chunks instead of real API calls.
            system_prompt: System prompt for the LLM.
            model: Model name passed to the API (default: derived from base_url).
        """
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.use_mock = use_mock
        self.system_prompt = system_prompt
        self.model = model or "phi-4-mini-instruct"

        if not use_mock:
            self._init_client()

    def _init_client(self) -> None:
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key="not-needed")
            logger.info("LLM client initialized at %s", self.base_url)
        except ImportError:
            logger.warning(
                "openai package not installed. Install with: pip install openai"
            )
            self._client = None
        except Exception as exc:
            logger.error("Failed to initialize LLM client: %s", exc)
            self._client = None

    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True,
    ) -> Generator[str, None, None]:
        """
        Send a chat request and yield response text chunks.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                      Typically [{"role": "user", "content": "..."}] or
                      [{"role": "system", ...}, {"role": "user", ...}].
            stream: If True, yield tokens as they arrive (SSE streaming).

        Yields:
            Text chunks (str) as they are generated.

        Raises:
            RuntimeError: If real mode is used but client is not initialized.
            ConnectionError: If the llama-server endpoint is unreachable.
        """
        if self.use_mock:
            yield from self._mock_chat(messages)
            return

        if self._client is None:
            raise RuntimeError(
                "LLM client is not initialized. Ensure llama-server is running at "
                f"{self.base_url} and the openai package is installed."
            )

        yield from self._real_chat(messages, stream)

    def _mock_chat(
        self,
        messages: List[Dict[str, str]],
    ) -> Generator[str, None, None]:
        """Yield canned text chunks for testing."""
        chunks = [
            "This ",
            "is ",
            "a ",
            "mock ",
            "LLM ",
            "response ",
            "for ",
            "testing ",
            "purposes.",
        ]
        for chunk in chunks:
            time.sleep(0.01)  # Simulate some latency
            yield chunk

    def _real_chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool,
    ) -> Generator[str, None, None]:
        """Make real API call to llama-server."""
        # Prepend system prompt if not already present
        full_messages = list(messages)
        if not any(m.get("role") == "system" for m in full_messages):
            full_messages.insert(
                0, {"role": "system", "content": self.system_prompt}
            )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stream=stream,
                extra_body={"repeat_penalty": self.repeat_penalty}
                if hasattr(self._client, "chat")
                else {},
            )

            if stream:
                for chunk in response:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta and delta.content:
                        yield delta.content
            else:
                content = response.choices[0].message.content or ""
                yield content

        except Exception as exc:
            logger.error("LLM API call failed: %s", exc)
            raise ConnectionError(
                f"Failed to reach LLM server at {self.base_url}: {exc}"
            ) from exc