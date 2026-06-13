"""
Turn-taking state machine for the speech-to-speech pipeline.

Manages conversation flow through states:
  IDLE -> LISTENING -> PROCESSING -> SPEAKING -> IDLE

Supports barge-in: SPEAKING -> LISTENING transition when user speech is
detected during assistant playback.
"""

import logging
from enum import Enum, auto
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class TurnState(Enum):
    """Conversation turn-taking states."""
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()

    def __str__(self) -> str:
        return self.name


# Valid transitions: (from_state, to_state)
_VALID_TRANSITIONS: set[tuple[TurnState, TurnState]] = {
    (TurnState.IDLE, TurnState.LISTENING),        # Start listening for audio
    (TurnState.IDLE, TurnState.PROCESSING),       # Direct text input (skip audio)
    (TurnState.LISTENING, TurnState.IDLE),        # No speech detected (timeout)
    (TurnState.LISTENING, TurnState.PROCESSING),  # Speech ended, start processing
    (TurnState.PROCESSING, TurnState.SPEAKING),   # Response ready, start TTS
    (TurnState.PROCESSING, TurnState.IDLE),       # Error or empty response
    (TurnState.SPEAKING, TurnState.IDLE),         # TTS finished
    (TurnState.SPEAKING, TurnState.LISTENING),    # Barge-in: user interrupts
    (TurnState.IDLE, TurnState.IDLE),             # No-op reset
}


class TurnTakingStateMachine:
    """
    Finite state machine for conversation turn-taking.

    Tracks the current state and enforces valid transitions.
    Supports optional callbacks for state enter/exit events.

    Barge-in support:
      While SPEAKING, if user speech is detected, the state machine
      transitions SPEAKING -> LISTENING, triggering the barge-in callback
      to stop TTS playback.
    """

    def __init__(self, initial_state: TurnState = TurnState.IDLE):
        """
        Args:
            initial_state: Starting state (default: IDLE).
        """
        self._state = initial_state
        self._on_enter_callbacks: dict[TurnState, list[Callable]] = {
            s: [] for s in TurnState
        }
        self._on_exit_callbacks: dict[TurnState, list[Callable]] = {
            s: [] for s in TurnState
        }
        self._transition_count: int = 0

    @property
    def state(self) -> TurnState:
        """Current state."""
        return self._state

    @property
    def transition_count(self) -> int:
        """Number of state transitions that have occurred."""
        return self._transition_count

    def on_enter(self, state: TurnState, callback: Callable) -> None:
        """
        Register a callback for when a state is entered.

        Args:
            state: The state to watch.
            callback: Callable invoked as callback(from_state, to_state).
        """
        self._on_enter_callbacks.setdefault(state, []).append(callback)

    def on_exit(self, state: TurnState, callback: Callable) -> None:
        """
        Register a callback for when a state is exited.

        Args:
            state: The state to watch.
            callback: Callable invoked as callback(from_state, to_state).
        """
        self._on_exit_callbacks.setdefault(state, []).append(callback)

    def transition(self, new_state: TurnState) -> bool:
        """
        Attempt to transition to a new state.

        Args:
            new_state: The target state.

        Returns:
            True if the transition was valid and executed, False otherwise.

        Raises:
            ValueError: If the transition is not valid per _VALID_TRANSITIONS.
        """
        old_state = self._state
        transition = (old_state, new_state)

        if old_state == new_state:
            # Self-transition (e.g., IDLE -> IDLE reset) is allowed
            return True

        if transition not in _VALID_TRANSITIONS:
            logger.warning(
                "Invalid transition: %s -> %s", old_state, new_state
            )
            raise ValueError(
                f"Invalid state transition: {old_state} -> {new_state}. "
                f"Valid from {old_state}: "
                f"{[t[1].name for t in _VALID_TRANSITIONS if t[0] == old_state]}"
            )

        # Fire exit callbacks for old state
        for cb in self._on_exit_callbacks.get(old_state, []):
            try:
                cb(old_state, new_state)
            except Exception as e:
                logger.error("Exit callback error: %s", e)

        # Execute transition
        self._state = new_state
        self._transition_count += 1

        # Fire enter callbacks for new state
        for cb in self._on_enter_callbacks.get(new_state, []):
            try:
                cb(old_state, new_state)
            except Exception as e:
                logger.error("Enter callback error: %s", e)

        logger.debug("Transition: %s -> %s (total: %d)", old_state, new_state, self._transition_count)
        return True

    def reset(self) -> None:
        """Reset to IDLE state."""
        self.transition(TurnState.IDLE)

    def can_transition_to(self, state: TurnState) -> bool:
        """Check if a transition to the given state is valid."""
        return (self._state, state) in _VALID_TRANSITIONS or self._state == state

    def is_idle(self) -> bool:
        """Check if in IDLE state."""
        return self._state == TurnState.IDLE

    def is_listening(self) -> bool:
        """Check if in LISTENING state."""
        return self._state == TurnState.LISTENING

    def is_processing(self) -> bool:
        """Check if in PROCESSING state."""
        return self._state == TurnState.PROCESSING

    def is_speaking(self) -> bool:
        """Check if in SPEAKING state."""
        return self._state == TurnState.SPEAKING