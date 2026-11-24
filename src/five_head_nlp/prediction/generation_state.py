"""Generation state tracker for sequence rollouts and stopping conditions."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple


class GenerationState:
    """Maintains token sequence history, rolling context, and sentence termination status.

    Sentence Boundary Rule:
        A terminal period ('.') marks the formal completion of a sentence.
        Once a terminal period is appended or max_tokens is reached, can_continue()
        becomes False and further token predictions are disallowed.
    """

    def __init__(
        self,
        prompt: Optional[Sequence[str]] = None,
        max_tokens: int = 25,
        terminal_token: str = ".",
    ) -> None:
        """Initialize generation state.

        Parameters
        ----------
        prompt : Optional[Sequence[str]], default=None
            Initial prompt tokens (if any).
        max_tokens : int, default=25
            Maximum total generated tokens permitted before mandatory termination.
        terminal_token : str, default='.'
            Token representing end of sentence.
        """
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")

        self.max_tokens: int = max_tokens
        self.terminal_token: str = terminal_token
        self._prompt: Tuple[str, ...] = tuple(prompt) if prompt else ()
        self._generated: List[str] = []
        self._is_terminated: bool = False

        # If prompt itself already ends with terminal token, immediately terminate
        if self._prompt and self._prompt[-1] == self.terminal_token:
            self._is_terminated = True

    @property
    def prompt(self) -> Tuple[str, ...]:
        """Return the initial prompt tokens."""
        return self._prompt

    @property
    def generated_tokens(self) -> List[str]:
        """Return list of generated tokens (excluding prompt)."""
        return list(self._generated)

    @property
    def all_tokens(self) -> List[str]:
        """Return full sequence (prompt + generated)."""
        return list(self._prompt) + self._generated

    @property
    def num_generated(self) -> int:
        """Return count of generated tokens so far."""
        return len(self._generated)

    @property
    def is_terminated(self) -> bool:
        """Return True if sentence has terminated (period reached or max tokens exhausted)."""
        return self._is_terminated or (self.num_generated >= self.max_tokens)

    def can_continue(self) -> bool:
        """Return True if further generation is allowed."""
        return not self.is_terminated

    def get_context(self, window_size: Optional[int] = None) -> Tuple[str, ...]:
        """Return the active context tuple.

        Parameters
        ----------
        window_size : Optional[int], default=None
            Number of recent tokens to include. If None, returns all tokens in sequence.

        Returns
        -------
        Tuple[str, ...]
            Context tokens.
        """
        full_seq = self.all_tokens
        if window_size is None or window_size >= len(full_seq):
            return tuple(full_seq)
        if window_size <= 0:
            return ()
        return tuple(full_seq[-window_size:])

    def append(self, token: str) -> None:
        """Append a newly selected token to the generation sequence.

        Parameters
        ----------
        token : str
            Selected token.

        Raises
        ------
        RuntimeError
            If attempted when can_continue() is False (sentence already terminated).
        """
        if not self.can_continue():
            raise RuntimeError(
                f"Cannot append token {token!r}: generation state is already terminated "
                f"(is_terminated={self.is_terminated}, num_generated={self.num_generated}, "
                f"max_tokens={self.max_tokens})."
            )

        self._generated.append(token)
        if token == self.terminal_token:
            self._is_terminated = True
        elif self.num_generated >= self.max_tokens:
            self._is_terminated = True

    def reset(self, new_prompt: Optional[Sequence[str]] = None) -> None:
        """Reset generation state for a new rollout."""
        self._prompt = tuple(new_prompt) if new_prompt else ()
        self._generated = []
        self._is_terminated = bool(self._prompt and self._prompt[-1] == self.terminal_token)
