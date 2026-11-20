"""Short-term conversation memory: the last few question and answer pairs."""

from collections import deque

from docqa.answer import Turn

DEFAULT_MAX_TURNS = 6


class ConversationMemory:
    """Remembers the most recent exchanges of one chat session; older ones are dropped."""

    def __init__(self, max_turns: int = DEFAULT_MAX_TURNS) -> None:
        if max_turns < 1:
            raise ValueError(f"max_turns must be at least 1, got {max_turns}")
        self._max_turns = max_turns
        self._turns: deque[Turn] = deque(maxlen=max_turns)

    @property
    def max_turns(self) -> int:
        """How many exchanges are remembered."""
        return self._max_turns

    @property
    def turns(self) -> tuple[Turn, ...]:
        """A snapshot of the remembered turns, oldest first."""
        return tuple(self._turns)

    def add(self, question: str, answer: str) -> None:
        """Remember one exchange; the oldest is dropped when the memory is full."""
        self._turns.append(Turn(question=question, answer=answer))

    def clear(self) -> None:
        """Forget every remembered exchange."""
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)
