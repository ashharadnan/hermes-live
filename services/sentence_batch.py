"""Sentence-batching TTS aggregator: group N sentences per synthesis request."""

import re
from collections.abc import AsyncIterator

from pipecat.utils.text.base_text_aggregator import (
    Aggregation,
    AggregationType,
    BaseTextAggregator,
)

# Sentence split on [.!?] followed by whitespace and a capital/quote/paren start.
# Deliberately simpler than the base's lookahead machinery: no abbreviation or
# decimal handling - LLM prose in this pipeline is clean spoken text, and a
# rare mis-split only shifts which boundary pairs up, never loses text.
_SPLIT = re.compile(r"""(?<=[.!?])[ \t]+(?=[A-Z'"\(])""")

class SentenceBatchAggregator(BaseTextAggregator):
    """Group sentences into requests: first release holds `lead` sentences,
    every following release holds `carry` sentences.

    Args:
        lead: Sentences in the FIRST release (small = fast first audio).
        carry: Sentences in every SUBSEQUENT release (larger = wider cushion
            against the next synthesis overtaking playback).
        aggregation_type: Base class knob; keep SENTENCE.
        language: Optional language tag for the base's tokenizer plumbing.

    """

    def __init__(
        self,
        *,
        lead: int = 2,
        carry: int = 2,
        aggregation_type: AggregationType = AggregationType.SENTENCE,
        language: str | None = None,
    ) -> None:
        """Initialise with an empty buffer and a full quota for the lead."""
        super().__init__(aggregation_type=aggregation_type, language=language)
        self._lead = max(1, lead)
        self._carry = max(1, carry)
        self._quota = self._lead  # sentences needed before the next release
        self._buf: str = ""

    @property
    def text(self) -> Aggregation:
        """Get the currently buffered (unreleased) text."""
        return Aggregation(text=self._buf.strip(), type=AggregationType.SENTENCE)

    async def aggregate(self, text: str) -> AsyncIterator[Aggregation]:
        """Buffer text and release one Aggregation per filled quota."""
        self._buf += text
        parts = _SPLIT.split(self._buf)
        while len(parts) > self._quota:
            batch = " ".join(parts[: self._quota])
            rest = " ".join(parts[self._quota :])
            self._buf = rest
            yield Aggregation(
                text=" ".join(batch.split()), type=AggregationType.SENTENCE
            )
            self._quota = self._carry  # lead applies only to the first release
            parts = _SPLIT.split(self._buf)

    async def flush(self) -> Aggregation | None:
        """Release any remaining buffered text at response end."""
        tail = " ".join(self._buf.split())
        self._buf = ""
        self._quota = self._lead  # next turn resets to the lead size
        return Aggregation(text=tail, type=AggregationType.SENTENCE) if tail else None

    async def handle_interruption(self) -> None:
        """Discard the buffer; the interrupted turn is cancelled upstream."""
        self._buf = ""

    async def reset(self) -> None:
        """Clear the buffer and reset the quota to the lead size."""
        self._buf = ""
        self._quota = self._lead
