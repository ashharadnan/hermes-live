"""Two-sentence TTS batching: release one Aggregation per two sentences."""

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


class TwoSentenceAggregator(BaseTextAggregator):
    """Buffer LLM text to two sentence boundaries before releasing one pair.

    Replaces the SimpleTextAggregator pairing path entirely: that base's
    lookahead-driven yields dropped text when wrapped in a pairing subclass
    (measured: mid-pair sentences vanished). An own character buffer with a
    regex split keeps every character accounted for.
    """

    def __init__(
        self,
        *,
        aggregation_type: AggregationType = AggregationType.SENTENCE,
        language: str | None = None,
    ) -> None:
        """Initialize with an empty buffer."""
        super().__init__(aggregation_type=aggregation_type, language=language)
        self._buf: str = ""

    @property
    def text(self) -> Aggregation:
        """Get the currently buffered (unreleased) text."""
        return Aggregation(text=self._buf.strip(), type=AggregationType.SENTENCE)

    async def aggregate(self, text: str) -> AsyncIterator[Aggregation]:
        """Buffer text and yield one Aggregation per completed sentence pair."""
        self._buf += text
        parts = _SPLIT.split(self._buf)
        while len(parts) > 2:
            pair = parts[0] + " " + parts[1]
            rest = " ".join(parts[2:])
            self._buf = rest
            yield Aggregation(text=pair.strip(), type=AggregationType.SENTENCE)
            parts = _SPLIT.split(self._buf)

    async def flush(self) -> Aggregation | None:
        """Release any remaining buffered text at response end."""
        tail = self._buf.strip()
        self._buf = ""
        return Aggregation(text=tail, type=AggregationType.SENTENCE) if tail else None

    async def handle_interruption(self) -> None:
        """Discard the buffer; the interrupted turn is cancelled upstream."""
        self._buf = ""

    async def reset(self) -> None:
        """Clear the buffer."""
        self._buf = ""
