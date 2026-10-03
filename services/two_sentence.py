"""Two-sentence TTS batching: release aggregated text only every OTHER sentence.

Halves the number of TTS HTTP requests (each carries the engine's ~2.2s fixed
startup), doubling the audio per request so playback can hide the next
request's synthesis. Trade-off: the first spoken word waits for two sentences.

Interruptions and resets clear the pending pair buffer along with the base
sentence buffer.
"""

from pipecat.utils.text.base_text_aggregator import Aggregation, AggregationType
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator


class TwoSentenceAggregator(SimpleTextAggregator):
    """Buffer to two sentence boundaries before yielding one combined Aggregation.

    Odd-boundary text accumulates across the LLM token stream; on every second
    confirmed sentence the pair is released as a single sentence-type
    Aggregation. flush() releases any trailing unpaired text at response end.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._pending: str = ""

    async def aggregate(self, text: str):
        async for agg in super().aggregate(text):
            self._pending += agg.text + " "
            if self._sentence_count(self._pending) >= 2:
                yield self._release()

    @staticmethod
    def _sentence_count(text: str) -> int:
        return sum(text.count(c) for c in ".!?")

    def _release(self) -> Aggregation:
        text = self._pending.strip()
        self._pending = ""
        return Aggregation(text=text, type=AggregationType.SENTENCE)

    async def flush(self):
        base_tail = await super().flush()
        if self._pending.strip():
            merged = self._release()
            if base_tail:
                merged.text += ' ' + base_tail.text
            return merged
        return base_tail

    async def handle_interruption(self):
        self._pending = ""
        await super().handle_interruption()

    async def reset(self):
        self._pending = ""
        await super().reset()
