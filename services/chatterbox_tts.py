"""Chatterbox TTS service: OpenAITTSService pinned to the local Chatterbox server."""

from loguru import logger
from pipecat.services.openai.tts import VALID_VOICES, OpenAITTSService
from pipecat.utils.text.base_text_aggregator import AggregationType

from config import CFG
from services.two_sentence import TwoSentenceAggregator


class ChatterboxTTSService(OpenAITTSService):
    """OpenAITTSService pinned to localhost Chatterbox.

    Args:
        base_url: Chatterbox server's OpenAI-compatible endpoint.
        model: Server-side model handle.
        voice: Reference clip filename in the server's voices/ or reference_audio/.
    """

    def __init__(self, base_url: str = CFG.tts.base_url, model: str = CFG.tts.model,
                 voice: str = CFG.tts.voice, **kwargs) -> None:
        # Client-side voice gate rejects non-OpenAI voices before any request.
        VALID_VOICES[voice] = voice
        mode = CFG.tts.aggregation_mode
        if mode not in ("sentence", "two_sentence", "token"):
            raise ValueError(
                f"tts.aggregation_mode '{mode}' is not sentence/two_sentence/token"
            )
        super().__init__(
            api_key="not-needed",
            base_url=base_url,
            settings=self.Settings(model=model, voice=voice),
            stop_frame_timeout_s=CFG.tts.stop_frame_timeout_s,
            text_aggregation_mode="token" if mode == "token" else "sentence",
            **kwargs,
        )
        if mode == "two_sentence":
            self._text_aggregator = TwoSentenceAggregator(
                aggregation_type=AggregationType.SENTENCE,
            )
        logger.info(
            f"ChatterboxTTSService -> {base_url} model={model} voice={voice}"
            f" aggregation={mode}"
        )
