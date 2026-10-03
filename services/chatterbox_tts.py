"""Chatterbox TTS service: OpenAITTSService pinned to the local Chatterbox server."""

from loguru import logger
from pipecat.services.openai.tts import VALID_VOICES, OpenAITTSService

from config import CFG
from services.sentence_batch import SentenceBatchAggregator


class ChatterboxTTSService(OpenAITTSService):
    """OpenAITTSService pinned to localhost Chatterbox.

    Args:
        base_url: Chatterbox server's OpenAI-compatible endpoint.
        model: Server-side model handle.
        voice: Reference clip filename in the server's voices/ or reference_audio/.

    """

    def __init__(self, base_url: str = CFG.tts.base_url, model: str = CFG.tts.model,
                 voice: str = CFG.tts.voice, **kwargs) -> None:
        """Pin the OpenAI TTS client to the Chatterbox endpoint and voice."""
        # Client-side voice gate rejects non-OpenAI voices before any request.
        VALID_VOICES[voice] = voice
        mode = CFG.tts.aggregation_mode
        batch: tuple[int, int] | None = None
        if mode == "batch":
            batch = (CFG.tts.batch_lead, CFG.tts.batch_carry)
        elif mode == "two_sentence":
            batch = (2, 2)
        elif mode not in ("sentence", "token"):
            raise ValueError(
                f"tts.aggregation_mode '{mode}' is not"
                " sentence/token/batch/two_sentence"
            )
        super().__init__(
            api_key="not-needed",
            base_url=base_url,
            settings=self.Settings(model=model, voice=voice),
            stop_frame_timeout_s=CFG.tts.stop_frame_timeout_s,
            text_aggregation_mode="token" if mode == "token" else "sentence",
            **kwargs,
        )
        self._text_aggregator = (
            SentenceBatchAggregator(lead=batch[0], carry=batch[1])
            if batch is not None
            else None
        )
        logger.info(
            f"ChatterboxTTSService -> {base_url} model={model} voice={voice}"
            f" aggregation={mode}"
            + (f" lead={batch[0]} carry={batch[1]}" if batch else "")
        )
