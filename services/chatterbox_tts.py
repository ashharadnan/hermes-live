"""Chatterbox TTS service: OpenAITTSService pinned to the local Chatterbox server."""

from loguru import logger
from pipecat.services.openai.tts import VALID_VOICES, OpenAITTSService

from config import CFG


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
        super().__init__(
            api_key="not-needed",
            base_url=base_url,
            settings=self.Settings(model=model, voice=voice),
            stop_frame_timeout_s=CFG.tts.stop_frame_timeout_s,
            **kwargs,
        )
        logger.info(f"ChatterboxTTSService -> {base_url} model={model} voice={voice}")
