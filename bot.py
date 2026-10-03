r"""Voice assistant: Moonshine STT -> OpenAI-compatible LLM -> OpenAI-compatible TTS.

Half-duplex but interruptible: VAD segments speech, Moonshine transcribes
(CPU/ONNX), the LLM server streams the reply, the TTS server returns pcm. User
speech during playback triggers InterruptionFrame -> in-flight output cancelled.

Prereqs: an OpenAI-compatible LLM server (:18080) and TTS server (:8004) started
before this script - endpoints and models come from config.yaml.

Usage: python bot.py  (local mic + speakers)
"""

import asyncio
import os
from pathlib import Path

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import WorkerRunner
from pipecat.pipeline.task import PipelineParams
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.moonshine.stt import MoonshineSTTService
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)

from config import CFG
from services.chatterbox_tts import ChatterboxTTSService
from services.llamacpp_llm import LlamacppLLMService

# Moonshine (moonshine_voice) reads MOONSHINE_VOICE_CACHE for its model cache;
# must be set before the STT service constructs and downloads on first run.
os.environ.setdefault(
    "MOONSHINE_VOICE_CACHE",
    str((Path(__file__).parent / CFG.stt.cache_dir).resolve()),
)


def _vad_analyzer() -> SileroVADAnalyzer:
    """Build the Silero VAD analyzer from the config.yaml vad section."""
    return SileroVADAnalyzer(
        params=VADParams(
            confidence=CFG.vad.confidence,
            start_secs=CFG.vad.start_secs,
            stop_secs=CFG.vad.stop_secs,
            min_volume=CFG.vad.min_volume,
        )
    )


async def main() -> None:
    """Assemble and run the voice pipeline on local mic/speakers."""
    transport = LocalAudioTransport(
        params=LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=CFG.audio.in_sample_rate,
            audio_out_enabled=True,
            audio_out_sample_rate=CFG.audio.out_sample_rate,
        ),
    )

    stt = MoonshineSTTService()
    llm = LlamacppLLMService()
    tts = ChatterboxTTSService()

    # VAD rides the USER aggregator in pipecat 1.12 (turn detection + interruptions).
    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=_vad_analyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=CFG.audio.in_sample_rate,
            audio_out_sample_rate=CFG.audio.out_sample_rate,
            enable_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_connected(transport, client):
        logger.info("Local audio session started - speak whenever ready")

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, client):
        logger.info("Session ended")
        await task.cancel()

    runner = WorkerRunner()
    await runner.add_workers(task)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
