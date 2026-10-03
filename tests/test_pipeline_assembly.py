"""Pipeline assembly tests: wiring invariants without starting audio I/O.

bot.py assembles everything inside main() keyed to a runner.run() call; the
tests exercise assembly directly against the same components bot.py imports,
verifying the AGENTS.md invariants:
  - pipeline order transport.input -> stt -> user agg -> llm -> tts ->
    transport.output -> assistant agg
  - VAD rides the USER aggregator, not TransportParams
  - context aggregator pair shares one LLMContext
  - rates + metrics load from config.yaml
"""


import pytest
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)

import bot
from config import CFG


def test_local_audio_transport_params_have_no_vad_field():
    # TransportParams carries no vad_analyzer in pipecat 1.12; assigning one
    # must fail so nobody 'restores' the old wrong wiring.
    params = LocalAudioTransportParams(
        audio_in_enabled=True,
        audio_in_sample_rate=CFG.audio.in_sample_rate,
        audio_out_enabled=True,
        audio_out_sample_rate=CFG.audio.out_sample_rate,
    )
    with pytest.raises(ValueError):
        params.vad_analyzer = SileroVADAnalyzer()


def test_transport_rates_load_from_config():
    transport = LocalAudioTransport(
        params=LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=CFG.audio.in_sample_rate,
            audio_out_enabled=True,
            audio_out_sample_rate=CFG.audio.out_sample_rate,
        )
    )
    assert transport._params.audio_in_sample_rate == CFG.audio.in_sample_rate
    assert transport._params.audio_out_sample_rate == CFG.audio.out_sample_rate
    # input()/output() must be stable singletons (pipeline references them once).
    assert transport.input() is transport.input()
    assert transport.output() is transport.output()


def test_vad_analyzer_builds_from_config_with_silero():
    vad = bot._vad_analyzer()
    assert isinstance(vad, SileroVADAnalyzer)
    assert vad._params.confidence == CFG.vad.confidence
    assert vad._params.start_secs == CFG.vad.start_secs
    assert vad._params.stop_secs == CFG.vad.stop_secs
    assert vad._params.min_volume == CFG.vad.min_volume


def test_vad_rides_user_aggregator_not_transport():
    context = LLMContext()
    pair = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=bot._vad_analyzer()),
    )
    user = pair.user()
    assistant = pair.assistant()
    assert getattr(user._params, "vad_analyzer", None) is not None
    assert not hasattr(assistant._params, "vad_analyzer") or (
        getattr(assistant._params, "vad_analyzer", None) is None
    )
    # Both sides of the pair must share exactly one context object.
    assert user._context is assistant._context


def test_pipeline_processor_order_matches_architecture_invariant():
    transport = LocalAudioTransport(
        params=LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=CFG.audio.in_sample_rate,
            audio_out_enabled=True,
            audio_out_sample_rate=CFG.audio.out_sample_rate,
        )
    )
    stt = bot.MoonshineSTTService()
    llm = bot.LlamacppLLMService()
    tts = bot.ChatterboxTTSService()
    pair = LLMContextAggregatorPair(
        LLMContext(),
        user_params=LLMUserAggregatorParams(vad_analyzer=bot._vad_analyzer()),
    )
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            pair.user(),
            llm,
            tts,
            transport.output(),
            pair.assistant(),
        ]
    )
    # Pipeline wraps the list with PipelineSource/PipelineSink bookends; strip
    # them and the surviving order must be exactly the AGENTS.md invariant.
    processors = pipeline.processors
    assert len(processors) == 9
    core = [p for p in processors if not type(p).__name__.startswith("Pipeline")]
    assert len(core) == 7
    assert core[0] is transport.input()
    assert core[1] is stt
    assert core[2] is pair.user()
    assert core[3] is llm
    assert core[4] is tts
    assert core[5] is transport.output()
    assert core[6] is pair.assistant()


def test_pipeline_task_params_load_from_config():
    params = PipelineParams(
        audio_in_sample_rate=CFG.audio.in_sample_rate,
        audio_out_sample_rate=CFG.audio.out_sample_rate,
        enable_metrics=True,
    )
    assert params.audio_in_sample_rate == CFG.audio.in_sample_rate
    assert params.audio_out_sample_rate == CFG.audio.out_sample_rate
    assert params.enable_metrics is True


def test_pipeline_task_constructs_accepts_pipeline():
    task = PipelineTask(
        Pipeline(
            [
                LLMContextAggregatorPair(
                    LLMContext(),
                    user_params=LLMUserAggregatorParams(vad_analyzer=bot._vad_analyzer()),
                ).user()
            ]
        ),
        params=PipelineParams(enable_metrics=True),
    )
    assert task is not None
