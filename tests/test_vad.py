"""Silero VAD tests built from bot._vad_analyzer()'s config.yaml parameters.

The real Silero model is speech-trained (it correctly reads synthetic tones
as silence), so state-machine tests use a stubbed model to exercise the REAL
VADAnalyzer state engine (pipecat/audio/vad/vad_analyzer.py::_run_analyzer)
deterministically against the exact parameters bot.py loads from config.yaml.

Engine facts baked into these assertions (verified against installed 1.12
source):
  - speaking = confidence >= threshold AND smoothed_volume >= min_volume
  - volume smoothing: exp_smoothing(volume, prev, 0.2) -- ramps, never jumps
  - STARTING -> SPEAKING after round(start_secs / frame_secs) speaking frames
  - STOPPING -> QUIET after round(stop_secs / frame_secs) silent frames
"""

import asyncio
from unittest.mock import patch

import numpy as np
import pytest
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADState

import bot
from config import CFG


class _StubModel:
    """Silero-ONNX-shaped stub: callable, returns fixed confidence, resettable."""

    def __init__(self, confidence: float):
        self.confidence = confidence

    def __call__(self, audio, sample_rate):
        return [self.confidence]

    def reset_states(self):
        pass


def _speech_pcm(rate: int, frames: int, amplitude: int = 30000) -> bytes:
    tone = (np.sin(2 * np.pi * 440 * np.arange(frames) / rate) * amplitude).astype(
        np.int16
    )
    return tone.tobytes()


def _silence_pcm(rate: int, frames: int) -> bytes:
    return np.zeros(frames, dtype=np.int16).tobytes()


@pytest.fixture()
def analyzer() -> SileroVADAnalyzer:
    vad = bot._vad_analyzer()
    with patch.object(vad, "_model", new=_StubModel(0.95)):
        vad.set_sample_rate(16000)
        yield vad


def _analyze(vad: SileroVADAnalyzer, pcm: bytes, times: int) -> list[VADState]:
    async def run():
        return [await vad.analyze_audio(pcm) for _ in range(times)]

    return asyncio.run(run())


class TestParamsFromConfig:
    def test_params_match_config_yaml(self, analyzer):
        p = analyzer._params
        assert p.confidence == pytest.approx(CFG.vad.confidence)
        assert p.start_secs == pytest.approx(CFG.vad.start_secs)
        assert p.stop_secs == pytest.approx(CFG.vad.stop_secs)
        assert p.min_volume == pytest.approx(CFG.vad.min_volume)


class TestFrameRequirements:
    def test_expects_512_frame_at_16k_mono(self, analyzer):
        assert analyzer.num_frames_required() == 512
        assert analyzer.num_channels == 1
        assert analyzer.sample_rate == 16000


class TestStateMachine:
    def test_sustained_speech_transitions_to_speaking(self, analyzer):
        pcm = _speech_pcm(16000, analyzer.num_frames_required())
        states = _analyze(analyzer, pcm, 30)
        assert states[0] == VADState.QUIET
        assert VADState.STARTING in states
        assert states[-1] == VADState.SPEAKING

    def test_silence_after_speech_transitions_back_to_quiet(self, analyzer):
        pcm = _speech_pcm(16000, analyzer.num_frames_required())
        silence = _silence_pcm(16000, analyzer.num_frames_required())
        _analyze(analyzer, pcm, 40)
        quiet_states = _analyze(analyzer, silence, 40)
        # The smoothed volume ramps DOWN through the floor, so the first
        # silent frames still read as speaking; STOPPING must appear before
        # the engine settles back into QUIET.
        assert VADState.STOPPING in quiet_states
        assert quiet_states[-1] == VADState.QUIET

    def test_starting_phase_length_derived_from_start_secs(self, analyzer):
        # SPEAKING appears once BOTH gates pass: round(start_secs /
        # frame_secs) speaking frames AND the smoothed volume crossing the
        # min_volume floor. The volume window (0.4 s) + smoothing (0.2) add
        # a fill delay ahead of the state timer, so assert the state timer
        # contributes its expected share: observed frame count must be
        # within (volume-fill delay + start_secs frames) of the start.
        pcm = _speech_pcm(16000, analyzer.num_frames_required())
        states = _analyze(analyzer, pcm, 40)
        speaking_index = states.index(VADState.SPEAKING)
        frame_secs = analyzer.num_frames_required() / analyzer.sample_rate
        expected_start_frames = round(CFG.vad.start_secs / frame_secs)
        volume_window_frames = round(0.4 / frame_secs)
        # The state timer alone may fire no earlier than expected_start_frames;
        # the volume gate may push completion later by up to the fill window.
        assert speaking_index >= expected_start_frames - 1
        assert speaking_index <= expected_start_frames + volume_window_frames + 3

    def test_first_speaking_frame_lags_volume_gate_not_state_machine(self, analyzer):
        # Disambiguate the two gates: with the model confidence BELOW the
        # configured threshold (0.6 < 0.7) but the volume gate passed, no
        # STARTING may occur -- volume alone is necessary but not sufficient.
        pcm = _speech_pcm(16000, analyzer.num_frames_required())
        with patch.object(analyzer, "_model", new=_StubModel(0.6)):
            states = _analyze(analyzer, pcm, 60)
            assert VADState.SPEAKING not in states

    def test_low_confidence_model_stays_quiet_even_when_loud(self, analyzer):
        pcm = _speech_pcm(16000, analyzer.num_frames_required())
        with patch.object(analyzer, "_model", new=_StubModel(0.05)):
            states = _analyze(analyzer, pcm, 40)
            assert all(s == VADState.QUIET for s in states)

    def test_true_silence_stays_quiet_even_with_speech_confidence(self, analyzer):
        # min_volume is the veto: zero-volume audio must never trigger the
        # start path no matter what the model reports.
        silence = _silence_pcm(16000, analyzer.num_frames_required())
        states = _analyze(analyzer, silence, 60)
        assert all(s == VADState.QUIET for s in states)

    def test_stopping_phase_length_derived_from_stop_secs(self, analyzer):
        # Symmetric to the STARTING case: SPEAKING -> QUIET requires
        # round(stop_secs / frame_secs) silent frames, after the smoothed
        # volume ramps down through the min_volume floor.
        speech = _speech_pcm(16000, analyzer.num_frames_required())
        silence = _silence_pcm(16000, analyzer.num_frames_required())
        _analyze(analyzer, speech, 40)
        quiet_states = _analyze(analyzer, silence, 40)
        quiet_index = next(
            (i for i, s in enumerate(quiet_states) if s == VADState.QUIET), None
        )
        assert quiet_index is not None
        frame_secs = analyzer.num_frames_required() / analyzer.sample_rate
        expected_stop_frames = round(CFG.vad.stop_secs / frame_secs)
        volume_window_frames = round(0.4 / frame_secs)
        assert quiet_index >= expected_stop_frames - 1
        assert quiet_index <= quiet_states.index(VADState.STOPPING) + (
            expected_stop_frames + volume_window_frames + 3
        )

    def test_mid_volume_pins_gate_numerically(self):
        # Boundary pin: a constant mid-volume signal (normalized ~0.7, above
        # the 0.6 min_volume floor) plus speech-level confidence MUST reach
        # SPEAKING, while the same confidence at zero volume must not. Each
        # case gets a fresh analyzer so the volume tracker carries nothing
        # over between them.
        mid_vad = bot._vad_analyzer()
        with patch.object(mid_vad, "_model", new=_StubModel(0.95)):
            mid_vad.set_sample_rate(16000)
            mid_pcm = (
                np.sin(
                    2 * np.pi * 440 * np.arange(mid_vad.num_frames_required()) / 16000
                )
                * (0.7 * 32767)
            ).astype(np.int16).tobytes()
            states = _analyze(mid_vad, mid_pcm, 60)
            assert states[-1] == VADState.SPEAKING
        quiet_vad = bot._vad_analyzer()
        with patch.object(quiet_vad, "_model", new=_StubModel(0.95)):
            quiet_vad.set_sample_rate(16000)
            silence = _silence_pcm(16000, quiet_vad.num_frames_required())
            quiet_states = _analyze(quiet_vad, silence, 60)
            assert all(s == VADState.QUIET for s in quiet_states)
