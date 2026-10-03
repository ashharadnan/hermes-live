"""Moonshine STT tests: real transcription ground truth, service wiring.

The hello.wav asset is 16 kHz mono 16-bit PCM generated with Windows SAPI;
the model transcribes it deterministically on CPU. First import of the STT
module downloads/loads the ONNX weights from the project model_cache/ (~1 s
warm; ~8 s if the cache was ever empty).
"""

import wave
from pathlib import Path

import pytest
from pipecat.services.moonshine.stt import MoonshineSTTService

WAV_PATH = Path(__file__).parent / "data" / "hello.wav"
EXPECTED_TEXT = (
    "Hello there, this is a voice pipeline test clip. The weather is sunny today."
)


def _load_wav_pcm() -> bytes:
    with wave.open(str(WAV_PATH), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        return wav.readframes(wav.getnframes())


@pytest.fixture(scope="module")
def stt_service() -> MoonshineSTTService:
    return MoonshineSTTService()


@pytest.fixture(scope="module")
def pcm() -> bytes:
    return _load_wav_pcm()


class TestTranscription:
    @pytest.mark.timeout(60)
    async def test_transcribes_reference_clip(self, stt_service, pcm):
        frames = [f async for f in stt_service.run_stt(pcm)]
        transcriptions = [f for f in frames if type(f).__name__ == "TranscriptionFrame"]
        assert transcriptions, "speech clip must produce a TranscriptionFrame"
        text = transcriptions[0].text.strip()
        assert EXPECTED_TEXT.lower() in text.lower()

    @pytest.mark.timeout(30)
    async def test_silence_yields_no_transcription(self, stt_service):
        silence = b"\x00" * 16000 * 2  # one second of digital silence
        frames = [f async for f in stt_service.run_stt(silence)]
        transcriptions = [f for f in frames if type(f).__name__ == "TranscriptionFrame"]
        assert not transcriptions, "silence must not produce a TranscriptionFrame"

    @pytest.mark.timeout(30)
    async def test_empty_audio_yields_no_transcription(self, stt_service):
        frames = [f async for f in stt_service.run_stt(b"")]
        transcriptions = [f for f in frames if type(f).__name__ == "TranscriptionFrame"]
        assert not transcriptions

    @pytest.mark.timeout(60)
    async def test_transcription_is_deterministic(self, stt_service, pcm):
        first = [f async for f in stt_service.run_stt(pcm)]
        second = [f async for f in stt_service.run_stt(pcm)]
        t1 = [getattr(f, "text", "") for f in first]
        t2 = [getattr(f, "text", "") for f in second]
        assert t1 == t2


class TestServiceWiring:
    def test_is_segmented_stt(self, stt_service):
        # Moonshine is a SegmentedSTTService: requires VAD upstream for
        # segmentation; no interim results by design.
        assert type(stt_service).__name__ == "MoonshineSTTService"
        bases = {c.__name__ for c in type(stt_service).__mro__}
        assert "SegmentedSTTService" in bases

    def test_language_setting_loaded(self, stt_service):
        assert getattr(stt_service._settings, "language", None) is not None
