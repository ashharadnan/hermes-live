"""ChatterboxTTSService tests: voice-table registration, frame flow, timeouts."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pipecat.frames.frames import (
    ErrorFrame,
    TTSAudioRawFrame,
)
from pipecat.services.openai.tts import VALID_VOICES

from config import CFG
from services.chatterbox_tts import ChatterboxTTSService


@pytest.fixture()
def tts() -> ChatterboxTTSService:
    return ChatterboxTTSService()


class TestConstruction:
    def test_custom_voice_registered_into_client_side_table(self, tts):
        assert CFG.tts.voice in VALID_VOICES
        assert VALID_VOICES[CFG.tts.voice] == CFG.tts.voice

    def test_settings_pinned_from_config(self, tts):
        assert tts._settings.model == CFG.tts.model
        assert tts._settings.voice == CFG.tts.voice

    def test_raised_stop_frame_timeout_wired(self, tts):
        assert tts._stop_frame_timeout_s == CFG.tts.stop_frame_timeout_s
        # The whole reason for the override: default 3.0 s abandons contexts
        # under GPU load; config value must clear it decisively.
        assert tts._stop_frame_timeout_s > 3.0

    def test_explicit_ctor_args_pin_into_settings(self):
        # The ctor path itself: constructor arguments (matching config.yaml)
        # must land in the settings object, not OpenAI defaults. NOTE: the
        # service injects stop_frame_timeout_s from CFG.tts itself -- passing
        # it again is a duplicate-kwarg TypeError, so the config value has
        # exactly one home.
        service = ChatterboxTTSService(
            base_url=CFG.tts.base_url,
            model=CFG.tts.model,
            voice=CFG.tts.voice,
        )
        assert service._settings.model == CFG.tts.model
        assert service._settings.voice == CFG.tts.voice
        assert service._stop_frame_timeout_s == CFG.tts.stop_frame_timeout_s


def _streaming_response(chunks, status_code: int = 200, error_text: str = ""):
    """Build a MagicMock shaped like the httpx streaming-response CM used by
    run_tts: `async with ... as r` with awaitable r.text and async-iterable
    r.iter_bytes(chunk_size)."""
    response = MagicMock()
    response.status_code = status_code
    response.text = AsyncMock(return_value=error_text)

    async def _aiter_bytes(chunk_size=None):
        for chunk in chunks:
            yield chunk

    response.iter_bytes = MagicMock(side_effect=_aiter_bytes)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=response)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _wire_client(tts: ChatterboxTTSService, cm) -> MagicMock:
    speech = MagicMock()
    speech.with_streaming_response.create = MagicMock(return_value=cm)
    client = MagicMock()
    client.audio.speech = speech
    tts._client = client
    return speech


class TestVoiceGate:
    async def test_unknown_voice_yields_error_frame_before_any_request(
        self, tts, monkeypatch
    ):
        # Remove the registered voice so run_tts takes the rejection path;
        # proves the gate itself is what rejects, with zero HTTP traffic.
        monkeypatch.delitem(VALID_VOICES, CFG.tts.voice)
        client_mock = MagicMock()
        tts._client = client_mock
        frames = [frame async for frame in tts.run_tts("hello", "ctx-1")]
        errors = [f for f in frames if isinstance(f, ErrorFrame)]
        assert errors, "unregistered voice must produce an ErrorFrame"
        assert "not supported" in errors[0].error
        client_mock.audio.speech.with_streaming_response.create.assert_not_called()

    async def test_registered_voice_passes_gate_and_attempts_request(
        self, tts, monkeypatch
    ):
        monkeypatch.setitem(VALID_VOICES, CFG.tts.voice, CFG.tts.voice)
        chunk = b"\x00\x01" * 100
        cm = _streaming_response([chunk])
        speech = _wire_client(tts, cm)

        frames = []
        async for frame in tts.run_tts("hello world", "ctx-1"):
            frames.append(frame)

        create_kwargs = speech.with_streaming_response.create.call_args.kwargs
        assert create_kwargs["voice"] == CFG.tts.voice
        assert create_kwargs["model"] == CFG.tts.model
        assert create_kwargs["response_format"] == "pcm"
        assert create_kwargs["input"] == "hello world"
        # Request-body shape: exactly the OpenAI speech surface, no stray
        # non-OpenAI params leaking through.
        assert set(create_kwargs) == {"input", "model", "voice", "response_format"}
        audio = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
        assert audio, "voice-gated request with 200 response must yield audio"
        assert audio[0].audio == chunk


class TestAudioContract:
    async def test_server_bytes_pass_through_headerless(self, tts, monkeypatch):
        """The pipeline's audio.out_rate contract says the bytes are played at
        the declared sample rate; run_tts must not resample or wrap them."""
        monkeypatch.setitem(VALID_VOICES, CFG.tts.voice, CFG.tts.voice)
        payload = bytes(range(256)) * 4
        cm = _streaming_response([payload])
        _wire_client(tts, cm)

        frames = [f async for f in tts.run_tts("x", "ctx-1")]
        audio = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
        assert audio[0].audio == payload

    async def test_non_200_yields_error_frame(self, tts, monkeypatch):
        monkeypatch.setitem(VALID_VOICES, CFG.tts.voice, CFG.tts.voice)
        cm = _streaming_response([], status_code=500, error_text="engine exploded")
        _wire_client(tts, cm)

        frames = [f async for f in tts.run_tts("x", "ctx-1")]
        errors = [f for f in frames if isinstance(f, ErrorFrame)]
        assert errors and "500" in errors[0].error

    async def test_empty_chunks_yield_no_audio_frames(self, tts, monkeypatch):
        monkeypatch.setitem(VALID_VOICES, CFG.tts.voice, CFG.tts.voice)
        cm = _streaming_response([b"", b""])
        _wire_client(tts, cm)

        frames = [f async for f in tts.run_tts("x", "ctx-1")]
        audio = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
        assert not audio, "empty chunks must be filtered, not emitted as audio"

    def test_default_voice_not_used(self, tts):
        # The reference-clip voice is the configured one; OpenAI built-ins
        # must not leak through as fallbacks.
        assert tts._settings.voice == CFG.tts.voice
