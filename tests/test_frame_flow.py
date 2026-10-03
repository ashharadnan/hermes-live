"""Frame-flow tests: user turn aggregation and LLM text streaming path.

Uses the real pipecat frame machinery. The OpenAI HTTP layer is stubbed at
``service._client`` (an object shaped like AsyncOpenAI), so these are true
pipeline-path tests, not dict-logic unit tests.
"""

from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from pipecat.frames.frames import (
    EndFrame,
    LLMTextFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.base_object import BaseObject

from services.llamacpp_llm import LlamacppLLMService

try:
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

    _OPENAI_TYPES_OK = True
except Exception:  # pragma: no cover - openai types are a hard dep of pipecat[openai]
    _OPENAI_TYPES_OK = False


def _chunk(delta_content: str | None, finish_reason: str | None = None):
    return ChatCompletionChunk(
        id="chunk-1",
        choices=[
            Choice(
                index=0,
                delta=ChoiceDelta(content=delta_content),
                finish_reason=finish_reason,
            )
        ],
        created=1700000000,
        model="test-model",
        object="chat.completion.chunk",
    )


class _FakeStream:
    """Async iterator over prepared chunks, shaped like AsyncStream."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


@pytest.fixture()
def llm() -> LlamacppLLMService:
    return LlamacppLLMService()


def _stub_completion(llm: LlamacppLLMService, chunks, capture: dict):
    client = MagicMock()

    async def capture_create(**kwargs):
        capture.update(kwargs)
        return _FakeStream(chunks)

    client.chat.completions.create = capture_create
    llm._client = client


@pytest.mark.skipif(not _OPENAI_TYPES_OK, reason="openai types unavailable")
class TestLLMTextPath:
    async def test_streamed_chunks_become_llm_text_frames(self, llm):
        """The real production path is _process_context: it converts each
        content chunk into a pushed LLMTextFrame."""
        chunks = [
            _chunk("Hello"),
            _chunk(" there"),
            _chunk(None, finish_reason="stop"),
        ]
        capture: dict = {}
        _stub_completion(llm, chunks, capture)

        pushed: list = []

        async def capture_push(frame, direction=None):
            pushed.append(frame)

        llm.push_frame = capture_push

        context = LLMContext()
        context.add_message({"role": "user", "content": "say hi"})
        await llm._process_context(context)
        texts = [f for f in pushed if isinstance(f, LLMTextFrame)]
        assert [f.text for f in texts] == ["Hello", " there"]

    async def test_request_params_carry_thinking_off_flag(self, llm):
        """The request the stubbed client receives must nest the thinking-off
        flag under extra_body on every completion request."""
        chunks = [_chunk("ok", finish_reason="stop")]
        capture: dict = {}
        _stub_completion(llm, chunks, capture)

        context = LLMContext()
        context.add_message({"role": "user", "content": "ping"})
        stream = await llm.get_chat_completions(context)
        _ = [f async for f in stream]
        assert capture["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
        assert capture["stream"] is True
        assert capture["model"] == llm._settings.model


class TestUserTurnAggregation:
    @pytest_asyncio.fixture()
    async def user_with_task_manager(self):
        """Aggregator + every nested component wired with a TaskManager.

        LLMUserAggregator embeds a component tree (UserTurnController, stop
        strategies, idle/metrics controllers) that each keep their own
        _task_manager slot; PipelineTask would wire them at runtime, so the
        fixture walks the object graph and wires any None slot instead.
        """
        from pipecat.utils.asyncio.task_manager import TaskManager

        context = LLMContext()
        pair = LLMContextAggregatorPair(context, user_params=LLMUserAggregatorParams())
        user = pair.user()
        task_manager = TaskManager()
        seen = {id(user)}
        stack = [user]
        while stack:
            obj = stack.pop()
            if isinstance(obj, BaseObject) and obj._task_manager is None:
                obj._task_manager = task_manager
            for value in vars(obj).values():
                items = value if isinstance(value, (list, tuple)) else [value]
                for item in items:
                    if (
                        hasattr(item, "__dict__")
                        and not isinstance(item, (str, bytes, int, float, bool))
                        and id(item) not in seen
                    ):
                        seen.add(id(item))
                        stack.append(item)
        return user, context

    async def test_transcription_frames_build_user_context_messages(
        self, user_with_task_manager
    ):
        user, context = user_with_task_manager

        await user.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        await user.process_frame(
            TranscriptionFrame("hello there", "uid", 12345.0), FrameDirection.DOWNSTREAM
        )
        await user.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await user.process_frame(
            TranscriptionFrame("hello again", "uid", 12346.0), FrameDirection.DOWNSTREAM
        )
        await user.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await user.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)

        messages = context.get_messages()
        assert messages, "user turn must be aggregated into context messages"
        user_msgs = [m for m in messages if m.get("role") == "user"]
        assert user_msgs and any(
            "hello" in str(m.get("content")) for m in user_msgs
        ), f"aggregated content must include the transcription; got {messages}"

    async def test_empty_transcriptions_do_not_reach_context(
        self, user_with_task_manager
    ):
        user, context = user_with_task_manager

        await user.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        await user.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await user.process_frame(
            TranscriptionFrame("", "uid", 12345.0), FrameDirection.DOWNSTREAM
        )
        await user.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await user.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)

        messages = context.get_messages()
        user_msgs = [m for m in messages if m.get("role") == "user"]
        assert not user_msgs or all(
            m.get("content") != "" for m in user_msgs
        ), "empty transcription must not create a user message"
