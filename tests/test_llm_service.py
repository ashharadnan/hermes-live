"""LlamacppLLMService tests: thinking-off flag rides extra_body, settings pin."""


import pytest

from config import CFG
from services.llamacpp_llm import LlamacppLLMService


@pytest.fixture()
def llm() -> LlamacppLLMService:
    return LlamacppLLMService()


class TestConstruction:
    def test_pins_endpoint_model_and_system_prompt_from_config(self, llm):
        assert llm._client is not None
        assert str(llm._client.base_url).rstrip("/") == CFG.llm.base_url.rstrip("/")
        assert llm._settings.model == CFG.llm.model
        assert llm._settings.system_instruction == CFG.llm.system_prompt

    def test_logs_thinking_flag_from_config(self, llm):
        # The startup log line pins the thinking state visibly at boot;
        # capture it through loguru (the sink receives the formatted string)
        # and assert the config values appear.
        from loguru import logger

        records: list[str] = []
        handler_id = logger.add(records.append, level="INFO")
        try:
            LlamacppLLMService()
        finally:
            logger.remove(handler_id)
        thinking_lines = [
            line
            for line in records
            if "thinking=" in line and "LlamacppLLMService" in line
        ]
        assert thinking_lines, (
            f"startup log must report the thinking flag; got {records}"
        )
        assert f"thinking={CFG.llm.enable_thinking}" in thinking_lines[0]
        assert f"model={CFG.llm.model}" in thinking_lines[0]


class TestChatCompletionParams:
    def test_thinking_off_flag_rides_extra_body(self, llm):
        params = llm.build_chat_completion_params({})
        extra = params["extra_body"]
        assert extra["chat_template_kwargs"]["enable_thinking"] is False

    def test_streaming_requested(self, llm):
        params = llm.build_chat_completion_params({})
        assert params["stream"] is True

    def test_model_matches_config(self, llm):
        params = llm.build_chat_completion_params({})
        assert params["model"] == CFG.llm.model

    def test_context_params_merged_through(self, llm):
        context_params = {"messages": [{"role": "user", "content": "hi"}]}
        params = llm.build_chat_completion_params(context_params)
        assert params["messages"] == context_params["messages"]

    def test_existing_extra_body_not_clobbered(self, llm):
        params_from_context = {"extra_body": {"cache_prompt": True}}
        params = llm.build_chat_completion_params(params_from_context)
        assert params["extra_body"]["cache_prompt"] is True
        assert "chat_template_kwargs" in params["extra_body"]

    def test_missing_extra_body_key_handled(self, llm):
        params = llm.build_chat_completion_params({})
        # The override must create extra_body when absent, not KeyError.
        assert params["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
