"""Local llama.cpp LLM service: OpenAILLMService pinned to the local llama-server.

Forces ``enable_thinking: false`` into every request via extra_body; the SDK
rejects unknown top-level kwargs, so nesting is the only correct channel.
Thinking-on costs ~5 s of silence before first speech (measured on this box).
"""

from loguru import logger
from pipecat.services.openai.llm import OpenAILLMService

from config import CFG


class LlamacppLLMService(OpenAILLMService):
    """OpenAILLMService pointed at llama-server, thinking off.

    Args:
        base_url: llama-server's OpenAI-compatible endpoint.
        model: Model name as llama-server registered it.
        system_instruction: Bot persona/system prompt.
    """

    def __init__(self, base_url: str = CFG.llm.base_url, model: str = CFG.llm.model,
                 system_instruction: str = CFG.llm.system_prompt, **kwargs) -> None:
        super().__init__(
            api_key=CFG.llm.api_key,
            base_url=base_url,
            settings=self.Settings(
                model=model,
                system_instruction=system_instruction,
            ),
            **kwargs,
        )
        logger.info(
            f"LlamacppLLMService -> {base_url} model={model}"
            f" thinking={CFG.llm.enable_thinking}"
        )

    def build_chat_completion_params(self, params_from_context: dict) -> dict:
        """Nest the thinking-off flag into extra_body on every request."""
        params = super().build_chat_completion_params(params_from_context)
        extra = dict(params.get("extra_body") or {})
        extra["chat_template_kwargs"] = {"enable_thinking": CFG.llm.enable_thinking}
        params["extra_body"] = extra
        return params
