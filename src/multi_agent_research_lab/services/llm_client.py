"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass

from multi_agent_research_lab.core.errors import StudentTodoError


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client skeleton."""

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion.

        Connects to OpenAI using the settings from get_settings().
        """
        from openai import OpenAI
        from multi_agent_research_lab.core.config import get_settings

        settings = get_settings()
        api_key = settings.openai_api_key
        model = settings.openai_model

        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            timeout=settings.timeout_seconds
        )

        content = response.choices[0].message.content or ""
        
        input_tokens = None
        output_tokens = None
        cost = None

        if response.usage:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            # Standard gpt-4o-mini pricing: $0.15/1M input, $0.60/1M output
            input_rate = 0.15 / 1_000_000
            output_rate = 0.60 / 1_000_000
            if "gpt-4o" in model and "mini" not in model:
                # Standard gpt-4o pricing: $2.50/1M input, $10.00/1M output
                input_rate = 2.50 / 1_000_000
                output_rate = 10.00 / 1_000_000
            cost = (input_tokens * input_rate) + (output_tokens * output_rate)

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost
        )

