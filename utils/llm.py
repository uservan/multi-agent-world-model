import json
import os


class LLMClient:
    """Unified LLM client for OpenAI-compatible APIs and AWS Bedrock (Anthropic models)."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        aws_region: str = "us-east-1",
    ):
        self._api_key = api_key
        self._base_url = base_url
        self._aws_region = aws_region
        self._openai_client = None
        self._bedrock_client = None

    @staticmethod
    def _is_bedrock(model: str) -> bool:
        return model.startswith(("us.anthropic.", "eu.anthropic.", "ap.anthropic.", "anthropic."))

    def _openai(self):
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI(
                api_key=self._api_key or os.environ.get("OPENAI_API_KEY"),
                base_url=self._base_url or os.environ.get("OPENAI_BASE_URL") or None,
            )
        return self._openai_client

    def _bedrock(self):
        if self._bedrock_client is None:
            import boto3
            from botocore.config import Config
            self._bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name=self._aws_region,
                config=Config(read_timeout=600, connect_timeout=10),
            )
        return self._bedrock_client

    def complete(self, model: str, messages: list[dict], max_tokens: int = 16384) -> str:
        if self._is_bedrock(model):
            return self._complete_bedrock(model, messages, max_tokens)
        return self._complete_openai(model, messages, max_tokens)

    def complete_with_usage(
        self, model: str, messages: list[dict], max_tokens: int = 16384, temperature: float | None = None
    ) -> tuple[str, dict]:
        """Like complete(), but also returns token usage: (text, {"in": int, "out": int})."""
        if self._is_bedrock(model):
            return self._complete_bedrock_usage(model, messages, max_tokens, temperature)
        return self._complete_openai_usage(model, messages, max_tokens, temperature)

    def _complete_openai_usage(
        self, model: str, messages: list[dict], max_tokens: int, temperature: float | None
    ) -> tuple[str, dict]:
        kwargs: dict = {"model": model, "messages": messages, "max_completion_tokens": max_tokens}
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = self._openai().chat.completions.create(**kwargs)
        msg = response.choices[0].message
        text = msg.content or ""               # clean content (server's reasoning_parser splits <think> out)
        usage = getattr(response, "usage", None)
        tokens = {
            "in": getattr(usage, "prompt_tokens", 0) or 0,
            "out": getattr(usage, "completion_tokens", 0) or 0,
        }
        reasoning = getattr(msg, "reasoning_content", None) or ""
        if reasoning:                          # carry reasoning out-of-band (recorded, never fed back)
            tokens["reasoning"] = reasoning
        return text, tokens

    def _complete_bedrock_usage(
        self, model: str, messages: list[dict], max_tokens: int, temperature: float | None
    ) -> tuple[str, dict]:
        system = None
        filtered = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                filtered.append(msg)
        body: dict = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": filtered,
        }
        if system:
            body["system"] = system
        if temperature is not None:
            body["temperature"] = temperature
        resp = self._bedrock().invoke_model(modelId=model, body=json.dumps(body))
        parsed = json.loads(resp["body"].read())
        blocks = parsed.get("content", [])
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        usage = parsed.get("usage", {})
        tokens = {"in": usage.get("input_tokens", 0), "out": usage.get("output_tokens", 0)}
        reasoning = "".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking")
        if reasoning:                          # carry reasoning out-of-band (recorded, never fed back)
            tokens["reasoning"] = reasoning
        return text, tokens

    def _complete_openai(self, model: str, messages: list[dict], max_tokens: int) -> str:
        response = self._openai().chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def _complete_bedrock(self, model: str, messages: list[dict], max_tokens: int) -> str:
        # Anthropic API separates system prompt from user/assistant messages
        system = None
        filtered = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                filtered.append(msg)

        body: dict = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": filtered,
        }
        if system:
            body["system"] = system

        resp = self._bedrock().invoke_model(modelId=model, body=json.dumps(body))
        return json.loads(resp["body"].read())["content"][0]["text"]
