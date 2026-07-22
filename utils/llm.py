import json
import os
import time


class LLMClient:
    """Unified LLM client for OpenAI-compatible APIs and AWS Bedrock (Anthropic models)."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        aws_region: str = "us-east-1",
        llm_params: dict | None = None,
        min_completion_tokens: int | None = None,
    ):
        # Secrets/endpoint — keep these in env (export.sh). Each falls back to its env var when None.
        self._api_key = api_key
        self._base_url = base_url
        self._aws_region = aws_region
        self._openai_client = None
        self._bedrock_client = None
        # All LLM tuning lives in this one passthrough dict, merged verbatim into the OpenAI call.
        # It carries sampling params (temperature, top_p, ...) AND any model-specific extras, e.g.
        # per-model thinking control via extra_body.chat_template_kwargs:
        #   Kimi-K2.6 instant: {"extra_body": {"chat_template_kwargs": {"thinking": false}}}
        #   GLM-5.2 no-think:  {"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}
        # Add/replace anything via config — no code change needed.
        self._llm_params = dict(llm_params or {})
        # The only knob that can't be a passthrough param (it's max() logic, not an API field):
        # an optional floor on max_completion_tokens so a small per-call cap doesn't get eaten by
        # the model's reasoning → empty content. None = no floor.
        self._min_completion_tokens = int(min_completion_tokens) if min_completion_tokens is not None else None

    @classmethod
    def from_config(cls, cfg, api_key: str | None = None, base_url: str | None = None,
                    extra_llm_params: dict | None = None) -> "LLMClient":
        """Build from a config object (PipelineConfig / EvalConfig). LLM tuning flows from cfg;
        api_key/base_url may be overridden (eval passes per-role orch_/sub_ values).
        extra_llm_params merge OVER cfg.llm_params (e.g. cfg.think_llm_params for env steps)."""
        return cls(
            api_key=api_key if api_key is not None else getattr(cfg, "api_key", None),
            base_url=base_url if base_url is not None else getattr(cfg, "base_url", None),
            aws_region=getattr(cfg, "aws_region", "us-east-1"),
            llm_params={**(getattr(cfg, "llm_params", None) or {}), **(extra_llm_params or {})},
            min_completion_tokens=getattr(cfg, "min_completion_tokens", None),
        )

    @staticmethod
    def _is_bedrock(model: str) -> bool:
        return model.startswith(("us.anthropic.", "eu.anthropic.", "ap.anthropic.", "anthropic."))

    def _floor_max_tokens(self, max_tokens: int) -> int:
        if self._min_completion_tokens is None:
            return max_tokens
        return max(max_tokens, self._min_completion_tokens)

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
            import boto3.session
            from botocore.config import Config
            # A FRESH Session per rebuild is load-bearing: boto3.client() would reuse the
            # module-level default session, whose credential chain caches the (possibly
            # expired) creds forever — a new Session re-reads ~/.aws/credentials.
            self._bedrock_client = boto3.session.Session().client(
                "bedrock-runtime",
                region_name=self._aws_region,
                config=Config(read_timeout=1200, connect_timeout=10),
            )
        return self._bedrock_client

    # Credential errors raised when the ada-managed session token in ~/.aws/credentials
    # has rotated/expired under a long-lived client. Recreating the client re-reads the file.
    _CRED_ERROR_CODES = {"ExpiredToken", "ExpiredTokenException", "UnrecognizedClientException", "InvalidClientTokenId"}

    def _invoke_bedrock(self, model: str, body: dict) -> dict:
        """invoke_model with credential-refresh retries: on an expired/invalid token,
        drop the cached client (forces a re-read of ~/.aws/credentials, which ada keeps
        fresh) and retry, waiting up to ~5 min for the refresher to land new creds."""
        from botocore.exceptions import ClientError, NoCredentialsError
        attempts = 10
        for attempt in range(attempts):
            try:
                resp = self._bedrock().invoke_model(modelId=model, body=json.dumps(body))
                return json.loads(resp["body"].read())
            except (ClientError, NoCredentialsError) as e:
                code = e.response["Error"]["Code"] if isinstance(e, ClientError) else "NoCredentials"
                if code not in self._CRED_ERROR_CODES and not isinstance(e, NoCredentialsError):
                    raise
                if attempt == attempts - 1:
                    raise
                from loguru import logger
                logger.warning(f"Bedrock credential error ({code}), rebuilding client and retrying "
                               f"(attempt {attempt + 1}/{attempts})")
                self._bedrock_client = None  # rebuild → re-read credentials file
                time.sleep(min(30, 2 ** attempt))

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
        kwargs: dict = {**self._llm_params,            # config passthrough (temperature, top_p, extra_body, ...)
                        "model": model, "messages": messages,
                        "max_completion_tokens": self._floor_max_tokens(max_tokens)}
        # llm_params.temperature (the more specific config) wins; else fall back to the caller's arg
        # (e.g. eval's cfg.temperature). So setting temperature in llm_params is never silently shadowed.
        if temperature is not None and "temperature" not in self._llm_params:
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
            "max_tokens": self._floor_max_tokens(max_tokens),
            "messages": filtered,
        }
        if system:
            body["system"] = system
        # Config passthrough, same contract as the OpenAI path: llm_params merge verbatim
        # into the request body (temperature, top_k, thinking, ...), so e.g. extended
        # thinking is enabled purely via config:
        #   orch_llm_params: {temperature: 1.0, thinking: {type: enabled, budget_tokens: N}}
        # ("extra_body" is an OpenAI-client envelope — skip it here.)
        body.update({k: v for k, v in self._llm_params.items() if k != "extra_body"})
        if temperature is not None and "temperature" not in self._llm_params:
            body["temperature"] = temperature
        parsed = self._invoke_bedrock(model, body)
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
            **{**self._llm_params,                     # config passthrough (temperature, top_p, extra_body, ...)
               "model": model, "messages": messages,
               "max_completion_tokens": self._floor_max_tokens(max_tokens)})
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
            "max_tokens": self._floor_max_tokens(max_tokens),
            "messages": filtered,
        }
        if system:
            body["system"] = system
        # Same config passthrough as _complete_bedrock_usage: llm_params merge verbatim
        # into the request body (thinking, output_config, temperature, ...);
        # "extra_body" is an OpenAI-client envelope — skip it here.
        body.update({k: v for k, v in self._llm_params.items() if k != "extra_body"})

        parsed = self._invoke_bedrock(model, body)
        # With thinking enabled the first block may be a thinking block — join text blocks only.
        return "".join(b.get("text", "") for b in parsed.get("content", []) if b.get("type") == "text")
