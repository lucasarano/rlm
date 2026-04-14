import asyncio
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from email.utils import parsedate_to_datetime
from threading import Semaphore
from typing import Any

import anthropic

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary


class AnthropicClient(BaseLM):
    """
    LM Client for running models with the Anthropic API.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str | None = None,
        max_tokens: int = 32768,
        rate_limit_retries: int = 3,
        max_concurrent_requests: int = 1,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.client = anthropic.Anthropic(api_key=api_key, timeout=self.timeout)
        self.async_client = anthropic.AsyncAnthropic(api_key=api_key, timeout=self.timeout)
        self.model_name = model_name
        self.max_tokens = max_tokens
        if rate_limit_retries < 0:
            raise ValueError("rate_limit_retries must be non-negative.")
        if max_concurrent_requests < 1:
            raise ValueError("max_concurrent_requests must be at least 1.")
        self.rate_limit_retries = rate_limit_retries
        self.request_semaphore = Semaphore(max_concurrent_requests)

        # Per-model usage tracking
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.model_total_tokens: dict[str, int] = defaultdict(int)

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        messages, system = self._prepare_messages(prompt)

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for Anthropic client.")

        kwargs = {"model": model, "max_tokens": self.max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system

        response = self._with_rate_limit_retries(lambda: self.client.messages.create(**kwargs))
        self._track_cost(response, model)
        return response.content[0].text

    def streaming_completion(
        self,
        prompt: str | list[dict[str, Any]],
        on_token: Callable[[str], None],
        model: str | None = None,
    ) -> str:
        """Stream completion with token-level callbacks. Returns full response."""
        messages, system = self._prepare_messages(prompt)

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for Anthropic client.")

        kwargs = {"model": model, "max_tokens": self.max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system

        tokens_emitted = False

        def token_cb(delta: str) -> None:
            nonlocal tokens_emitted
            tokens_emitted = True
            on_token(delta)

        response = self._with_rate_limit_retries(
            lambda: self._stream_response(kwargs, token_cb),
            should_retry=lambda _error: not tokens_emitted,
        )

        self._track_cost(response, model)
        return response.content[0].text

    async def acompletion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        messages, system = self._prepare_messages(prompt)

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for Anthropic client.")

        kwargs = {"model": model, "max_tokens": self.max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system

        response = await self._with_rate_limit_retries_async(
            lambda: self.async_client.messages.create(**kwargs)
        )
        self._track_cost(response, model)
        return response.content[0].text

    def _stream_response(
        self,
        kwargs: dict[str, Any],
        on_token: Callable[[str], None],
    ) -> anthropic.types.Message:
        with self.client.messages.stream(**kwargs) as stream:
            for delta in stream.text_stream:
                on_token(delta)
            return stream.get_final_message()

    def _with_rate_limit_retries(
        self,
        request_fn: Callable[[], Any],
        should_retry: Callable[[anthropic.APIStatusError], bool] | None = None,
    ):
        for attempt in range(self.rate_limit_retries + 1):
            try:
                with self._request_slot():
                    return request_fn()
            except anthropic.APIStatusError as e:
                if not self._is_retryable_status_error(e) or attempt == self.rate_limit_retries:
                    raise
                if should_retry is not None and not should_retry(e):
                    raise
                time.sleep(self._retry_after_seconds(e, attempt))

        raise RuntimeError("unreachable")

    async def _with_rate_limit_retries_async(self, request_fn: Callable[[], Any]):
        for attempt in range(self.rate_limit_retries + 1):
            try:
                await self._acquire_request_slot()
                try:
                    return await request_fn()
                finally:
                    self.request_semaphore.release()
            except anthropic.APIStatusError as e:
                if not self._is_retryable_status_error(e) or attempt == self.rate_limit_retries:
                    raise
                await asyncio.sleep(self._retry_after_seconds(e, attempt))

        raise RuntimeError("unreachable")

    @contextmanager
    def _request_slot(self):
        self.request_semaphore.acquire()
        try:
            yield
        finally:
            self.request_semaphore.release()

    async def _acquire_request_slot(self) -> None:
        await asyncio.to_thread(self.request_semaphore.acquire)

    def _is_retryable_status_error(self, error: anthropic.APIStatusError) -> bool:
        if isinstance(error, anthropic.RateLimitError):
            return True
        if error.status_code in {503, 529}:
            return True

        body = error.body
        if isinstance(body, dict):
            error_body = body.get("error")
            if isinstance(error_body, dict) and error_body.get("type") == "overloaded_error":
                return True

        return False

    def _retry_after_seconds(self, error: anthropic.APIStatusError, attempt: int) -> float:
        retry_after = error.response.headers.get("retry-after")
        if retry_after is None:
            return float(2**attempt)

        try:
            return max(0.0, float(retry_after))
        except ValueError:
            try:
                retry_datetime = parsedate_to_datetime(retry_after)
            except (TypeError, ValueError):
                return float(2**attempt)
            return max(0.0, retry_datetime.timestamp() - time.time())

    def _prepare_messages(
        self, prompt: str | list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Prepare messages and extract system prompt for Anthropic API."""
        system = None

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            # Extract system message if present (Anthropic handles system separately)
            messages = []
            for msg in prompt:
                if msg.get("role") == "system":
                    system = msg.get("content")
                else:
                    messages.append(msg)
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        return messages, system

    def _track_cost(self, response: anthropic.types.Message, model: str):
        self.model_call_counts[model] += 1
        self.model_input_tokens[model] += response.usage.input_tokens
        self.model_output_tokens[model] += response.usage.output_tokens
        self.model_total_tokens[model] += response.usage.input_tokens + response.usage.output_tokens

        # Track last call for handler to read
        self.last_prompt_tokens = response.usage.input_tokens
        self.last_completion_tokens = response.usage.output_tokens

    def get_usage_summary(self) -> UsageSummary:
        model_summaries = {}
        for model in self.model_call_counts:
            model_summaries[model] = ModelUsageSummary(
                total_calls=self.model_call_counts[model],
                total_input_tokens=self.model_input_tokens[model],
                total_output_tokens=self.model_output_tokens[model],
            )
        return UsageSummary(model_usage_summaries=model_summaries)

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=self.last_prompt_tokens,
            total_output_tokens=self.last_completion_tokens,
        )
