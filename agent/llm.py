"""
agent/llm.py

Unified LLM client interface configured for Moonshot AI (Kimi)
via the OpenAI-compatible API.
"""

from typing import List, Dict, Optional, Any, Iterable
import logging
import time
import re
from openai import OpenAI, APIError, APITimeoutError, AuthenticationError, RateLimitError
from openai.types.chat import ChatCompletionMessageParam

from .config import AgentConfig, get_config

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Client for interacting with Kimi (Moonshot AI) or any OpenAI-compatible endpoint.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or get_config()
        self._client: Optional[OpenAI] = None

        if self.config.validate():
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=120.0,  # 2-minute hard timeout — prevents indefinite hang
            )
        else:
            logger.warning(
                "No valid Kimi API key found. LLM calls will fail until "
                "KIMI_API_KEY is configured in .env"
            )

    @property
    def is_configured(self) -> bool:
        """Returns True if a valid API key is present."""
        return self._client is not None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Sends a single prompt with an optional system instruction and returns the text response.
        """
        messages: List[ChatCompletionMessageParam] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def chat(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Executes a multi-turn chat completion using the Kimi model.
        """
        if not self._client:
            raise RuntimeError(
                "Kimi LLM client is not configured. Please set KIMI_API_KEY in your .env file."
            )

        temp = temperature if temperature is not None else self.config.temperature
        # kimi-k2.7-code strictly requires temperature=1.0
        if "kimi-k2.7-code" in self.config.model:
            temp = 1.0
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        for attempt_idx in range(4):
            try:
                response = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=tokens,
                    timeout=120.0,  # 2-minute per-request timeout
                )
                choice = response.choices[0]
                content = choice.message.content or ""

                if getattr(choice, "finish_reason", None) == "length":
                    logger.warning(
                        "Kimi response was truncated due to max_tokens limit. "
                        f"Partial content length: {len(content)} chars. "
                        "Consider raising MAX_TOKENS in your .env file."
                    )
                    # Auto-close an unclosed markdown code fence so _extract_clean_code
                    # can recover the partial code block instead of returning empty.
                    if re.search(r"```(?:python[0-9]?|py)?\s*\n", content, re.IGNORECASE) and not content.rstrip().endswith("```"):
                        content = content.rstrip() + "\n```"
                        logger.warning("Auto-closed truncated code fence in Kimi response.")

                # Fallback: if content is empty but reasoning_content contains code
                if not content.strip() and hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
                    reasoning = choice.message.reasoning_content
                    pattern = r"```(?:python[0-9]?|py)?\s*\n?(.*?)\s*```"
                    matches = re.findall(pattern, reasoning, re.DOTALL | re.IGNORECASE)
                    valid_matches = [m.strip() for m in matches if m.strip()]
                    if valid_matches:
                        longest = max(valid_matches, key=len)
                        logger.warning("Extracted code block from model reasoning_content fallback.")
                        return f"```python\n{longest}\n```"

                return content or ""

            except APITimeoutError:
                if attempt_idx < 3:
                    logger.warning(f"Kimi API timed out (attempt {attempt_idx + 1}/4). Retrying...")
                    time.sleep(2.0)
                    continue
                raise RuntimeError("Kimi API timed out after 120s on all 4 attempts.")

            except AuthenticationError as e:
                logger.error("Authentication failed: Check your KIMI_API_KEY in .env.")
                raise RuntimeError(f"Kimi API authentication error: {e}") from e

            except RateLimitError as e:
                if attempt_idx < 3:
                    sleep_time = 2.0 * (attempt_idx + 1)
                    logger.warning(f"Kimi rate limit reached (RPM limit). Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                    continue
                logger.error("Rate limit reached on Kimi API.")
                raise RuntimeError(f"Kimi API rate limit exceeded: {e}") from e

            except APIError as e:
                logger.error(f"Kimi API returned an error: {e}")
                raise RuntimeError(f"Kimi API error: {e}") from e

        return ""
