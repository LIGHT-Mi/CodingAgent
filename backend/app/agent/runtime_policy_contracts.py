"""正式运行策略使用的配置与当前运行状态契约。"""

from __future__ import annotations

import math
from dataclasses import dataclass


DEFAULT_MAX_AGENT_STEPS = 8
DEFAULT_MAX_LLM_RETRIES = 2
DEFAULT_LLM_RETRY_BASE_SECONDS = 1.0
DEFAULT_LLM_RETRY_MAX_SECONDS = 8.0
DEFAULT_AGENT_LOOP_REPEAT_THRESHOLD = 3


@dataclass(frozen=True, slots=True)
class RuntimePolicyConfig:
    """运行策略的固定限制；重试次数不包含首次 LLM 调用。"""

    max_agent_steps: int = DEFAULT_MAX_AGENT_STEPS
    max_llm_retries: int = DEFAULT_MAX_LLM_RETRIES
    retry_base_seconds: float = DEFAULT_LLM_RETRY_BASE_SECONDS
    retry_max_seconds: float = DEFAULT_LLM_RETRY_MAX_SECONDS
    loop_repeat_threshold: int = DEFAULT_AGENT_LOOP_REPEAT_THRESHOLD

    def __post_init__(self) -> None:
        _require_positive_integer(self.max_agent_steps, "max_agent_steps")
        _require_non_negative_integer(
            self.max_llm_retries,
            "max_llm_retries",
        )
        retry_base_seconds = _normalize_non_negative_number(
            self.retry_base_seconds,
            "retry_base_seconds",
        )
        retry_max_seconds = _normalize_non_negative_number(
            self.retry_max_seconds,
            "retry_max_seconds",
        )
        if retry_max_seconds < retry_base_seconds:
            raise ValueError(
                "retry_max_seconds must be greater than or equal to "
                "retry_base_seconds"
            )
        if (
            isinstance(self.loop_repeat_threshold, bool)
            or not isinstance(self.loop_repeat_threshold, int)
        ):
            raise TypeError("loop_repeat_threshold must be an integer")
        if self.loop_repeat_threshold < 2:
            raise ValueError(
                "loop_repeat_threshold must be greater than or equal to 2"
            )

        object.__setattr__(self, "retry_base_seconds", retry_base_seconds)
        object.__setattr__(self, "retry_max_seconds", retry_max_seconds)


@dataclass(frozen=True, slots=True)
class RuntimeState:
    """AgentRuntime 传给运行策略的供应商无关状态快照。"""

    next_step_number: int = 0
    llm_retry_count: int = 0
    cancel_requested: bool = False
    last_loop_fingerprint: str | None = None
    consecutive_loop_count: int = 0

    def __post_init__(self) -> None:
        _require_non_negative_integer(
            self.next_step_number,
            "next_step_number",
        )
        _require_non_negative_integer(
            self.llm_retry_count,
            "llm_retry_count",
        )
        if not isinstance(self.cancel_requested, bool):
            raise TypeError("cancel_requested must be a boolean")
        _require_non_negative_integer(
            self.consecutive_loop_count,
            "consecutive_loop_count",
        )

        if self.last_loop_fingerprint is None:
            if self.consecutive_loop_count != 0:
                raise ValueError(
                    "consecutive_loop_count must be zero when "
                    "last_loop_fingerprint is None"
                )
            return

        if not isinstance(self.last_loop_fingerprint, str):
            raise TypeError("last_loop_fingerprint must be a string or None")
        if not self.last_loop_fingerprint.strip():
            raise ValueError("last_loop_fingerprint must not be blank")
        if self.consecutive_loop_count == 0:
            raise ValueError(
                "consecutive_loop_count must be greater than zero when "
                "last_loop_fingerprint is present"
            )


def _require_positive_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero")


def _require_non_negative_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(
            f"{field_name} must be greater than or equal to zero"
        )


def _normalize_non_negative_number(
    value: float,
    field_name: str,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    if normalized < 0:
        raise ValueError(
            f"{field_name} must be greater than or equal to zero"
        )
    return normalized
