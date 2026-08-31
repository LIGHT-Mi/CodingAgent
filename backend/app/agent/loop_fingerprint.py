"""为连续重复的已完成工具交互生成稳定指纹。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from app.agent.contracts import ToolCallRequest, ToolResult


RUN_COMMAND_TOOL_NAME = "run_command"


def build_loop_fingerprint(
    tool_calls: Sequence[ToolCallRequest],
    tool_results: Sequence[ToolResult],
) -> str:
    """按 call_index 为一轮完整且已持久化的工具交互计算 SHA-256。"""

    calls = tuple(tool_calls)
    results = tuple(tool_results)
    if not calls:
        raise ValueError("tool_calls must contain at least one request")
    if any(not isinstance(call, ToolCallRequest) for call in calls):
        raise TypeError("tool_calls must contain only ToolCallRequest values")
    if any(not isinstance(result, ToolResult) for result in results):
        raise TypeError("tool_results must contain only ToolResult values")
    call_ids = [call.tool_call_id for call in calls]
    if len(call_ids) != len(set(call_ids)):
        raise ValueError("tool_calls must have unique tool_call_id values")
    call_indexes = [call.call_index for call in calls]
    if len(call_indexes) != len(set(call_indexes)):
        raise ValueError("tool_calls must have unique call_index values")

    results_by_id: dict[str, ToolResult] = {}
    for result in results:
        if result.tool_call_id in results_by_id:
            raise ValueError(
                "tool_results must contain exactly one result per tool call"
            )
        results_by_id[result.tool_call_id] = result

    ordered_calls = tuple(sorted(calls, key=lambda call: call.call_index))
    if set(results_by_id) != set(call_ids):
        raise ValueError(
            "tool_results must contain exactly one result for every tool call"
        )

    interactions: list[dict[str, object]] = []
    for call in ordered_calls:
        result = results_by_id[call.tool_call_id]
        if result.tool_name != call.tool_name:
            raise ValueError("ToolResult tool_name must match its ToolCallRequest")
        interactions.append(
            {
                "tool_name": call.tool_name,
                "arguments": _normalize_arguments(call),
                "result": _stable_result_summary(result),
            }
        )

    serialized = json.dumps(
        interactions,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(
        serialized.encode("utf-8", errors="surrogatepass")
    ).hexdigest()


def _normalize_arguments(call: ToolCallRequest) -> str:
    try:
        return json.dumps(
            call.arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"arguments for tool {call.tool_name!r} are not JSON serializable"
        ) from exc


def _stable_result_summary(result: ToolResult) -> dict[str, object]:
    if result.tool_name != RUN_COMMAND_TOOL_NAME:
        return {
            "status": result.status.value,
            "content": result.content,
            "error": result.error,
        }

    metadata = result.metadata
    return {
        "status": result.status.value,
        "exit_code": metadata.get("exit_code"),
        "stdout": metadata.get("stdout"),
        "stderr": metadata.get("stderr"),
        "timeout": metadata.get("timeout"),
        "error": result.error,
    }
