"""Task 输入在应用服务层共用的校验规则。"""

from __future__ import annotations


class TaskPromptValidationError(ValueError):
    """用户提供的任务 Prompt 无效。"""


def validate_task_prompt(prompt: str) -> None:
    """校验 Task Prompt，同时保留原始文本供后续持久化。"""

    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if not prompt.strip():
        raise TaskPromptValidationError("prompt must not be blank")
