"""Session 展示标题的确定性生成规则。"""

from __future__ import annotations


SESSION_TITLE_PROMPT_CHARACTERS = 30
SESSION_TITLE_ELLIPSIS = "…"


def generate_session_title(prompt: str) -> str:
    """将第一条 Task Prompt 归一化为可持久化的 Session title。"""

    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    normalized_prompt = " ".join(prompt.split())
    if not normalized_prompt:
        raise ValueError("prompt must not be blank")
    if len(normalized_prompt) <= SESSION_TITLE_PROMPT_CHARACTERS:
        return normalized_prompt
    return (
        normalized_prompt[:SESSION_TITLE_PROMPT_CHARACTERS]
        + SESSION_TITLE_ELLIPSIS
    )
