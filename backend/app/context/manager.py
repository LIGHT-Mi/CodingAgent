"""当前阶段只构造 System Prompt 和原始任务的最小上下文。"""

from __future__ import annotations

from app.db.persistence import PersistenceService
from app.llm.contracts import LLMContext, LLMMessage, LLMMessageRole


MINIMAL_SYSTEM_PROMPT = (
    "你是一个编程助手。请根据用户任务直接返回最终答案。"
    "当前没有任何工具可用，不要请求调用工具，也不要声称已经读取文件、"
    "执行命令或观察工作区内容。"
)


class ContextManagerError(RuntimeError):
    """ContextManager 无法为模型构造上下文。"""


class ContextTaskNotFoundError(ContextManagerError):
    """构造上下文所需的 Task 不存在。"""


class ContextManager:
    """从持久化 Task 构造当前阶段的两条消息上下文。"""

    def __init__(self, persistence: PersistenceService) -> None:
        if not isinstance(persistence, PersistenceService):
            raise TypeError("persistence must be a PersistenceService")
        self._persistence = persistence

    def build(self, task_id: str) -> LLMContext:
        """读取 Task 原始需求并返回固定 SYSTEM、USER 上下文。"""

        task = self._persistence.get_task(task_id)
        if task is None:
            raise ContextTaskNotFoundError(f"Task {task_id} was not found")
        return LLMContext(
            messages=(
                LLMMessage(
                    role=LLMMessageRole.SYSTEM,
                    content=MINIMAL_SYSTEM_PROMPT,
                ),
                LLMMessage(
                    role=LLMMessageRole.USER,
                    content=task.original_prompt,
                ),
            )
        )
