"""创建并执行单次编程任务的应用服务。"""

from __future__ import annotations

from pathlib import Path

from app.agent.contracts import AgentResult
from app.agent.runtime import AgentRuntime
from app.api.workspace import WorkspaceValidator
from app.db.persistence import PersistenceService


class TaskPromptValidationError(ValueError):
    """用户提供的任务 Prompt 无效。"""


class TaskService:
    """管理 Session 和 Task 生命周期，并委托 AgentRuntime 执行任务。"""

    def __init__(
        self,
        persistence: PersistenceService,
        workspace_validator: WorkspaceValidator,
        agent_runtime: AgentRuntime,
    ) -> None:
        if not isinstance(persistence, PersistenceService):
            raise TypeError("persistence must be a PersistenceService")
        if not isinstance(workspace_validator, WorkspaceValidator):
            raise TypeError(
                "workspace_validator must be a WorkspaceValidator"
            )
        if not isinstance(agent_runtime, AgentRuntime):
            raise TypeError("agent_runtime must be an AgentRuntime")
        self._persistence = persistence
        self._workspace_validator = workspace_validator
        self._agent_runtime = agent_runtime

    def run(self, prompt: str, workspace: str | Path) -> AgentResult:
        """校验输入、创建新 Session 和 Task，并返回 Agent 最终结果。"""

        _validate_prompt(prompt)
        validated_workspace = self._workspace_validator.validate(workspace)

        coding_session = self._persistence.create_session()
        task = self._persistence.create_task(
            coding_session.id,
            original_prompt=prompt,
            workspace=str(validated_workspace),
        )
        self._persistence.start_task(task.id)

        result = self._agent_runtime.run(task.id)
        self._persistence.finish_task(task.id, result)
        return result


def _validate_prompt(prompt: str) -> None:
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if not prompt.strip():
        raise TaskPromptValidationError("prompt must not be blank")
