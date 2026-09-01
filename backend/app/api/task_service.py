"""创建并执行单次编程任务的应用服务。"""

from __future__ import annotations

from pathlib import Path

from app.agent.cancellation import CancellationToken
from app.agent.contracts import AgentResult
from app.agent.runtime import AgentRuntime
from app.api.session_title import generate_session_title
from app.api.task_validation import validate_task_prompt
from app.api.workspace import WorkspaceValidator
from app.db.persistence import PersistenceService


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

    def create_task(
        self,
        prompt: str,
        workspace: str | Path,
    ) -> str:
        """校验输入并创建 PENDING Task，不启动 Agent Runtime。"""

        validate_task_prompt(prompt)
        validated_workspace = self._workspace_validator.validate(workspace)

        _, task = self._persistence.create_session_with_task(
            title=generate_session_title(prompt),
            original_prompt=prompt,
            workspace=str(validated_workspace),
        )
        return task.id

    def execute_task(
        self,
        task_id: str,
        cancellation_token: CancellationToken | None = None,
    ) -> AgentResult:
        """启动已有 PENDING Task，执行 Agent 并持久化终态。"""

        if cancellation_token is not None and not isinstance(
            cancellation_token,
            CancellationToken,
        ):
            raise TypeError("cancellation_token must be a CancellationToken")
        self._persistence.start_task(task_id)

        result = self._agent_runtime.run(task_id, cancellation_token)
        self._persistence.finish_task(task_id, result)
        return result

    def run_task_and_wait(
        self,
        prompt: str,
        workspace: str | Path,
        cancellation_token: CancellationToken | None = None,
    ) -> AgentResult:
        """使用创建和执行两个边界同步完成一次任务。"""

        task_id = self.create_task(prompt, workspace)
        return self.execute_task(task_id, cancellation_token)
