"""解析任务文本并生成完成状态统计。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TaskSummary:
    completed: int
    pending: int

    @property
    def total(self) -> int:
        return self.completed + self.pending


def summarize_tasks(content: str) -> TaskSummary:
    """统计 ``[x]`` 与 ``[ ]`` 开头的非空任务行。"""

    task_lines = [line.strip() for line in content.splitlines() if line.strip()]
    completed = sum(line.startswith("[x]") for line in task_lines)
    pending = sum(line.startswith("[ ]") for line in task_lines)
    return TaskSummary(completed=completed, pending=pending)


def main() -> None:
    task_file = Path(__file__).parents[1] / "data" / "tasks.txt"
    summary = summarize_tasks(task_file.read_text(encoding="utf-8"))
    print(
        f"completed={summary.completed} "
        f"pending={summary.pending} total={summary.total}"
    )


if __name__ == "__main__":
    main()

