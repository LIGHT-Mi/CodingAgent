import unittest
from dataclasses import FrozenInstanceError

from app.agent.contracts import (
    AgentResult,
    AgentStepStatus,
    FinalAction,
    InvalidAction,
    RuntimeDecision,
    RuntimeDecisionType,
    RuntimeEvent,
    RuntimeEventType,
    TaskStatus,
    ToolCallRequest,
    ToolCallsAction,
    ToolCallStatus,
    ToolResult,
    ToolResultStatus,
)


class StatusEnumTests(unittest.TestCase):
    def test_task_status_values_match_database_contract(self) -> None:
        self.assertEqual(
            {status.value for status in TaskStatus},
            {
                "PENDING",
                "RUNNING",
                "COMPLETED",
                "FAILED",
                "CANCELLED",
                "TERMINATED",
            },
        )

    def test_agent_step_status_values_match_database_contract(self) -> None:
        self.assertEqual(
            {status.value for status in AgentStepStatus},
            {"RUNNING", "COMPLETED", "FAILED", "INTERRUPTED"},
        )

    def test_tool_call_status_values_match_database_contract(self) -> None:
        self.assertEqual(
            {status.value for status in ToolCallStatus},
            {"PENDING", "RUNNING", "COMPLETED", "ERROR", "REJECTED", "TIMEOUT"},
        )


class AgentActionTests(unittest.TestCase):
    def test_construct_all_action_types(self) -> None:
        final_action = FinalAction(content="任务已经完成。")
        first_call = ToolCallRequest(
            tool_call_id="provider-call-1",
            tool_name="read_file",
            arguments={"path": "src/main.py"},
            call_index=0,
        )
        second_call = ToolCallRequest(
            tool_call_id="provider-call-2",
            tool_name="list_files",
            arguments={"path": "src"},
            call_index=1,
        )
        tool_action = ToolCallsAction(
            tool_calls=(first_call, second_call),
            content="我先查看相关文件。",
        )
        invalid_action = InvalidAction(
            reason="arguments 不是合法 JSON",
            raw_response={"arguments": "{"},
        )

        self.assertEqual(final_action.content, "任务已经完成。")
        self.assertEqual(tool_action.tool_calls[0].arguments["path"], "src/main.py")
        self.assertEqual(tool_action.tool_calls[1].call_index, 1)
        self.assertEqual(invalid_action.raw_response, {"arguments": "{"})

    def test_action_contracts_reject_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            FinalAction(content="  ")
        with self.assertRaises(ValueError):
            ToolCallsAction(tool_calls=())
        with self.assertRaises(ValueError):
            InvalidAction(reason="")

        duplicate_call = ToolCallRequest(
            tool_call_id="same-id",
            tool_name="read_file",
            arguments={},
        )
        with self.assertRaises(ValueError):
            ToolCallsAction(tool_calls=(duplicate_call, duplicate_call))

    def test_contracts_are_frozen_value_objects(self) -> None:
        action = FinalAction(content="完成")
        with self.assertRaises(FrozenInstanceError):
            action.content = "被修改"  # type: ignore[misc]


class ToolResultTests(unittest.TestCase):
    def test_construct_completed_command_observation_with_nonzero_exit_code(self) -> None:
        result = ToolResult(
            tool_call_id="command-1",
            tool_name="run_command",
            status=ToolResultStatus.COMPLETED,
            content="命令执行结束，测试未通过。",
            metadata={
                "exit_code": 1,
                "stdout": "1 failed",
                "stderr": "",
            },
        )

        self.assertEqual(result.status, ToolResultStatus.COMPLETED)
        self.assertEqual(result.metadata["exit_code"], 1)
        self.assertIsNone(result.error)

    def test_construct_all_unsuccessful_tool_results(self) -> None:
        for status in (
            ToolResultStatus.ERROR,
            ToolResultStatus.REJECTED,
            ToolResultStatus.TIMEOUT,
        ):
            with self.subTest(status=status):
                result = ToolResult(
                    tool_call_id=f"call-{status.value}",
                    tool_name="run_command",
                    status=status,
                    error=f"tool finished with {status.value}",
                )
                self.assertEqual(result.status, status)

    def test_tool_result_enforces_status_and_error_consistency(self) -> None:
        with self.assertRaises(ValueError):
            ToolResult(
                tool_call_id="call-1",
                tool_name="read_file",
                status=ToolResultStatus.ERROR,
            )
        with self.assertRaises(ValueError):
            ToolResult(
                tool_call_id="call-2",
                tool_name="read_file",
                status=ToolResultStatus.COMPLETED,
                error="不应存在",
            )


class RuntimeContractTests(unittest.TestCase):
    def test_construct_every_runtime_event_type(self) -> None:
        for event_type in RuntimeEventType:
            with self.subTest(event_type=event_type):
                event = RuntimeEvent(
                    event_type=event_type,
                    message=f"发生事件：{event_type.value}",
                    source="unit_test",
                    details={"attempt": 1},
                )
                self.assertEqual(event.event_type, event_type)

    def test_construct_every_runtime_decision(self) -> None:
        decisions = (
            RuntimeDecision(RuntimeDecisionType.CONTINUE),
            RuntimeDecision(
                RuntimeDecisionType.RETRY,
                reason="临时网络错误",
                retry_after_seconds=0.5,
            ),
            RuntimeDecision(RuntimeDecisionType.FAILED, reason="基础设施错误"),
            RuntimeDecision(RuntimeDecisionType.CANCELLED, reason="用户取消"),
            RuntimeDecision(RuntimeDecisionType.TERMINATED, reason="MAX_STEPS"),
        )

        self.assertEqual(
            {decision.decision for decision in decisions},
            set(RuntimeDecisionType),
        )

    def test_non_retry_decision_rejects_backoff(self) -> None:
        with self.assertRaises(ValueError):
            RuntimeDecision(
                RuntimeDecisionType.FAILED,
                reason="基础设施错误",
                retry_after_seconds=1,
            )

    def test_non_continue_decision_requires_reason(self) -> None:
        with self.assertRaises(ValueError):
            RuntimeDecision(RuntimeDecisionType.TERMINATED)


class AgentResultTests(unittest.TestCase):
    def test_construct_every_terminal_agent_result(self) -> None:
        results = (
            AgentResult(TaskStatus.COMPLETED, final_answer="任务完成"),
            AgentResult(TaskStatus.FAILED, error="模型调用失败"),
            AgentResult(TaskStatus.CANCELLED, termination_reason="用户取消"),
            AgentResult(TaskStatus.TERMINATED, termination_reason="MAX_STEPS"),
        )

        self.assertEqual(
            {result.status for result in results},
            {
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
                TaskStatus.TERMINATED,
            },
        )

    def test_agent_result_rejects_nonterminal_and_inconsistent_values(self) -> None:
        with self.assertRaises(ValueError):
            AgentResult(TaskStatus.RUNNING)
        with self.assertRaises(ValueError):
            AgentResult(TaskStatus.COMPLETED)
        with self.assertRaises(ValueError):
            AgentResult(TaskStatus.FAILED, error="失败", final_answer="不应存在")
        with self.assertRaises(ValueError):
            AgentResult(TaskStatus.CANCELLED)


if __name__ == "__main__":
    unittest.main()
