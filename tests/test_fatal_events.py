import unittest

from app.agent import RuntimeEventType
from app.agent.fatal_events import (
    RuntimeFailureBoundary,
    build_fatal_runtime_event,
)
from app.context import ContextHistoryError
from app.db.persistence import PersistenceServiceError


class FatalRuntimeEventFactoryTests(unittest.TestCase):
    def test_context_history_error_becomes_state_corruption(self) -> None:
        event = build_fatal_runtime_event(
            ContextHistoryError("tool result has no source call"),
            RuntimeFailureBoundary.CONTEXT_MANAGER,
        )

        self.assertEqual(
            event.event_type,
            RuntimeEventType.AGENT_STATE_CORRUPTED,
        )
        self.assertEqual(event.source, "context_manager")
        self.assertEqual(event.message, "tool result has no source call")
        self.assertEqual(event.details["error_type"], "ContextHistoryError")

    def test_tool_boundary_error_becomes_fatal_tool_event(self) -> None:
        event = build_fatal_runtime_event(
            RuntimeError("executor crashed"),
            RuntimeFailureBoundary.TOOL_ROUTER,
        )

        self.assertEqual(event.event_type, RuntimeEventType.FATAL_TOOL_ERROR)
        self.assertEqual(event.source, "tool_router")

    def test_persistence_error_overrides_original_boundary(self) -> None:
        event = build_fatal_runtime_event(
            PersistenceServiceError("save tool result failed"),
            RuntimeFailureBoundary.TOOL_ROUTER,
        )

        self.assertEqual(event.event_type, RuntimeEventType.FATAL_SYSTEM_ERROR)
        self.assertEqual(event.source, "persistence_service")
        self.assertEqual(event.details["boundary"], "tool_router")

    def test_unexpected_runtime_error_becomes_fatal_system_event(self) -> None:
        event = build_fatal_runtime_event(
            RuntimeError(),
            RuntimeFailureBoundary.AGENT_RUNTIME,
        )

        self.assertEqual(event.event_type, RuntimeEventType.FATAL_SYSTEM_ERROR)
        self.assertEqual(event.source, "agent_runtime")
        self.assertEqual(
            event.message,
            "RuntimeError occurred without an error message",
        )

    def test_rejects_invalid_factory_inputs(self) -> None:
        with self.assertRaises(TypeError):
            build_fatal_runtime_event(  # type: ignore[arg-type]
                "not an exception",
                RuntimeFailureBoundary.AGENT_RUNTIME,
            )
        with self.assertRaises(TypeError):
            build_fatal_runtime_event(  # type: ignore[arg-type]
                RuntimeError("failure"),
                "agent_runtime",
            )


if __name__ == "__main__":
    unittest.main()
