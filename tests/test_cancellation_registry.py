import threading
import unittest

from app.agent import CancellationToken
from app.application import (
    CancellationRequestStatus,
    CancellationTokenAlreadyRegisteredError,
    CancellationTokenRegistry,
)


class CancellationTokenRegistryTests(unittest.TestCase):
    def test_register_request_cancel_and_unregister(self) -> None:
        registry = CancellationTokenRegistry()
        token = CancellationToken()

        registry.register("task-1", token)

        self.assertEqual(registry.registered_task_ids, ("task-1",))
        self.assertEqual(
            registry.request_cancel("task-1", "user cancelled"),
            CancellationRequestStatus.REQUESTED,
        )
        self.assertTrue(token.is_cancelled())
        self.assertEqual(token.reason, "user cancelled")
        self.assertEqual(
            registry.request_cancel("task-1", "later request"),
            CancellationRequestStatus.ALREADY_REQUESTED,
        )
        self.assertEqual(token.reason, "user cancelled")
        self.assertTrue(registry.unregister("task-1", token))
        self.assertEqual(registry.registered_task_ids, ())
        self.assertEqual(
            registry.request_cancel("task-1", "after completion"),
            CancellationRequestStatus.NOT_REGISTERED,
        )

    def test_duplicate_registration_does_not_replace_original_token(self) -> None:
        registry = CancellationTokenRegistry()
        original = CancellationToken()
        replacement = CancellationToken()
        registry.register("task-1", original)

        with self.assertRaises(CancellationTokenAlreadyRegisteredError):
            registry.register("task-1", replacement)

        registry.request_cancel("task-1", "cancel original")
        self.assertTrue(original.is_cancelled())
        self.assertFalse(replacement.is_cancelled())

    def test_stale_unregister_cannot_remove_current_token(self) -> None:
        registry = CancellationTokenRegistry()
        current = CancellationToken()
        stale = CancellationToken()
        registry.register("task-1", current)

        self.assertFalse(registry.unregister("task-1", stale))
        self.assertEqual(registry.registered_task_ids, ("task-1",))
        self.assertTrue(registry.unregister("task-1", current))

    def test_concurrent_cancel_requests_have_one_first_request(self) -> None:
        registry = CancellationTokenRegistry()
        token = CancellationToken()
        registry.register("task-1", token)
        start = threading.Barrier(9)
        statuses: list[CancellationRequestStatus] = []
        statuses_lock = threading.Lock()

        def request_cancel(index: int) -> None:
            start.wait()
            status = registry.request_cancel("task-1", f"request-{index}")
            with statuses_lock:
                statuses.append(status)

        threads = [
            threading.Thread(target=request_cancel, args=(index,))
            for index in range(8)
        ]
        for thread in threads:
            thread.start()
        start.wait()
        for thread in threads:
            thread.join(timeout=2)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(
            statuses.count(CancellationRequestStatus.REQUESTED),
            1,
        )
        self.assertEqual(
            statuses.count(CancellationRequestStatus.ALREADY_REQUESTED),
            7,
        )


if __name__ == "__main__":
    unittest.main()
