import unittest
from unittest.mock import Mock

from app.agent import RetryWaiter


class RetryWaiterTests(unittest.TestCase):
    def test_injected_wait_function_receives_normalized_seconds(self) -> None:
        wait_function = Mock()
        waiter = RetryWaiter(wait_function)

        cancelled = waiter.wait(2)

        self.assertFalse(cancelled)
        wait_function.assert_called_once_with(2.0)

    def test_cancellation_handle_replaces_sleep(self) -> None:
        wait_function = Mock()
        cancellation_token = Mock()
        cancellation_token.wait.return_value = True
        waiter = RetryWaiter(wait_function)

        cancelled = waiter.wait(4, cancellation_token)

        self.assertTrue(cancelled)
        cancellation_token.wait.assert_called_once_with(4.0)
        wait_function.assert_not_called()

    def test_rejects_invalid_wait_values_and_dependencies(self) -> None:
        with self.assertRaises(TypeError):
            RetryWaiter(object())  # type: ignore[arg-type]

        waiter = RetryWaiter(lambda seconds: None)
        for invalid_seconds in (True, -1, float("inf"), float("nan")):
            with self.subTest(invalid_seconds=invalid_seconds):
                expected_error = (
                    TypeError if invalid_seconds is True else ValueError
                )
                with self.assertRaises(expected_error):
                    waiter.wait(invalid_seconds)

    def test_cancellation_handle_must_return_boolean(self) -> None:
        cancellation_token = Mock()
        cancellation_token.wait.return_value = None

        with self.assertRaises(TypeError):
            RetryWaiter(lambda seconds: None).wait(1, cancellation_token)


if __name__ == "__main__":
    unittest.main()
