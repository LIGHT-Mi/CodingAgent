import unittest

from app.agent import CancellationToken


class CancellationTokenTests(unittest.TestCase):
    def test_cancel_sets_first_reason_and_wakes_waiters(self) -> None:
        token = CancellationToken()

        self.assertFalse(token.is_cancelled())
        self.assertIsNone(token.reason)
        self.assertFalse(token.wait(0))

        token.cancel("  user requested stop  ")
        token.cancel("later reason")

        self.assertTrue(token.is_cancelled())
        self.assertEqual(token.reason, "user requested stop")
        self.assertTrue(token.wait(0))

    def test_rejects_invalid_reason_and_timeout(self) -> None:
        token = CancellationToken()

        for reason in (None, "", "   ", 1):
            with self.subTest(reason=reason):
                expected_error = TypeError if reason in (None, 1) else ValueError
                with self.assertRaises(expected_error):
                    token.cancel(reason)  # type: ignore[arg-type]

        for timeout in (True, -1, float("inf"), float("nan")):
            with self.subTest(timeout=timeout):
                expected_error = TypeError if timeout is True else ValueError
                with self.assertRaises(expected_error):
                    token.wait(timeout)


if __name__ == "__main__":
    unittest.main()
