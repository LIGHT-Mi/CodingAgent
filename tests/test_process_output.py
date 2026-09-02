import io
import os
import threading
import unittest

from app.tools import (
    CollectedProcessOutput,
    ProcessOutputCollectionError,
    ProcessOutputCollector,
)


class ProcessOutputCollectorTests(unittest.TestCase):
    def test_collects_untruncated_stdout_and_stderr(self) -> None:
        collector = ProcessOutputCollector(64, read_chunk_bytes=3)
        collector.start(
            io.BytesIO("正常输出\n".encode("utf-8")),
            io.BytesIO(b"warning\n"),
        )

        output = collector.finish()

        self.assertIsInstance(output, CollectedProcessOutput)
        self.assertEqual(output.stdout.text, "正常输出\n")
        self.assertEqual(output.stderr.text, "warning\n")
        self.assertFalse(output.stdout.truncated)
        self.assertEqual(output.stdout.discarded_byte_count, 0)
        self.assertEqual(
            output.stdout.original_byte_count,
            len("正常输出\n".encode("utf-8")),
        )

    def test_truncates_with_both_beginning_and_end_and_explicit_marker(self) -> None:
        data = b"BEGIN" + (b"x" * 100) + b"END"
        collector = ProcessOutputCollector(20, read_chunk_bytes=7)
        collector.start(io.BytesIO(data), io.BytesIO())

        stream = collector.finish().stdout

        self.assertTrue(stream.truncated)
        self.assertEqual(stream.original_byte_count, len(data))
        self.assertEqual(stream.retained_byte_count, 20)
        self.assertEqual(
            stream.discarded_byte_count,
            len(data) - stream.retained_byte_count,
        )
        self.assertTrue(stream.text.startswith("BEGIN"))
        self.assertTrue(stream.text.endswith("END"))
        self.assertIn("stdout truncated", stream.text)
        self.assertIn(
            f"{stream.discarded_byte_count} bytes omitted",
            stream.text,
        )

    def test_invalid_utf8_is_replaced_instead_of_raising(self) -> None:
        collector = ProcessOutputCollector(32)
        collector.start(io.BytesIO(b"before\xffafter"), None)

        output = collector.finish()

        self.assertEqual(output.stdout.text, "before\ufffdafter")
        self.assertEqual(output.stderr.text, "")

    def test_concurrently_drains_large_stdout_and_stderr_pipes(self) -> None:
        stdout_read_fd, stdout_write_fd = os.pipe()
        stderr_read_fd, stderr_write_fd = os.pipe()
        stdout = os.fdopen(stdout_read_fd, "rb", buffering=0)
        stderr = os.fdopen(stderr_read_fd, "rb", buffering=0)
        payload = b"0123456789abcdef" * 16_384
        collector = ProcessOutputCollector(128)
        collector.start(stdout, stderr)

        def write_all(file_descriptor: int, data: bytes) -> None:
            try:
                view = memoryview(data)
                while view:
                    written = os.write(file_descriptor, view)
                    view = view[written:]
            finally:
                os.close(file_descriptor)

        stdout_writer = threading.Thread(
            target=write_all,
            args=(stdout_write_fd, b"stdout:" + payload),
        )
        stderr_writer = threading.Thread(
            target=write_all,
            args=(stderr_write_fd, b"stderr:" + payload),
        )
        stdout_writer.start()
        stderr_writer.start()
        stdout_writer.join(timeout=5)
        stderr_writer.join(timeout=5)

        try:
            self.assertFalse(stdout_writer.is_alive())
            self.assertFalse(stderr_writer.is_alive())
            output = collector.finish()
        finally:
            stdout.close()
            stderr.close()

        self.assertEqual(
            output.stdout.original_byte_count,
            len(b"stdout:" + payload),
        )
        self.assertEqual(
            output.stderr.original_byte_count,
            len(b"stderr:" + payload),
        )
        self.assertEqual(output.stdout.retained_byte_count, 128)
        self.assertEqual(output.stderr.retained_byte_count, 128)
        self.assertTrue(output.stdout.truncated)
        self.assertTrue(output.stderr.truncated)

    def test_finish_is_idempotent_and_lifecycle_is_validated(self) -> None:
        collector = ProcessOutputCollector(8)
        with self.assertRaises(RuntimeError):
            collector.finish()

        collector.start(io.BytesIO(b"value"), None)
        first = collector.finish()
        self.assertIs(collector.finish(), first)
        with self.assertRaises(RuntimeError):
            collector.start(io.BytesIO(), io.BytesIO())

    def test_stream_read_errors_are_propagated_after_both_threads_finish(self) -> None:
        class BrokenStream:
            def read(self, size: int) -> bytes:
                raise OSError("broken pipe reader")

        collector = ProcessOutputCollector(8)
        collector.start(BrokenStream(), io.BytesIO(b"stderr complete"))

        with self.assertRaises(ProcessOutputCollectionError) as raised:
            collector.finish()
        self.assertIn("stdout", str(raised.exception))
        self.assertIn("broken pipe reader", str(raised.exception))

    def test_rejects_limits_that_cannot_keep_head_and_tail(self) -> None:
        for limit in (0, 1, True, 1.5):
            with self.subTest(limit=limit):
                with self.assertRaises((TypeError, ValueError)):
                    ProcessOutputCollector(limit)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
