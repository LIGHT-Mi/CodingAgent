import { TaskStatus } from "./contracts";

export function isActiveTaskStatus(status: TaskStatus | null): boolean {
  return status === TaskStatus.PENDING || status === TaskStatus.RUNNING;
}

export function isTerminalTaskStatus(status: TaskStatus | null): boolean {
  return (
    status === TaskStatus.COMPLETED ||
    status === TaskStatus.FAILED ||
    status === TaskStatus.CANCELLED ||
    status === TaskStatus.TERMINATED
  );
}
