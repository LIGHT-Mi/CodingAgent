import { useEffect, useState } from "react";

import {
  ApiClientError,
  getApiErrorMessage,
  getTaskSnapshot,
  isApiRequestAborted,
} from "../api/client";
import type { TaskSnapshot } from "../api/contracts";
import { isTerminalTaskStatus } from "../api/taskStatus";

const SNAPSHOT_POLL_INTERVAL_MILLISECONDS = 1_500;

interface TaskSnapshotState {
  readonly snapshot: TaskSnapshot | null;
  readonly isLoading: boolean;
  readonly error: string | null;
  readonly notFound: boolean;
  readonly refresh: () => void;
}

interface StoredSnapshot {
  readonly taskId: string;
  readonly value: TaskSnapshot;
}

interface StoredError {
  readonly taskId: string;
  readonly message: string;
  readonly notFound: boolean;
}

export function useTaskSnapshot(taskId: string | null): TaskSnapshotState {
  const [storedSnapshot, setStoredSnapshot] =
    useState<StoredSnapshot | null>(null);
  const [storedError, setStoredError] = useState<StoredError | null>(null);
  const [refreshVersion, setRefreshVersion] = useState(0);
  const visibleSnapshot =
    storedSnapshot?.taskId === taskId ? storedSnapshot.value : null;
  const visibleError =
    storedError?.taskId === taskId ? storedError.message : null;

  useEffect(() => {
    if (taskId === null) {
      return;
    }

    const activeTaskId = taskId;
    const requestController = new AbortController();
    let nextPoll: number | null = null;

    async function pollSnapshot() {
      try {
        const response = await getTaskSnapshot(activeTaskId, {
          signal: requestController.signal,
        });
        setStoredSnapshot({ taskId: activeTaskId, value: response });
        setStoredError(null);
        if (!isTerminalTaskStatus(response.task.status)) {
          nextPoll = window.setTimeout(
            pollSnapshot,
            SNAPSHOT_POLL_INTERVAL_MILLISECONDS,
          );
        }
      } catch (requestError: unknown) {
        if (isApiRequestAborted(requestError)) {
          return;
        }
        const notFound =
          requestError instanceof ApiClientError &&
          requestError.status === 404;
        setStoredError({
          taskId: activeTaskId,
          message: notFound
            ? "任务不存在或已被删除。"
            : getApiErrorMessage(requestError),
          notFound,
        });
        if (notFound) {
          return;
        }
        nextPoll = window.setTimeout(
          pollSnapshot,
          SNAPSHOT_POLL_INTERVAL_MILLISECONDS,
        );
      }
    }

    void pollSnapshot();

    return () => {
      requestController.abort();
      if (nextPoll !== null) {
        window.clearTimeout(nextPoll);
      }
    };
  }, [refreshVersion, taskId]);

  return {
    snapshot: visibleSnapshot,
    isLoading:
      taskId !== null && visibleSnapshot === null && visibleError === null,
    error: visibleError,
    notFound:
      storedError?.taskId === taskId ? storedError.notFound : false,
    refresh: () => setRefreshVersion((current) => current + 1),
  };
}
