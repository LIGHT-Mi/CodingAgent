import { useCallback, useEffect, useRef, useState } from "react";

import {
  getApiErrorMessage,
  getSessionTasks,
  isApiRequestAborted,
} from "../api/client";
import type { Task } from "../api/contracts";

interface SessionTasksState {
  readonly tasks: Task[];
  readonly loadedSessionId: string | null;
  readonly isLoading: boolean;
  readonly error: string | null;
  readonly refresh: () => Promise<void>;
}

interface StoredTasks {
  readonly sessionId: string;
  readonly values: Task[];
}

interface StoredError {
  readonly sessionId: string;
  readonly message: string;
}

export function useSessionTasks(
  sessionId: string | null,
): SessionTasksState {
  const [storedTasks, setStoredTasks] = useState<StoredTasks | null>(null);
  const [loadingSessionId, setLoadingSessionId] = useState<string | null>(null);
  const [storedError, setStoredError] = useState<StoredError | null>(null);
  const activeRequest = useRef<AbortController | null>(null);
  const visibleTasks =
    storedTasks?.sessionId === sessionId ? storedTasks.values : [];
  const visibleError =
    storedError?.sessionId === sessionId ? storedError.message : null;

  const refresh = useCallback(async () => {
    activeRequest.current?.abort();
    if (sessionId === null) {
      activeRequest.current = null;
      setLoadingSessionId(null);
      return;
    }

    const requestController = new AbortController();
    activeRequest.current = requestController;
    setLoadingSessionId(sessionId);
    setStoredError(null);
    try {
      const response = await getSessionTasks(sessionId, {
        signal: requestController.signal,
      });
      setStoredTasks({ sessionId, values: response });
    } catch (requestError: unknown) {
      if (!isApiRequestAborted(requestError)) {
        setStoredError({
          sessionId,
          message: getApiErrorMessage(requestError),
        });
      }
    } finally {
      if (activeRequest.current === requestController) {
        activeRequest.current = null;
        setLoadingSessionId(null);
      }
    }
  }, [sessionId]);

  useEffect(() => {
    void refresh();
    return () => activeRequest.current?.abort();
  }, [refresh]);

  return {
    tasks: visibleTasks,
    loadedSessionId:
      storedTasks?.sessionId === sessionId ? sessionId : null,
    isLoading: sessionId !== null && loadingSessionId === sessionId,
    error: visibleError,
    refresh,
  };
}
