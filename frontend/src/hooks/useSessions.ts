import { useCallback, useEffect, useRef, useState } from "react";

import {
  getApiErrorMessage,
  isApiRequestAborted,
  listSessions,
} from "../api/client";
import type { SessionSummary } from "../api/contracts";

interface SessionsState {
  readonly sessions: SessionSummary[];
  readonly isLoading: boolean;
  readonly error: string | null;
  readonly refresh: () => Promise<void>;
}

export function useSessions(): SessionsState {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const activeRequest = useRef<AbortController | null>(null);

  const refresh = useCallback(async () => {
    activeRequest.current?.abort();
    const requestController = new AbortController();
    activeRequest.current = requestController;
    setIsLoading(true);
    setError(null);

    try {
      const response = await listSessions({
        signal: requestController.signal,
      });
      setSessions(response);
      setError(null);
    } catch (requestError: unknown) {
      if (!isApiRequestAborted(requestError)) {
        setError(getApiErrorMessage(requestError));
      }
    } finally {
      if (activeRequest.current === requestController) {
        activeRequest.current = null;
        setIsLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    void refresh();
    return () => activeRequest.current?.abort();
  }, [refresh]);

  return { sessions, isLoading, error, refresh };
}
