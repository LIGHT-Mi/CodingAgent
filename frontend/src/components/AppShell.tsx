import { useEffect, useMemo, useRef, useState } from "react";

import {
  getApiErrorMessage,
  getSession,
  isApiRequestAborted,
} from "../api/client";
import type {
  CreateSessionResponse,
  CreateSessionTaskResponse,
  SessionSummary,
  TaskStatus,
} from "../api/contracts";
import { CommandApprovalStatus as ApprovalStatus } from "../api/contracts";
import { isTerminalTaskStatus } from "../api/taskStatus";
import { usePersistentBoolean } from "../hooks/usePersistentBoolean";
import {
  isNarrowViewport,
  ResponsiveLayout,
  useResponsiveLayout,
} from "../hooks/useResponsiveLayout";
import { useSessionTasks } from "../hooks/useSessionTasks";
import { useSessions } from "../hooks/useSessions";
import { useTaskSnapshot } from "../hooks/useTaskSnapshot";
import { AgentInspector } from "./AgentInspector";
import { ChatWorkspace } from "./ChatWorkspace";
import { ConversationSidebar } from "./ConversationSidebar";
import type { TaskSubmission } from "./Composer";

const LEFT_SIDEBAR_STORAGE_KEY = "leftSidebarCollapsed";
const RIGHT_INSPECTOR_STORAGE_KEY = "rightInspectorCollapsed";

export interface ActiveConversation {
  readonly sessionId: string;
  readonly title: string;
  readonly taskId: string;
  readonly initialStatus: TaskStatus;
  readonly latestWorkspace: string;
}

export function AppShell() {
  const [activeConversation, setActiveConversation] =
    useState<ActiveConversation | null>(null);
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [pendingUrlTaskId, setPendingUrlTaskId] = useState<string | null>(null);
  const [isRestoringUrl, setIsRestoringUrl] = useState(true);
  const [urlRestoreError, setUrlRestoreError] = useState<string | null>(null);
  const responsiveLayout = useResponsiveLayout();
  const [mobileLeftDrawerOpen, setMobileLeftDrawerOpen] = useState(false);
  const [mobileRightDrawerOpen, setMobileRightDrawerOpen] = useState(false);
  const [leftSidebarCollapsed, toggleLeftSidebar] = usePersistentBoolean(
    LEFT_SIDEBAR_STORAGE_KEY,
    isNarrowViewport(),
  );
  const [rightInspectorCollapsed, toggleRightInspector] =
    usePersistentBoolean(RIGHT_INSPECTOR_STORAGE_KEY);
  const sessionList = useSessions();
  const sessionTasks = useSessionTasks(
    activeConversation?.sessionId ?? null,
  );
  const refreshSessions = sessionList.refresh;
  const refreshSessionTasks = sessionTasks.refresh;
  const latestSnapshotState = useTaskSnapshot(
    !isRestoringUrl ? activeConversation?.taskId ?? null : null,
  );
  const historicalSelectionId =
    !isRestoringUrl &&
    selectedTaskId !== null &&
    selectedTaskId !== activeConversation?.taskId
      ? selectedTaskId
      : null;
  const historicalSnapshotState = useTaskSnapshot(historicalSelectionId);
  const refreshedTerminalTaskId = useRef<string | null>(null);
  const isMobile = responsiveLayout === ResponsiveLayout.MOBILE;
  const leftPanelOpen = isMobile
    ? mobileLeftDrawerOpen
    : !leftSidebarCollapsed;
  const rightPanelOpen = isMobile
    ? mobileRightDrawerOpen
    : !rightInspectorCollapsed;
  const overlayPanelSide = isMobile
    ? mobileLeftDrawerOpen
      ? "left"
      : mobileRightDrawerOpen
        ? "right"
        : null
    : responsiveLayout === ResponsiveLayout.TABLET && leftPanelOpen
      ? "left"
      : null;

  const tasks = useMemo(() => {
    const currentSessionTasks = sessionTasks.tasks;
    const latestSnapshotTask = latestSnapshotState.snapshot?.task ?? null;
    if (latestSnapshotTask === null) {
      return currentSessionTasks;
    }
    const latestTaskIndex = currentSessionTasks.findIndex(
      (task) => task.id === latestSnapshotTask.id,
    );
    if (latestTaskIndex < 0) {
      return [...currentSessionTasks, latestSnapshotTask];
    }
    return currentSessionTasks.map((task, index) =>
      index === latestTaskIndex ? latestSnapshotTask : task,
    );
  }, [latestSnapshotState.snapshot, sessionTasks.tasks]);
  const latestTask =
    latestSnapshotState.snapshot?.task ??
    tasks.find((task) => task.id === activeConversation?.taskId) ??
    null;
  const selectedSnapshotState =
    selectedTaskId !== null && selectedTaskId === activeConversation?.taskId
      ? latestSnapshotState
      : historicalSnapshotState;
  const selectedTask =
    selectedSnapshotState.snapshot?.task ??
    tasks.find((task) => task.id === selectedTaskId) ??
    null;
  const pendingApprovalCount =
    latestSnapshotState.snapshot?.command_approvals.filter(
      (approval) => approval.status === ApprovalStatus.PENDING,
    ).length ?? 0;

  useEffect(() => {
    if (!isMobile) {
      setMobileLeftDrawerOpen(false);
      setMobileRightDrawerOpen(false);
    }
  }, [isMobile]);

  useEffect(() => {
    if (overlayPanelSide === null) {
      return;
    }

    const panelId =
      overlayPanelSide === "left"
        ? "conversation-sidebar"
        : "agent-inspector";
    const focusPanel = window.requestAnimationFrame(() => {
      document.getElementById(panelId)?.focus();
    });
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closeOverlayPanel(true);
        return;
      }
      if (event.key !== "Tab") {
        return;
      }

      const panel = document.getElementById(panelId);
      if (panel === null) {
        return;
      }
      const focusableElements = Array.from(
        panel.querySelectorAll<HTMLElement>(
          'button:not(:disabled), input:not(:disabled), textarea:not(:disabled), summary, [href], [tabindex]:not([tabindex="-1"])',
        ),
      ).filter((element) => element.getClientRects().length > 0);
      if (focusableElements.length === 0) {
        event.preventDefault();
        panel.focus();
        return;
      }

      const firstElement = focusableElements[0];
      const lastElement = focusableElements[focusableElements.length - 1];
      if (
        event.shiftKey &&
        (document.activeElement === firstElement ||
          document.activeElement === panel)
      ) {
        event.preventDefault();
        lastElement.focus();
      } else if (!event.shiftKey && document.activeElement === lastElement) {
        event.preventDefault();
        firstElement.focus();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      window.cancelAnimationFrame(focusPanel);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [overlayPanelSide]);

  useEffect(() => {
    const currentUrl = new URL(window.location.href);
    const sessionId = currentUrl.searchParams.get("session")?.trim() || null;
    const taskId = currentUrl.searchParams.get("task")?.trim() || null;

    if (sessionId === null) {
      if (taskId !== null) {
        setUrlRestoreError("URL 中包含 task，但缺少对应的 session。");
      }
      setIsRestoringUrl(false);
      return;
    }

    const requestedSessionId = sessionId;
    const requestController = new AbortController();
    async function restoreSession() {
      try {
        const session = await getSession(requestedSessionId, {
          signal: requestController.signal,
        });
        setActiveConversation(toActiveConversation(session));
        setPendingUrlTaskId(taskId ?? session.latest_task_id);
      } catch (requestError: unknown) {
        if (!isApiRequestAborted(requestError)) {
          setUrlRestoreError(getApiErrorMessage(requestError));
          setIsRestoringUrl(false);
        }
      }
    }
    void restoreSession();
    return () => requestController.abort();
  }, []);

  useEffect(() => {
    if (
      !isRestoringUrl ||
      pendingUrlTaskId === null ||
      activeConversation === null
    ) {
      return;
    }
    if (sessionTasks.error !== null) {
      setUrlRestoreError(sessionTasks.error);
      setPendingUrlTaskId(null);
      setIsRestoringUrl(false);
      return;
    }
    if (sessionTasks.loadedSessionId !== activeConversation.sessionId) {
      return;
    }

    const requestedTask = sessionTasks.tasks.find(
      (task) =>
        task.id === pendingUrlTaskId &&
        task.session_id === activeConversation.sessionId,
    );
    if (requestedTask === undefined) {
      setUrlRestoreError("URL 指定的 Task 不属于该 Session。");
    } else {
      setSelectedTaskId(requestedTask.id);
    }
    setPendingUrlTaskId(null);
    setIsRestoringUrl(false);
  }, [
    activeConversation,
    isRestoringUrl,
    pendingUrlTaskId,
    sessionTasks.error,
    sessionTasks.loadedSessionId,
    sessionTasks.tasks,
  ]);

  useEffect(() => {
    if (isRestoringUrl || urlRestoreError !== null) {
      return;
    }
    const currentUrl = new URL(window.location.href);
    if (activeConversation === null) {
      currentUrl.searchParams.delete("session");
      currentUrl.searchParams.delete("task");
    } else if (selectedTaskId !== null) {
      currentUrl.searchParams.set("session", activeConversation.sessionId);
      currentUrl.searchParams.set("task", selectedTaskId);
    } else {
      return;
    }
    window.history.replaceState(
      null,
      "",
      `${currentUrl.pathname}${currentUrl.search}${currentUrl.hash}`,
    );
  }, [activeConversation, isRestoringUrl, selectedTaskId, urlRestoreError]);

  useEffect(() => {
    if (
      latestTask === null ||
      !isTerminalTaskStatus(latestTask.status) ||
      refreshedTerminalTaskId.current === latestTask.id
    ) {
      return;
    }
    refreshedTerminalTaskId.current = latestTask.id;
    void refreshSessionTasks();
    void refreshSessions();
  }, [latestTask, refreshSessionTasks, refreshSessions]);

  function handleSessionCreated(
    session: CreateSessionResponse,
    submission: TaskSubmission,
  ) {
    setActiveConversation({
      sessionId: session.session_id,
      title: session.title,
      taskId: session.task_id,
      initialStatus: session.status,
      latestWorkspace: submission.workspace,
    });
    setSelectedTaskId(session.task_id);
    setUrlRestoreError(null);
    setIsRestoringUrl(false);
    refreshedTerminalTaskId.current = null;
    void refreshSessions();
  }

  function handleTaskCreated(
    task: CreateSessionTaskResponse,
    submission: TaskSubmission,
  ) {
    setActiveConversation((current) => {
      if (current === null || current.sessionId !== task.session_id) {
        return current;
      }
      return {
        ...current,
        taskId: task.task_id,
        initialStatus: task.status,
        latestWorkspace: submission.workspace,
      };
    });
    setSelectedTaskId(task.task_id);
    setUrlRestoreError(null);
    refreshedTerminalTaskId.current = null;
    void refreshSessionTasks();
    void refreshSessions();
  }

  function handleSessionSelected(session: SessionSummary) {
    setActiveConversation(toActiveConversation(session));
    setSelectedTaskId(session.latest_task_id);
    setUrlRestoreError(null);
    setIsRestoringUrl(false);
    refreshedTerminalTaskId.current = null;
    if (overlayPanelSide === "left") {
      closeOverlayPanel(true);
    }
  }

  function handleTaskSelected(taskId: string) {
    setSelectedTaskId(taskId);
    setUrlRestoreError(null);
  }

  function handleNewConversation() {
    setActiveConversation(null);
    setSelectedTaskId(null);
    setUrlRestoreError(null);
    setIsRestoringUrl(false);
    refreshedTerminalTaskId.current = null;
    if (overlayPanelSide === "left") {
      closeOverlayPanel(true);
    }
  }

  function handleToggleLeftPanel() {
    if (isMobile) {
      setMobileRightDrawerOpen(false);
      setMobileLeftDrawerOpen((current) => !current);
      return;
    }
    toggleLeftSidebar();
  }

  function handleToggleRightPanel() {
    if (isMobile) {
      setMobileLeftDrawerOpen(false);
      setMobileRightDrawerOpen((current) => !current);
      return;
    }
    toggleRightInspector();
  }

  function closeOverlayPanel(restoreFocus: boolean) {
    const side = overlayPanelSide;
    if (side === "left") {
      if (isMobile) {
        setMobileLeftDrawerOpen(false);
      } else if (!leftSidebarCollapsed) {
        toggleLeftSidebar();
      }
    } else if (side === "right") {
      setMobileRightDrawerOpen(false);
    }
    if (restoreFocus && side !== null) {
      const buttonId =
        side === "left" ? "toggle-left-panel" : "toggle-right-panel";
      window.requestAnimationFrame(() => {
        document.getElementById(buttonId)?.focus();
      });
    }
  }

  return (
    <main
      className={[
        "app-shell",
        `layout-${responsiveLayout.toLowerCase()}`,
        !leftPanelOpen ? "left-sidebar-collapsed" : "",
        !rightPanelOpen ? "right-inspector-collapsed" : "",
        overlayPanelSide !== null ? "has-open-drawer" : "",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <ConversationSidebar
        isOpen={leftPanelOpen}
        isDrawer={
          isMobile || responsiveLayout === ResponsiveLayout.TABLET
        }
        sessions={sessionList.sessions}
        selectedSessionId={activeConversation?.sessionId ?? null}
        selectedTaskStatus={latestTask?.status ?? null}
        isLoading={sessionList.isLoading}
        error={sessionList.error}
        onNewConversation={handleNewConversation}
        onSessionSelected={handleSessionSelected}
        onClose={() => closeOverlayPanel(true)}
        onRetry={() => void refreshSessions()}
      />
      <ChatWorkspace
        activeConversation={activeConversation}
        tasks={tasks}
        latestTask={latestTask}
        selectedTaskId={selectedTaskId}
        tasksLoading={sessionTasks.isLoading || isRestoringUrl}
        tasksError={
          sessionTasks.error ?? latestSnapshotState.error ?? urlRestoreError
        }
        navigationLoading={isRestoringUrl}
        leftSidebarCollapsed={!leftPanelOpen}
        rightInspectorCollapsed={!rightPanelOpen}
        pendingApprovalCount={pendingApprovalCount}
        onToggleLeftSidebar={handleToggleLeftPanel}
        onToggleRightInspector={handleToggleRightPanel}
        onTaskSelected={handleTaskSelected}
        onRetryTasks={() => void refreshSessionTasks()}
        onSessionCreated={handleSessionCreated}
        onTaskCreated={handleTaskCreated}
      />
      <AgentInspector
        isOpen={rightPanelOpen}
        isDrawer={isMobile}
        selectedTaskId={selectedTaskId}
        fallbackTask={selectedTask}
        snapshot={selectedSnapshotState.snapshot}
        isLoading={selectedSnapshotState.isLoading || isRestoringUrl}
        error={selectedSnapshotState.error}
        notFound={selectedSnapshotState.notFound}
        onClose={() => closeOverlayPanel(true)}
        onRetry={selectedSnapshotState.refresh}
      />
      {overlayPanelSide !== null ? (
        <button
          className="panel-backdrop"
          type="button"
          onClick={() => closeOverlayPanel(true)}
          aria-label={
            overlayPanelSide === "left"
              ? "关闭会话抽屉"
              : "关闭执行抽屉"
          }
        />
      ) : null}
    </main>
  );
}

function toActiveConversation(session: SessionSummary): ActiveConversation {
  return {
    sessionId: session.id,
    title: session.title,
    taskId: session.latest_task_id,
    initialStatus: session.latest_task_status,
    latestWorkspace: session.latest_workspace,
  };
}
