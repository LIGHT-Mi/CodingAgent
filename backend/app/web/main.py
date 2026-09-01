"""FastAPI 应用入口和受控的 HTTP 基础设施边界。"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.conversation_service import ConversationNotFoundError
from app.api.task_validation import TaskPromptValidationError
from app.api.workspace import WorkspaceValidationError
from app.application import ApplicationFactory, TaskRunner
from app.db.persistence import PersistenceServiceError, RecordNotFoundError
from app.web.contracts import ErrorResponse
from app.web.routes import router


logger = logging.getLogger(__name__)


def create_web_app(
    application_factory: ApplicationFactory | None = None,
    task_runner: TaskRunner | None = None,
) -> FastAPI:
    """创建复用统一应用装配、TaskRunner 和请求级 Session 的 Web API。"""

    if application_factory is not None and not isinstance(
        application_factory,
        ApplicationFactory,
    ):
        raise TypeError("application_factory must be an ApplicationFactory")
    if task_runner is not None and not isinstance(task_runner, TaskRunner):
        raise TypeError("task_runner must be a TaskRunner")

    configured_factory = application_factory or ApplicationFactory()
    configured_runner = task_runner or TaskRunner(configured_factory)
    owns_runner = task_runner is None

    @asynccontextmanager
    async def lifespan(web_app: FastAPI):
        del web_app
        try:
            yield
        finally:
            if owns_runner:
                configured_runner.shutdown(wait=False, cancel_running=True)

    web_app = FastAPI(
        title="Coding Agent API",
        version="0.1.0",
        lifespan=lifespan,
    )
    web_app.state.application_factory = configured_factory
    web_app.state.task_runner = configured_runner
    if configured_factory.config.WEB_CORS_ALLOWED_ORIGINS:
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=list(
                configured_factory.config.WEB_CORS_ALLOWED_ORIGINS
            ),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Accept", "Content-Type"],
        )
    web_app.include_router(router)

    @web_app.exception_handler(ConversationNotFoundError)
    async def conversation_not_found_handler(
        request: Request,
        exc: ConversationNotFoundError,
    ) -> JSONResponse:
        del request, exc
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=ErrorResponse(
                detail="Session was not found"
            ).model_dump(),
        )

    @web_app.exception_handler(RecordNotFoundError)
    async def record_not_found_handler(
        request: Request,
        exc: RecordNotFoundError,
    ) -> JSONResponse:
        del request, exc
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=ErrorResponse(detail="Task was not found").model_dump(),
        )

    @web_app.exception_handler(
        TaskPromptValidationError,
    )
    @web_app.exception_handler(WorkspaceValidationError)
    async def invalid_task_input_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(detail=str(exc)).model_dump(),
        )

    @web_app.exception_handler(PersistenceServiceError)
    async def persistence_error_handler(
        request: Request,
        exc: PersistenceServiceError,
    ) -> JSONResponse:
        del request
        logger.error(
            "Web persistence operation failed",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                detail="Persistence service unavailable"
            ).model_dump(),
        )

    @web_app.exception_handler(Exception)
    async def unexpected_error_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        del request
        logger.error(
            "Unexpected Web API error",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(detail="Internal server error").model_dump(),
        )

    return web_app


app = create_web_app()
