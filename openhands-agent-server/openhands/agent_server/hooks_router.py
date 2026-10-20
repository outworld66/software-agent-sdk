"""Hooks router for OpenHands Agent Server.

This module defines the HTTP API endpoints for hook operations.
Business logic is delegated to hooks_service.py.
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from openhands.agent_server.hooks_service import load_hooks_from_workspace
from openhands.agent_server.user_isolation import oidc_owner, user_workspace
from openhands.sdk.hooks import HookConfig


hooks_router = APIRouter(prefix="/hooks", tags=["Hooks"])


class HooksRequest(BaseModel):
    """Request body for loading hooks."""

    project_dir: str | None = Field(
        default=None, description="Workspace directory path for project hooks"
    )


class HooksResponse(BaseModel):
    """Response containing hooks configuration."""

    hook_config: HookConfig | None = Field(
        default=None,
        description="Hook configuration loaded from the workspace, or None if not found",  # noqa: E501
    )


@hooks_router.post("", response_model=HooksResponse)
def get_hooks(request: HooksRequest, http_request: Request) -> HooksResponse:
    """Load hooks from the workspace .openhands/hooks.json file.

    This endpoint reads the hooks configuration from the project's
    .openhands/hooks.json file if it exists.

    Args:
        request: HooksRequest containing the project directory path.

    Returns:
        HooksResponse containing the hook configuration or None.
    """
    project_dir = request.project_dir
    owner = oidc_owner(http_request)
    if owner is not None and "admin" not in getattr(
        http_request.state, "auth_roles", []
    ):
        root = user_workspace(http_request.app.state.config, owner)
        if project_dir is not None:
            candidate = Path(project_dir).resolve()
            if candidate != root and not candidate.is_relative_to(root):
                raise HTTPException(
                    status.HTTP_403_FORBIDDEN,
                    "Project directory is outside your workspace",
                )
        project_dir = str(root) if project_dir is None else project_dir
    hook_config = load_hooks_from_workspace(project_dir=project_dir)
    return HooksResponse(hook_config=hook_config)
