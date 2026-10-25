"""OIDC user boundaries for persisted conversations and workspace files."""

import hashlib
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request, status

from openhands.agent_server.config import Config


def owner_id(config: Config, user: dict[str, Any]) -> str:
    identity = f"{config.oidc_issuer_url}\0{user['sub']}"
    return hashlib.sha256(identity.encode()).hexdigest()


def user_workspace(config: Config, owner: str) -> Path:
    return (config.conversations_path.parent / "users" / owner / "workspace").resolve()


def oidc_owner(request: Request) -> str | None:
    config: Config = request.app.state.config
    user = request.scope.get("session", {}).get("oidc_user")
    if not config.oidc_issuer_url or not isinstance(user, dict) or not user.get("sub"):
        return None
    return owner_id(config, user)


def conversation_owner_filter(request: Request) -> str | None:
    if "admin" in getattr(request.state, "auth_roles", []):
        return None
    return oidc_owner(request)


def check_conversation_access(request: Request) -> None:
    owner = oidc_owner(request)
    if owner is None or "admin" in getattr(request.state, "auth_roles", []):
        return
    if request.url.path.startswith("/api/bash") or "/vscode/" in request.url.path:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Administrator access required")
    raw_id = request.path_params.get("conversation_id") or request.path_params.get(
        "runtime_conversation_id"
    )
    if raw_id is None:
        return
    service = getattr(request.app.state, "conversation_service", None)
    try:
        stored_owner = service.get_conversation_owner(raw_id)
    except (AttributeError, OSError, ValueError):
        stored_owner = None
    if stored_owner != owner:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")


def check_user_file_access(request: Request) -> None:
    owner = oidc_owner(request)
    if owner is None or "admin" in getattr(request.state, "auth_roles", []):
        return
    raw_path = next(
        (
            request.query_params.get(name)
            for name in ("path", "workspace_dir", "cwd")
            if request.query_params.get(name) is not None
        ),
        None,
    )
    if raw_path is None:
        return
    target = Path(raw_path)
    if not target.is_absolute():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Path must be absolute")
    root = user_workspace(request.app.state.config, owner)
    candidate = target.resolve()
    if candidate != root and not candidate.is_relative_to(root):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Path is outside your workspace")


def websocket_conversation_owner(
    websocket: Any, service: Any, conversation_id: str
) -> bool:
    config: Config = websocket.app.state.config
    session = websocket.scope.get("session", {})
    user = session.get("oidc_user")
    if config.oidc_issuer_url and isinstance(user, dict):
        if "admin" in session.get("oidc_roles", []):
            return True
        try:
            return service.get_conversation_owner(conversation_id) == owner_id(
                config, user
            )
        except (AttributeError, OSError, ValueError):
            return False
    key = websocket.query_params.get("session_api_key") or websocket.headers.get(
        "x-session-api-key"
    )
    if key and key in config.session_api_keys:
        return True
    if "admin" in websocket.scope.get("auth_roles", []):
        return True
    if not config.oidc_issuer_url or not isinstance(user, dict):
        return True
    if "admin" in session.get("oidc_roles", []):
        return True
    owner = owner_id(config, user)
    try:
        return service.get_conversation_owner(conversation_id) == owner
    except (AttributeError, OSError, ValueError):
        return False
