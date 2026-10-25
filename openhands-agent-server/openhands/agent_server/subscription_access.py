"""OIDC-scoped access to ChatGPT subscription credentials."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, Request, status

from openhands.agent_server.config import Config
from openhands.agent_server.persistence import get_settings_store
from openhands.agent_server.user_isolation import oidc_owner
from openhands.sdk.llm.auth.credentials import get_credentials_dir


_GRANTS_KEY = "chatgpt_shared_token_users"


def is_admin(request: Request) -> bool:
    return "admin" in getattr(request.state, "auth_roles", [])


def grants(config: Config) -> set[str]:
    settings = get_settings_store(config).load()
    values = (settings.misc_settings if settings else {}).get(_GRANTS_KEY, [])
    return (
        {value for value in values if isinstance(value, str)}
        if isinstance(values, list)
        else set()
    )


def user_sub(request: Request) -> str:
    user = request.scope.get("session", {}).get("oidc_user")
    if not isinstance(user, dict) or not user.get("sub"):
        if not request.app.state.config.oidc_issuer_url and is_admin(request):
            return "local-admin"
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "OIDC login required")
    return str(user["sub"])


def can_use_shared(request: Request) -> bool:
    return is_admin(request) or user_sub(request) in grants(request.app.state.config)


def personal_auth_dir(request: Request) -> Path:
    owner = oidc_owner(request)
    if owner is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "OIDC login required")
    return request.app.state.config.conversations_path.parent / "users" / owner / "auth"


def auth_dir(request: Request, shared: bool = False) -> Path:
    if shared:
        if not can_use_shared(request):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN, "Shared ChatGPT access is not enabled"
            )
        return get_credentials_dir()
    return personal_auth_dir(request)


def set_shared_grant(config: Config, sub: str, enabled: bool) -> list[str]:
    def update(settings):
        values = settings.misc_settings.get(_GRANTS_KEY, [])
        users = (
            {value for value in values if isinstance(value, str)}
            if isinstance(values, list)
            else set()
        )
        users.add(sub) if enabled else users.discard(sub)
        settings.misc_settings[_GRANTS_KEY] = sorted(users)
        return settings

    return get_settings_store(config).update(update).misc_settings[_GRANTS_KEY]


def shared_grants(config: Config) -> list[str]:
    return sorted(grants(config))


def shared_auth_dir() -> Path:
    return get_credentials_dir()
