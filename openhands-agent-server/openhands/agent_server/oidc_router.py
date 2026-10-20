"""OpenID Connect login and session endpoints for self-hosted deployments."""

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from authlib.integrations.base_client.errors import OAuthError
from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from openhands.agent_server.config import Config
from openhands.agent_server.dependencies import check_oidc_csrf


oidc_router = APIRouter(prefix="/api/auth/oidc", tags=["Auth"])


class OIDCStatus(BaseModel):
    enabled: bool
    user: dict[str, Any] | None = None
    roles: list[str] = Field(default_factory=list)


def _get_oauth_client(config: Config):
    if not config.oidc_issuer_url:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "OIDC is not enabled")
    oauth = OAuth()
    oauth.register(
        name="openhands",
        client_id=config.oidc_client_id,
        client_secret=config.oidc_client_secret.get_secret_value()
        if config.oidc_client_secret
        else None,
        server_metadata_url=(
            f"{config.oidc_issuer_url.rstrip('/')}/.well-known/openid-configuration"
        ),
        client_kwargs={"scope": "openid profile email"},
    )
    return oauth.openhands


def _safe_next(value: str | None) -> str:
    if (
        value
        and value.startswith("/")
        and not value.startswith("//")
        and "\\" not in value
        and not any(ord(char) < 32 for char in value)
        and not urlsplit(value).netloc
    ):
        return value
    return "/"


@oidc_router.get("", response_model=OIDCStatus)
async def oidc_status(request: Request) -> OIDCStatus:
    config: Config = request.app.state.config
    user = request.session.get("oidc_user") if config.oidc_issuer_url else None
    return OIDCStatus(
        enabled=bool(config.oidc_issuer_url),
        user=user,
        roles=request.session.get("oidc_roles", []) if user else [],
    )


@oidc_router.get("/login", name="oidc_login")
async def oidc_login(request: Request, next: str | None = None):
    config: Config = request.app.state.config
    client = _get_oauth_client(config)
    request.session["oidc_next"] = _safe_next(next)
    return await client.authorize_redirect(request, request.url_for("oidc_callback"))


@oidc_router.get("/callback", name="oidc_callback")
async def oidc_callback(request: Request):
    config: Config = request.app.state.config
    client = _get_oauth_client(config)
    try:
        token = await client.authorize_access_token(request)
    except OAuthError as exc:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "OIDC authentication failed"
        ) from exc
    userinfo = token.get("userinfo")
    if not isinstance(userinfo, Mapping) or not userinfo.get("sub"):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "OIDC user identity is missing"
        )

    groups: Any = userinfo.get(config.oidc_groups_claim, [])
    if not isinstance(groups, list):
        groups = []
    roles = ["user"]
    if set(config.oidc_admin_groups).intersection(str(group) for group in groups):
        roles.append("admin")

    next_url = _safe_next(request.session.get("oidc_next"))
    request.session.clear()
    request.session["oidc_user"] = {
        "sub": str(userinfo["sub"]),
        "name": userinfo.get("name"),
        "email": userinfo.get("email"),
    }
    request.session["oidc_roles"] = roles
    return RedirectResponse(next_url, status_code=status.HTTP_303_SEE_OTHER)


@oidc_router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(check_oidc_csrf)],
)
async def oidc_logout(request: Request):
    if not request.app.state.config.oidc_issuer_url:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "OIDC is not enabled")
    request.session.clear()
