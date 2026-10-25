"""Router for LLM model, provider, and subscription information endpoints."""

from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass
from uuid import UUID

import httpx
from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from openhands.agent_server.subscription_access import (
    auth_dir,
    is_admin,
    set_shared_grant,
    shared_grants,
    user_sub,
)
from openhands.agent_server.user_isolation import owner_id
from openhands.sdk.llm.auth.credentials import CredentialStore
from openhands.sdk.llm.auth.openai import (
    DEVICE_CODE_TIMEOUT_SECONDS,
    OPENAI_CODEX_MODELS,
    DeviceCode,
    OpenAISubscriptionAuth,
)
from openhands.sdk.llm.utils.unverified_models import (
    _extract_model_and_provider,
    _get_litellm_provider_names,
    get_supported_llm_models,
)
from openhands.sdk.llm.utils.verified_models import VERIFIED_MODELS


llm_router = APIRouter(prefix="/llm", tags=["LLM"])


@dataclass(frozen=True)
class PendingDeviceLogin:
    """Server-side state for an in-progress device-code login."""

    device_code: DeviceCode
    expires_at: int
    epoch: int
    user_sub: str
    shared: bool


_PENDING_OPENAI_DEVICE_LOGINS: dict[str, PendingDeviceLogin] = {}
_IN_FLIGHT_OPENAI_DEVICE_LOGINS: set[str] = set()
_OPENAI_DEVICE_LOGIN_LOCK = asyncio.Lock()
_OPENAI_DEVICE_LOGIN_EPOCH = 0


class ProvidersResponse(BaseModel):
    """Response containing the list of available LLM providers."""

    providers: list[str]


class ModelsResponse(BaseModel):
    """Response containing the list of available LLM models."""

    models: list[str]


class VerifiedModelsResponse(BaseModel):
    """Response containing verified LLM models organized by provider."""

    models: dict[str, list[str]]


class SubscriptionStatusResponse(BaseModel):
    """Safe subscription authentication status."""

    vendor: str = "openai"
    connected: bool
    account_email: str | None = None
    expires_at: int | None = None
    shared_available: bool = False


class SubscriptionDeviceStartResponse(BaseModel):
    """Device-code challenge details for browser sign-in."""

    device_code: str = Field(description="Opaque server-side polling token.")
    user_code: str
    verification_uri: str
    verification_uri_complete: str | None = None
    expires_at: int
    interval_seconds: int


class SubscriptionDevicePollRequest(BaseModel):
    """Poll request for a previously-started subscription device login."""

    device_code: str


class SubscriptionModelsResponse(BaseModel):
    """Models available through a subscription provider."""

    vendor: str = "openai"
    models: list[str]


def _get_openai_subscription_auth(directory) -> OpenAISubscriptionAuth:
    return OpenAISubscriptionAuth(CredentialStore(directory))


def _status_from_auth(auth: OpenAISubscriptionAuth) -> SubscriptionStatusResponse:
    creds = auth.get_credentials()
    if creds is None or creds.is_expired():
        return SubscriptionStatusResponse(connected=False)
    return SubscriptionStatusResponse(connected=True, expires_at=creds.expires_at)


def _drop_expired_device_logins() -> None:
    now = int(time.time() * 1000)
    for key, pending in list(_PENDING_OPENAI_DEVICE_LOGINS.items()):
        if pending.expires_at <= now:
            _PENDING_OPENAI_DEVICE_LOGINS.pop(key, None)


async def _stop_runtimes(request: Request, *, shared: bool) -> None:
    registry = getattr(request.app.state, "conversation_registry", None)
    provisioning = getattr(registry, "provisioning", None)
    if provisioning is None:
        return
    owner = owner_id(request.app.state.config, {"sub": user_sub(request)})
    for path in provisioning.control_root.glob("*.json"):
        try:
            conversation_id = UUID(hex=path.stem)
            identity = provisioning.load_optional(conversation_id)
            matches = identity and (
                identity.shared_auth
                if shared
                else identity.auth_owner == owner and not identity.shared_auth
            )
            if matches:
                await registry.stop(conversation_id)
        except (OSError, ValueError):
            continue


@llm_router.get("/providers", response_model=ProvidersResponse)
async def list_providers() -> ProvidersResponse:
    """List all available LLM providers supported by LiteLLM."""
    providers = sorted(_get_litellm_provider_names())
    return ProvidersResponse(providers=providers)


@llm_router.get("/models", response_model=ModelsResponse)
async def list_models(
    provider: str | None = Query(
        default=None,
        description="Filter models by provider (e.g., 'openai', 'anthropic')",
    ),
) -> ModelsResponse:
    """List all available LLM models supported by LiteLLM.

    Args:
        provider: Optional provider name to filter models by.

    Note: Bedrock models are excluded unless AWS credentials are configured.
    """
    all_models = get_supported_llm_models()

    if provider is None:
        models = sorted(set(all_models))
    else:
        filtered_models = []
        verified_provider_models = set(VERIFIED_MODELS.get(provider, ()))
        for model in all_models:
            model_provider, _, _ = _extract_model_and_provider(model)
            if model_provider == provider or (
                model in verified_provider_models
                # A namespaced verified entry (an OpenRouter id with the
                # ``openrouter/`` prefix stripped, e.g. ``deepseek/deepseek-chat``)
                # can collide with another provider's real catalog model of the
                # same name. When it does, the model extracts to that other
                # provider, so only honor the verified-list match when the model
                # does not resolve to a different provider.
                and not ("/" in model and model_provider and model_provider != provider)
            ):
                filtered_models.append(model)
        models = sorted(set(filtered_models))

    return ModelsResponse(models=models)


@llm_router.get("/models/verified", response_model=VerifiedModelsResponse)
async def list_verified_models() -> VerifiedModelsResponse:
    """List all verified LLM models organized by provider.

    Verified models are those that have been tested and confirmed to work well
    with OpenHands.
    """
    return VerifiedModelsResponse(models=VERIFIED_MODELS)


@llm_router.get(
    "/subscription/openai/models", response_model=SubscriptionModelsResponse
)
async def list_openai_subscription_models() -> SubscriptionModelsResponse:
    """List models available through ChatGPT subscription authentication."""
    return SubscriptionModelsResponse(models=sorted(OPENAI_CODEX_MODELS))


@llm_router.get(
    "/subscription/openai/status", response_model=SubscriptionStatusResponse
)
async def get_openai_subscription_status(
    request: Request,
) -> SubscriptionStatusResponse:
    """Return safe ChatGPT subscription connection state without tokens."""
    user_sub(request)
    shared = is_admin(request)
    auth = _get_openai_subscription_auth(auth_dir(request, shared))
    try:
        await auth.refresh_if_needed()
    except (RuntimeError, httpx.HTTPError):
        # A rejected refresh (RuntimeError) or an unreachable auth.openai.com
        # (httpx transport error) both leave no usable login right now.
        return SubscriptionStatusResponse(
            connected=False,
            shared_available=(
                is_admin(request)
                or user_sub(request) in shared_grants(request.app.state.config)
            ),
        )
    result = _status_from_auth(auth)
    return result.model_copy(
        update={
            "shared_available": is_admin(request)
            or user_sub(request) in shared_grants(request.app.state.config)
        }
    )


@llm_router.post(
    "/subscription/openai/device/start",
    response_model=SubscriptionDeviceStartResponse,
)
async def start_openai_subscription_device_login(
    request: Request, shared: bool | None = Query(default=None)
) -> SubscriptionDeviceStartResponse:
    """Start ChatGPT device-code sign-in without returning tokens."""
    identity = user_sub(request)
    shared = is_admin(request) if shared is None else shared
    if shared and not is_admin(request):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Only administrators can connect the shared account",
        )
    auth = _get_openai_subscription_auth(auth_dir(request, shared))
    challenge = await auth.start_device_login()
    token = secrets.token_urlsafe(32)
    expires_at = int(time.time() * 1000) + (DEVICE_CODE_TIMEOUT_SECONDS * 1000)
    async with _OPENAI_DEVICE_LOGIN_LOCK:
        _drop_expired_device_logins()
        _PENDING_OPENAI_DEVICE_LOGINS[token] = PendingDeviceLogin(
            device_code=challenge,
            expires_at=expires_at,
            epoch=_OPENAI_DEVICE_LOGIN_EPOCH,
            user_sub=identity,
            shared=shared,
        )
    return SubscriptionDeviceStartResponse(
        device_code=token,
        user_code=challenge.user_code,
        verification_uri=challenge.verification_url,
        expires_at=expires_at,
        interval_seconds=challenge.interval,
    )


@llm_router.post(
    "/subscription/openai/device/poll", response_model=SubscriptionStatusResponse
)
async def poll_openai_subscription_device_login(
    request: SubscriptionDevicePollRequest,
    http_request: Request,
) -> SubscriptionStatusResponse:
    """Poll a ChatGPT device-code sign-in without returning tokens."""
    async with _OPENAI_DEVICE_LOGIN_LOCK:
        _drop_expired_device_logins()
        pending = _PENDING_OPENAI_DEVICE_LOGINS.pop(request.device_code, None)
        if pending is None:
            if request.device_code in _IN_FLIGHT_OPENAI_DEVICE_LOGINS:
                return SubscriptionStatusResponse(connected=False)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Subscription device login not found or expired",
            )
        if pending.user_sub != user_sub(http_request):
            _PENDING_OPENAI_DEVICE_LOGINS[request.device_code] = pending
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                "Subscription device login not found or expired",
            )
        if pending.shared and not is_admin(http_request):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN, "Administrator role required"
            )
        _IN_FLIGHT_OPENAI_DEVICE_LOGINS.add(request.device_code)

    auth = _get_openai_subscription_auth(auth_dir(http_request, pending.shared))
    credentials = None
    try:
        credentials = await auth.poll_device_login(pending.device_code, persist=False)
    finally:
        async with _OPENAI_DEVICE_LOGIN_LOCK:
            _IN_FLIGHT_OPENAI_DEVICE_LOGINS.discard(request.device_code)
            # Keep the opaque poll token usable if the provider is still pending
            # or if the polling request fails before credentials are obtained.
            if credentials is None and pending.epoch == _OPENAI_DEVICE_LOGIN_EPOCH:
                _PENDING_OPENAI_DEVICE_LOGINS[request.device_code] = pending

    async with _OPENAI_DEVICE_LOGIN_LOCK:
        current_epoch = _OPENAI_DEVICE_LOGIN_EPOCH
        if credentials is None:
            return SubscriptionStatusResponse(connected=False)
        if pending.epoch != current_epoch:
            return SubscriptionStatusResponse(connected=False)
        auth.save_credentials(credentials)
    await _stop_runtimes(http_request, shared=pending.shared)
    return SubscriptionStatusResponse(connected=True, expires_at=credentials.expires_at)


@llm_router.post(
    "/subscription/openai/logout", response_model=SubscriptionStatusResponse
)
async def logout_openai_subscription(
    request: Request, shared: bool | None = Query(default=None)
) -> SubscriptionStatusResponse:
    """Remove stored ChatGPT subscription credentials."""
    user_sub(request)
    shared = is_admin(request) if shared is None else shared
    if shared and not is_admin(request):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Only administrators can disconnect the shared account",
        )
    auth = _get_openai_subscription_auth(auth_dir(request, shared))
    async with _OPENAI_DEVICE_LOGIN_LOCK:
        global _OPENAI_DEVICE_LOGIN_EPOCH
        _OPENAI_DEVICE_LOGIN_EPOCH += 1
        for key, pending in list(_PENDING_OPENAI_DEVICE_LOGINS.items()):
            if pending.user_sub == user_sub(request) and pending.shared == shared:
                _PENDING_OPENAI_DEVICE_LOGINS.pop(key, None)
        auth.logout()
    await _stop_runtimes(request, shared=shared)
    return SubscriptionStatusResponse(connected=False)


class SharedGrantRequest(BaseModel):
    sub: str = Field(min_length=1, max_length=512)
    enabled: bool


@llm_router.get("/subscription/openai/grants")
async def list_openai_shared_grants(request: Request) -> dict[str, list[str]]:
    if not is_admin(request):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Administrator role required")
    return {"users": shared_grants(request.app.state.config)}


@llm_router.post("/subscription/openai/grants")
async def update_openai_shared_grant(
    body: SharedGrantRequest, request: Request
) -> dict[str, list[str]]:
    if not is_admin(request):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Administrator role required")
    sub = body.sub.strip()
    if not sub or any(ord(char) < 32 for char in sub):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "Invalid OIDC subject"
        )
    users = set_shared_grant(request.app.state.config, sub, body.enabled)
    if not body.enabled:
        registry = getattr(request.app.state, "conversation_registry", None)
        provisioning = getattr(registry, "provisioning", None)
        if provisioning is not None:
            owner = owner_id(request.app.state.config, {"sub": sub})
            for path in provisioning.control_root.glob("*.json"):
                try:
                    conversation_id = UUID(hex=path.stem)
                    identity = provisioning.load_optional(conversation_id)
                    if (
                        identity
                        and identity.auth_owner == owner
                        and identity.shared_auth
                    ):
                        provisioning.save(
                            identity.model_copy(update={"shared_auth": False})
                        )
                        await registry.stop(conversation_id)
                except (OSError, ValueError):
                    continue
    return {"users": users}
