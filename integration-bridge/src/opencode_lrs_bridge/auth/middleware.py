"""
Authentication and authorization layer for opencode ↔ LRS-Agents integration.
"""

import asyncio
import ssl
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Security, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import httpx
import structlog

from ..config.settings import IntegrationBridgeConfig
from ..models.schemas import AgentState

logger = structlog.get_logger(__name__)


class AuthenticationError(Exception):
    """Authentication error."""

    pass


class AuthorizationError(Exception):
    """Authorization error."""

    pass


class TokenManager:
    """JWT token management."""

    def __init__(self, config: IntegrationBridgeConfig):
        self.config = config
        self.algorithm = config.security.jwt_algorithm
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7

    def create_access_token(
        self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=self.access_token_expire_minutes
            )

        to_encode.update({"exp": expire, "type": "access"})
        return jwt.encode(to_encode, self._get_secret_key(), algorithm=self.algorithm)

    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        return jwt.encode(to_encode, self._get_secret_key(), algorithm=self.algorithm)

    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token and return payload."""
        try:
            payload = jwt.decode(
                token, self._get_secret_key(), algorithms=[self.algorithm]
            )
            return payload
        except JWTError as e:
            logger.error("Token verification failed", error=str(e))
            raise AuthenticationError("Invalid token")

    def _get_secret_key(self) -> str:
        """Get JWT secret key."""
        # In production, this should be loaded from secure storage
        return self.config.security.oauth_client_secret


class OAuth2Client:
    """OAuth2 client for token exchange with opencode."""

    def __init__(self, config: IntegrationBridgeConfig):
        self.config = config
        self.client_id = config.security.oauth_client_id
        self.client_secret = config.security.oauth_client_secret
        self.token_url = f"{config.security.oauth_provider_url}/token"
        self.authorize_url = f"{config.security.oauth_provider_url}/authorize"

    async def exchange_code_for_tokens(
        self, code: str, redirect_uri: str
    ) -> Dict[str, str]:
        """Exchange authorization code for tokens."""
        data = {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "redirect_uri": redirect_uri,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(self.token_url, data=data)

        if response.status_code != 200:
            raise AuthenticationError("Failed to exchange code for tokens")

        return response.json()

    async def refresh_access_token(self, refresh_token: str) -> Dict[str, str]:
        """Refresh access token using refresh token."""
        data = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": refresh_token,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(self.token_url, data=data)

        if response.status_code != 200:
            raise AuthenticationError("Failed to refresh access token")

        return response.json()


class PermissionManager:
    """Permission management for agents and users."""

    def __init__(self):
        self.permissions = {
            "read_state": ["read_agent_state", "read_precision_data"],
            "write_state": ["write_agent_state", "update_config"],
            "execute_tools": ["execute_tools", "read_tools"],
            "manage_agents": ["create_agents", "delete_agents", "control_agents"],
            "system_admin": ["all"],
        }

    def has_permission(
        self, user_permissions: List[str], required_permission: str
    ) -> bool:
        """Check if user has required permission."""
        if "system_admin" in user_permissions:
            return True

        for perm in user_permissions:
            if perm in self.permissions.get(required_permission, []):
                return True
        return False

    def get_agent_permissions(self, agent_id: str, user_id: str) -> List[str]:
        """Get permissions for a specific agent."""
        # In production, this would query a database
        return ["read_state", "write_state", "execute_tools"]

    def get_user_permissions(self, user_id: str) -> List[str]:
        """Get permissions for a specific user."""
        # In production, this would query a database
        return ["read_state", "write_state", "execute_tools"]


class MTLSValidator:
    """mTLS certificate validation for enterprise security."""

    def __init__(self, config: IntegrationBridgeConfig):
        self.config = config
        self.enabled = config.security.enable_mtls

    def validate_client_certificate(self, request: Request) -> bool:
        """Validate client TLS certificate."""
        if not self.enabled:
            return True

        client_cert = request.client.ssl

        if not client_cert:
            raise AuthenticationError("Client certificate required")

        # Validate certificate against CA
        return self._validate_cert_chain(client_cert)

    def _validate_cert_chain(self, cert: Any) -> bool:
        """Validate certificate chain against trusted CA."""
        # Implementation would validate certificate against CA
        return True


class AuthenticationMiddleware:
    """Authentication middleware for the integration bridge."""

    def __init__(self, config: IntegrationBridgeConfig):
        self.config = config
        self.token_manager = TokenManager(config)
        self.oauth_client = OAuth2Client(config)
        self.permission_manager = PermissionManager()
        self.mtls_validator = MTLSValidator(config)
        self.security = HTTPBearer(auto_error=False)

    async def authenticate_request(
        self,
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(
            HTTPBearer(auto_error=False)
        ),
    ) -> Dict[str, Any]:
        """Authenticate incoming request."""
        # Validate mTLS if enabled
        self.mtls_validator.validate_client_certificate(request)

        # Check for JWT token
        if credentials and credentials.scheme == "Bearer":
            return await self._authenticate_jwt(credentials.credentials)

        # Check for API key in headers
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return await self._authenticate_api_key(api_key)

        raise AuthenticationError("No valid authentication provided")

    async def _authenticate_jwt(self, token: str) -> Dict[str, Any]:
        """Authenticate using JWT token."""
        try:
            payload = self.token_manager.verify_token(token)

            if payload.get("type") != "access":
                raise AuthenticationError("Invalid token type")

            # Check if token is expired
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
                raise AuthenticationError("Token expired")

            return payload

        except JWTError:
            raise AuthenticationError("Invalid JWT token")

    async def _authenticate_api_key(self, api_key: str) -> Dict[str, Any]:
        """Authenticate using API key."""
        # In production, validate against database
        if api_key == "valid-api-key":
            return {"user_id": "api-user", "permissions": ["read_state", "write_state"]}

        raise AuthenticationError("Invalid API key")

    def authorize_action(
        self,
        user_info: Dict[str, Any],
        required_permission: str,
        agent_id: Optional[str] = None,
    ) -> bool:
        """Authorize user action."""
        user_id = user_info.get("user_id")
        user_permissions = user_info.get("permissions", [])

        # Check system permissions
        if self.permission_manager.has_permission(
            user_permissions, required_permission
        ):
            return True

        # Check agent-specific permissions
        if agent_id:
            agent_permissions = self.permission_manager.get_agent_permissions(
                agent_id, user_id
            )
            if self.permission_manager.has_permission(
                agent_permissions, required_permission
            ):
                return True

        raise AuthorizationError("Insufficient permissions")


def get_current_user():
    """FastAPI dependency to get current authenticated user."""

    async def dependency(
        request: Request,
        auth_middleware: AuthenticationMiddleware = Depends(),
    ):
        return await auth_middleware.authenticate_request(request)

    return dependency


def require_permission(permission: str, agent_id: Optional[str] = None):
    """FastAPI dependency to require specific permission."""

    async def dependency(
        request: Request,
        auth_middleware: AuthenticationMiddleware = Depends(),
    ):
        current_user = await auth_middleware.authenticate_request(request)
        auth_middleware.authorize_action(current_user, permission, agent_id)
        return current_user

    return dependency


class SessionManager:
    """Session management for WebSocket connections."""

    def __init__(self, config: IntegrationBridgeConfig):
        self.config = config
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self, session_id: str, user_info: Dict[str, Any]) -> str:
        """Create new session."""
        session_token = self.token_manager.create_access_token(
            {
                "session_id": session_id,
                "user_id": user_info.get("user_id"),
                "type": "session",
            }
        )

        self.sessions[session_id] = {
            "user_info": user_info,
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "websocket_connections": [],
        }

        return session_token

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information."""
        return self.sessions.get(session_id)

    def update_session_activity(self, session_id: str):
        """Update session last activity."""
        if session_id in self.sessions:
            self.sessions[session_id]["last_activity"] = datetime.utcnow()

    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        current_time = datetime.utcnow()
        expired_sessions = []

        for session_id, session_data in self.sessions.items():
            last_activity = session_data.get("last_activity", current_time)
            if current_time - last_activity > timedelta(hours=24):
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            del self.sessions[session_id]
            logger.info("Cleaned up expired session", session_id=session_id)
