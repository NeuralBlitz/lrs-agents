"""
NeuralBlitz v50 JWT Authentication Module
Provides secure token-based authentication for API access
"""

import jwt
import uuid
import time
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from flask import request, jsonify, g


class TokenType(Enum):
    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"


@dataclass
class TokenPayload:
    """Represents a decoded JWT token payload"""

    token_id: str
    user_id: str
    token_type: TokenType
    scopes: List[str]
    issued_at: int
    expires_at: int
    not_before: int
    subject: str
    issuer: str = "neuralblitz-v50"
    audience: str = "neuralblitz-api"


@dataclass
class APIUser:
    """Represents an authenticated API user"""

    user_id: str
    username: str
    email: Optional[str]
    scopes: List[str]
    roles: List[str]
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]


class JWTKeyManager:
    """Manages JWT signing and verification keys"""

    def __init__(self, secret_key: str = None):
        self._secret_key = secret_key or self._generate_secret()
        self._rotation_index = 0
        self._key_versions = {}
        self._revoked_tokens = set()
        self._last_rotation = time.time()

    def _generate_secret(self) -> str:
        """Generate a cryptographically secure secret key"""
        return secrets.token_urlsafe(64)

    def get_signing_key(self, kid: str = None) -> tuple:
        """Get the current signing key and its ID"""
        if kid is None:
            kid = f"v{self._rotation_index}"

        # Create key version if not exists
        if kid not in self._key_versions:
            self._key_versions[kid] = {
                "key": secrets.token_urlsafe(32),
                "created": time.time(),
                "rotations": self._rotation_index,
            }

        return self._key_versions[kid]["key"], kid

    def get_verification_keys(self) -> Dict[str, str]:
        """Get all valid verification keys"""
        return {kid: data["key"] for kid, data in self._key_versions.items()}

    def rotate_keys(self) -> str:
        """Rotate to a new key version"""
        self._rotation_index += 1
        self._last_rotation = time.time()
        new_kid = f"v{self._rotation_index}"
        self.get_signing_key(new_kid)
        return new_kid

    def revoke_token(self, jti: str, expires_at: int = None):
        """Add a token ID to the revoked list"""
        self._revoked_tokens.add(jti)
        # Store expiry for cleanup
        if expires_at:
            # Clean up expired revocations periodically
            current_time = time.time()
            self._revoked_tokens = {
                jti
                for jti in self._revoked_tokens
                if jti != jti or (expires_at and int(jti, 16) > current_time)
            }

    def is_revoked(self, jti: str) -> bool:
        """Check if a token ID has been revoked"""
        return jti in self._revoked_tokens


class TokenGenerator:
    """Generates JWT tokens for API authentication"""

    ALGORITHM = "HS512"
    TOKEN_LIFETIME = {
        TokenType.ACCESS: timedelta(hours=1),
        TokenType.REFRESH: timedelta(days=7),
        TokenType.API_KEY: timedelta(days=365),
    }

    def __init__(self, key_manager: JWTKeyManager):
        self._key_manager = key_manager
        self._default_issuer = "neuralblitz-v50"

    def generate_access_token(
        self,
        user_id: str,
        username: str,
        scopes: List[str],
        subject: str = "api_access",
        audience: str = "neuralblitz-api",
    ) -> str:
        """Generate a short-lived access token"""
        return self._create_token(
            user_id=user_id,
            username=username,
            token_type=TokenType.ACCESS,
            scopes=scopes,
            subject=subject,
            audience=audience,
        )

    def generate_refresh_token(
        self,
        user_id: str,
        username: str,
        scopes: List[str],
        subject: str = "token_refresh",
        audience: str = "neuralblitz-api",
    ) -> str:
        """Generate a long-lived refresh token"""
        return self._create_token(
            user_id=user_id,
            username=username,
            token_type=TokenType.REFRESH,
            scopes=scopes,
            subject=subject,
            audience=audience,
        )

    def generate_api_key(
        self,
        user_id: str,
        username: str,
        scopes: List[str],
        key_name: str = "default",
        subject: str = "api_access",
        audience: str = "neuralblitz-api",
    ) -> tuple:
        """Generate a long-lived API key and its token"""
        # Generate random API key
        api_key = f"nb_{key_name}_{secrets.token_urlsafe(32)}"

        # Create token payload
        token = self._create_token(
            user_id=user_id,
            username=username,
            token_type=TokenType.API_KEY,
            scopes=scopes,
            subject=subject,
            audience=audience,
            additional_claims={"api_key": api_key},
        )

        return api_key, token

    def _create_token(
        self,
        user_id: str,
        username: str,
        token_type: TokenType,
        scopes: List[str],
        subject: str,
        audience: str,
        additional_claims: Dict[str, Any] = None,
    ) -> str:
        """Create a JWT token with the specified parameters"""
        now = int(time.time())
        lifetime = self.TOKEN_LIFETIME[token_type]

        # Generate unique token ID
        jti = secrets.token_urlsafe(16)

        # Build payload
        payload = {
            "jti": jti,  # Token ID
            "sub": subject,
            "iss": self._default_issuer,
            "aud": audience,
            "iat": now,
            "nbf": now,  # Not before
            "exp": int((datetime.utcnow() + lifetime).timestamp()),
            "type": token_type.value,
            "user_id": user_id,
            "username": username,
            "scopes": scopes,
        }

        # Add any additional claims
        if additional_claims:
            payload.update(additional_claims)

        # Sign token
        signing_key, kid = self._key_manager.get_signing_key()
        payload["kid"] = kid  # Key ID

        return jwt.encode(payload, signing_key, algorithm=self.ALGORITHM)

    def generate_token_pair(self, user_id: str, username: str, scopes: List[str]) -> Dict[str, str]:
        """Generate both access and refresh tokens"""
        return {
            "access_token": self.generate_access_token(user_id, username, scopes),
            "refresh_token": self.generate_refresh_token(user_id, username, scopes),
            "token_type": "Bearer",
            "expires_in": int(self.TOKEN_LIFETIME[TokenType.ACCESS].total_seconds()),
        }


class TokenValidator:
    """Validates and decodes JWT tokens"""

    ALGORITHM = "HS512"

    def __init__(self, key_manager: JWTKeyManager):
        self._key_manager = key_manager
        self._clock_skew = 30  # 30 seconds tolerance

    def validate(
        self,
        token: str,
        expected_type: TokenType = None,
        expected_audience: str = None,
        expected_subject: str = None,
    ) -> tuple:
        """
        Validate a JWT token

        Returns:
            tuple: (success: bool, payload: dict or None, error: str or None)
        """
        try:
            # Get the header to find the key ID
            unverified_header = jwt.get_unverified_header(token)
            kid = unverified_header.get("kid", "v0")

            # Get verification keys
            keys = self._key_manager.get_verification_keys()

            if kid not in keys:
                # Try with "v" prefix
                kid = f"v{kid}" if not kid.startswith("v") else kid

            if kid not in keys:
                return False, None, f"Unknown key ID: {kid}"

            # Decode and verify
            payload = jwt.decode(
                token,
                keys[kid],
                algorithms=[self.ALGORITHM],
                audience=expected_audience,
                subject=expected_subject,
                clock_skew=self._clock_skew,
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_nbf": True,
                    "verify_aud": expected_audience is not None,
                    "verify_sub": expected_subject is not None,
                },
            )

            # Check if token type matches
            if expected_type:
                token_type = payload.get("type")
                if token_type != expected_type.value:
                    return (
                        False,
                        None,
                        f"Invalid token type: expected {expected_type.value}, got {token_type}",
                    )

            # Check if token is revoked
            jti = payload.get("jti")
            if self._key_manager.is_revoked(jti):
                return False, None, "Token has been revoked"

            return True, payload, None

        except jwt.ExpiredSignatureError:
            return False, None, "Token has expired"
        except jwt.InvalidAudienceError:
            return False, None, "Invalid audience"
        except jwt.InvalidSubjectError:
            return False, None, "Invalid subject"
        except jwt.InvalidTokenError as e:
            return False, None, f"Invalid token: {str(e)}"
        except Exception as e:
            return False, None, f"Token validation error: {str(e)}"

    def decode_without_verification(self, token: str) -> Optional[Dict]:
        """Decode token without verification (for inspection)"""
        try:
            return jwt.decode(token, options={"verify_signature": False})
        except:
            return None

    def get_token_payload(self, token: str) -> Optional[TokenPayload]:
        """Get a structured token payload"""
        success, payload, error = self.validate(token)
        if not success:
            return None

        try:
            return TokenPayload(
                token_id=payload.get("jti", ""),
                user_id=payload.get("user_id", ""),
                token_type=TokenType(payload.get("type", "access")),
                scopes=payload.get("scopes", []),
                issued_at=payload.get("iat", 0),
                expires_at=payload.get("exp", 0),
                not_before=payload.get("nbf", 0),
                subject=payload.get("sub", ""),
                issuer=payload.get("iss", ""),
                audience=payload.get("aud", ""),
            )
        except:
            return None


class JWTAuthenticator:
    """Main JWT authentication interface"""

    # Default scopes for different access levels
    SCOPES = {
        "admin": ["read", "write", "execute", "admin", "metrics"],
        "operator": ["read", "write", "execute", "metrics"],
        "developer": ["read", "write", "execute"],
        "viewer": ["read"],
        "api_key": ["read", "execute"],
    }

    def __init__(self, secret_key: str = None, default_expiration: int = 3600):
        self._key_manager = JWTKeyManager(secret_key)
        self._token_generator = TokenGenerator(self._key_manager)
        self._token_validator = TokenValidator(self._key_manager)
        self._default_expiration = default_expiration
        self._users = {}  # In production, use a proper user store

    def register_user(
        self,
        user_id: str,
        username: str,
        password_hash: str,
        email: str = None,
        roles: List[str] = None,
        scopes: List[str] = None,
    ) -> APIUser:
        """Register a new API user"""
        user = APIUser(
            user_id=user_id,
            username=username,
            email=email,
            scopes=scopes or self.SCOPES.get(roles[0] if roles else "viewer", ["read"]),
            roles=roles or ["viewer"],
            is_active=True,
            created_at=datetime.utcnow(),
            last_login=None,
        )
        self._users[user_id] = {"user": user, "password_hash": password_hash}
        return user

    def authenticate(self, username: str, password: str, grant_type: str = "password") -> tuple:
        """
        Authenticate a user and return tokens

        Returns:
            tuple: (success: bool, response: dict or None, error: str or None)
        """
        if grant_type not in ["password", "api_key"]:
            return False, None, "Unsupported grant type"

        # Find user
        user_data = None
        for uid, data in self._users.items():
            if data["user"].username == username:
                user_data = data
                break

        if not user_data:
            return False, None, "Invalid credentials"

        user = user_data["user"]

        if not user.is_active:
            return False, None, "Account is disabled"

        # Verify password (simplified - use proper password hashing in production)
        if grant_type == "password":
            expected_hash = hashlib.sha256(password.encode()).hexdigest()
            if user_data["password_hash"] != expected_hash:
                return False, None, "Invalid credentials"

        # Update last login
        user.last_login = datetime.utcnow()

        # Generate tokens
        tokens = self._token_generator.generate_token_pair(
            user_id=user.user_id, username=user.username, scopes=user.scopes
        )

        return True, tokens, None

    def refresh_access_token(self, refresh_token: str) -> tuple:
        """Refresh an access token using a refresh token"""
        success, payload, error = self._token_validator.validate(
            refresh_token, expected_type=TokenType.REFRESH
        )

        if not success:
            return False, None, error

        user_id = payload.get("user_id")
        username = payload.get("username")

        # Find user
        if user_id not in self._users:
            return False, None, "User not found"

        user = self._users[user_id]["user"]

        # Generate new access token
        new_access_token = self._token_generator.generate_access_token(
            user_id=user_id, username=username, scopes=user.scopes
        )

        return (
            True,
            {
                "access_token": new_access_token,
                "token_type": "Bearer",
                "expires_in": self._default_expiration,
            },
            None,
        )

    def revoke_token(self, token: str) -> tuple:
        """Revoke a token"""
        success, payload, error = self._token_validator.validate(token)

        if not success:
            return False, error

        jti = payload.get("jti")
        exp = payload.get("exp", 0)

        self._key_manager.revoke_token(jti, exp)

        return True, None

    def validate_access_token(self, token: str) -> tuple:
        """Validate an access token for API authentication"""
        return self._token_validator.validate(token, expected_type=TokenType.ACCESS)

    def get_token_info(self, token: str) -> Optional[TokenPayload]:
        """Get detailed information about a token"""
        return self._token_validator.get_token_payload(token)


def require_auth(f=None, *, required_scopes: List[str] = None):
    """Flask decorator to require JWT authentication.

    Can be used as:
        @require_auth
        def my_view():
            ...

    Or with scope requirements:
        @require_auth(required_scopes=["read", "metrics"])
        def my_view():
            ...
    """
    if f is not None:
        # Used as @require_auth without parentheses
        @wraps(f)
        def decorated_no_scopes(*args, **kwargs):
            return _verify_and_call(f, *args, **kwargs)

        return decorated_no_scopes
    else:
        # Used as @require_auth() or @require_auth(required_scopes=[...])
        def decorator(func):
            @wraps(func)
            def decorated_with_scopes(*args, **kwargs):
                if required_scopes:
                    # First verify auth
                    result = _verify_and_call(func, *args, **kwargs)
                    if hasattr(g, "scopes") and not all(s in g.scopes for s in required_scopes):
                        return jsonify(
                            {
                                "error": "insufficient_scope",
                                "message": f"Required scopes: {required_scopes}",
                                "required": required_scopes,
                                "current": getattr(g, "scopes", []),
                            }
                        ), 403
                    return result
                else:
                    return _verify_and_call(func, *args, **kwargs)

            return decorated_with_scopes

        return decorator


def _verify_and_call(f, *args, **kwargs):
    """Internal function to verify token and call the decorated function"""
    auth_header = request.headers.get("Authorization")

    if not auth_header:
        return jsonify(
            {
                "error": "missing_authorization_header",
                "message": "Authorization header is required",
            }
        ), 401

    try:
        scheme, token = auth_header.split(None, 1)
    except ValueError:
        return jsonify(
            {
                "error": "invalid_authorization_header",
                "message": "Invalid authorization header format",
            }
        ), 401

    if scheme.lower() != "bearer":
        return jsonify(
            {
                "error": "invalid_authorization_scheme",
                "message": "Authorization scheme must be Bearer",
            }
        ), 401

    # Validate token
    from applications.auth.jwt_auth import JWTAuthenticator

    authenticator = JWTAuthenticator()
    success, payload, error = authenticator.validate_access_token(token)

    if not success:
        return jsonify({"error": "invalid_token", "message": error}), 401

    # Store user info in Flask's g object
    g.current_user = payload.get("username")
    g.user_id = payload.get("user_id")
    g.scopes = payload.get("scopes", [])

    return f(*args, **kwargs)


def require_scope(required_scope: str):
    """Flask decorator to require a specific scope"""

    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not hasattr(g, "scopes") or required_scope not in g.scopes:
                return jsonify(
                    {
                        "error": "insufficient_scope",
                        "message": f"Required scope: {required_scope}",
                        "required": required_scope,
                        "current": getattr(g, "scopes", []),
                    }
                ), 403
            return f(*args, **kwargs)

        return decorated

    return decorator


# Example usage and initialization
def create_authenticator() -> JWTAuthenticator:
    """Create and initialize a JWT authenticator with demo users"""
    auth = JWTAuthenticator()

    # Register demo users
    auth.register_user(
        user_id="user-001",
        username="admin",
        password_hash=hashlib.sha256("admin123".encode()).hexdigest(),
        email="admin@neuralblitz.ai",
        roles=["admin"],
        scopes=auth.SCOPES["admin"],
    )

    auth.register_user(
        user_id="user-002",
        username="operator",
        password_hash=hashlib.sha256("operator123".encode()).hexdigest(),
        email="operator@neuralblitz.ai",
        roles=["operator"],
        scopes=auth.SCOPES["operator"],
    )

    auth.register_user(
        user_id="user-003",
        username="viewer",
        password_hash=hashlib.sha256("viewer123".encode()).hexdigest(),
        email="viewer@neuralblitz.ai",
        roles=["viewer"],
        scopes=auth.SCOPES["viewer"],
    )

    return auth


# Export main classes
__all__ = [
    "JWTAuthenticator",
    "JWTKeyManager",
    "TokenGenerator",
    "TokenValidator",
    "TokenPayload",
    "TokenType",
    "APIUser",
    "require_auth",
    "require_scope",
    "create_authenticator",
]
