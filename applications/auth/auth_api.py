"""
NeuralBlitz v50 Authentication API
Provides secure token-based authentication endpoints
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from flask import Blueprint, request, jsonify, g
from functools import wraps
import jwt
import time
from datetime import datetime, timedelta

# Import JWT authentication modules
from auth.jwt_auth import (
    JWTAuthenticator,
    TokenType,
    require_auth,
    require_scope,
    create_authenticator,
)

# Create Blueprint for authentication routes
auth_bp = Blueprint("auth", __name__, url_prefix="/api/v1/auth")

# Initialize authenticator (singleton pattern)
_authenticator = None


def get_authenticator():
    """Get or create the JWT authenticator instance"""
    global _authenticator
    if _authenticator is None:
        _authenticator = create_authenticator()
    return _authenticator


@auth_bp.route("/health", methods=["GET"])
def health_check():
    """Health check for auth service"""
    return jsonify(
        {
            "status": "healthy",
            "service": "neuralblitz-auth",
            "timestamp": int(time.time()),
        }
    )


@auth_bp.route("/token", methods=["POST"])
def token():
    """
    OAuth2-style token endpoint
    Grants: password, refresh_token, api_key
    """
    authenticator = get_authenticator()

    # Get request parameters
    grant_type = request.form.get("grant_type", "password")
    username = request.form.get("username")
    password = request.form.get("password")
    refresh_token = request.form.get("refresh_token")
    api_key = request.form.get("api_key")

    # Validate grant type
    if grant_type not in ["password", "refresh_token", "api_key"]:
        return jsonify(
            {
                "error": "unsupported_grant_type",
                "error_description": "Grant type must be 'password', 'refresh_token', or 'api_key'",
            }
        ), 400

    # Authenticate based on grant type
    if grant_type == "password":
        if not username or not password:
            return jsonify(
                {
                    "error": "invalid_request",
                    "error_description": "Username and password are required",
                }
            ), 400

        success, response, error = authenticator.authenticate(
            username=username, password=password, grant_type="password"
        )

    elif grant_type == "refresh_token":
        if not refresh_token:
            return jsonify(
                {
                    "error": "invalid_request",
                    "error_description": "Refresh token is required",
                }
            ), 400

        success, response, error = authenticator.refresh_access_token(refresh_token)

        if success:
            # Also return a new refresh token (token rotation)
            authenticator = get_authenticator()
            success2, response2, error2 = authenticator.authenticate(
                username=response["username"] if "username" in response else "unknown",
                password="",  # Not needed for refresh
                grant_type="refresh_token",
            )

    elif grant_type == "api_key":
        if not api_key:
            return jsonify(
                {"error": "invalid_request", "error_description": "API key is required"}
            ), 400

        # For API key grant, we need a different flow
        # This is a simplified implementation
        success = False
        error = "API key authentication not implemented"
        response = None

    if not success:
        return jsonify({"error": "invalid_grant", "error_description": error}), 401

    return jsonify(response)


@auth_bp.route("/introspect", methods=["POST"])
@require_auth()
def introspect():
    """
    Token introspection endpoint
    Returns information about the current access token
    """
    authenticator = get_authenticator()

    # Get token from Authorization header
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return jsonify({"active": False, "reason": "No token provided"}), 200

    try:
        scheme, token = auth_header.split(None, 1)
        if scheme.lower() != "bearer":
            raise ValueError()
    except ValueError:
        return jsonify({"active": False, "reason": "Invalid authorization header"}), 200

    # Validate token
    success, payload, error = authenticator.validate_access_token(token)

    if not success:
        return jsonify({"active": False, "reason": error}), 200

    # Return token information
    return jsonify(
        {
            "active": True,
            "client_id": payload.get("client_id", payload.get("username")),
            "username": payload.get("username"),
            "scope": " ".join(payload.get("scopes", [])),
            "sub": payload.get("sub"),
            "iss": payload.get("iss"),
            "aud": payload.get("aud"),
            "exp": payload.get("exp"),
            "iat": payload.get("iat"),
            "nbf": payload.get("nbf"),
            "token_type": payload.get("type"),
        }
    )


@auth_bp.route("/revoke", methods=["POST"])
@require_auth()
def revoke():
    """
    Token revocation endpoint
    Revokes the current access token or a refresh token
    """
    authenticator = get_authenticator()

    # Get token to revoke
    token = request.form.get("token")
    token_type_hint = request.form.get("token_type_hint", "access_token")

    if not token:
        return jsonify(
            {"error": "invalid_request", "error_description": "Token is required"}
        ), 400

    # Revoke the token
    success, error = authenticator.revoke_token(token)

    if not success:
        return jsonify({"error": "invalid_token", "error_description": error}), 400

    # Token revocation should return 200 even if token was invalid
    # (RFC 7009: OAuth 2.0 Token Revocation)
    return jsonify({"message": "Token revoked successfully"}), 200


@auth_bp.route("/userinfo", methods=["GET"])
@require_auth()
def userinfo():
    """
    User information endpoint
    Returns information about the authenticated user
    """
    authenticator = get_authenticator()

    # Get token info
    auth_header = request.headers.get("Authorization")
    try:
        scheme, token = auth_header.split(None, 1)
    except ValueError:
        return jsonify({"error": "Invalid authorization header"}), 401

    token_info = authenticator.get_token_info(token)

    if not token_info:
        return jsonify({"error": "Invalid token"}), 401

    return jsonify(
        {
            "user_id": token_info.user_id,
            "username": token_info.subject,  # Using subject as username
            "scopes": token_info.scopes,
            "token_type": token_info.token_type.value,
            "issued_at": datetime.fromtimestamp(token_info.issued_at).isoformat(),
            "expires_at": datetime.fromtimestamp(token_info.expires_at).isoformat(),
        }
    )


@auth_bp.route("/register", methods=["POST"])
def register():
    """
    User registration endpoint
    Creates a new API user account
    """
    data = request.get_json()

    if not data:
        return jsonify(
            {
                "error": "invalid_request",
                "error_description": "Request body is required",
            }
        ), 400

    username = data.get("username")
    password = data.get("password")
    email = data.get("email")
    role = data.get("role", "viewer")

    # Validate required fields
    if not username or not password:
        return jsonify(
            {
                "error": "invalid_request",
                "error_description": "Username and password are required",
            }
        ), 400

    # Validate username format
    if len(username) < 3 or len(username) > 50:
        return jsonify(
            {
                "error": "invalid_request",
                "error_description": "Username must be between 3 and 50 characters",
            }
        ), 400

    # Validate password strength
    if len(password) < 8:
        return jsonify(
            {
                "error": "invalid_request",
                "error_description": "Password must be at least 8 characters",
            }
        ), 400

    # Validate role
    valid_roles = ["admin", "operator", "developer", "viewer"]
    if role not in valid_roles:
        return jsonify(
            {
                "error": "invalid_request",
                "error_description": f"Role must be one of: {', '.join(valid_roles)}",
            }
        ), 400

    authenticator = get_authenticator()

    # Check if username already exists (simplified check)
    # In production, check against user database
    for uid, data in authenticator._users.items():
        if data["user"].username == username:
            return jsonify(
                {
                    "error": "invalid_request",
                    "error_description": "Username already exists",
                }
            ), 409

    # Generate user ID
    import hashlib

    user_id = f"user-{hashlib.md5(username.encode()).hexdigest()[:8]}"

    # Hash password (simplified - use bcrypt/argon2 in production)
    import hashlib

    password_hash = hashlib.sha256(password.encode()).hexdigest()

    # Get scopes for role
    from auth.jwt_auth import JWTAuthenticator

    scopes = JWTAuthenticator.SCOPES.get(role, JWTAuthenticator.SCOPES["viewer"])

    # Register user
    user = authenticator.register_user(
        user_id=user_id,
        username=username,
        password_hash=password_hash,
        email=email,
        roles=[role],
        scopes=scopes,
    )

    return jsonify(
        {
            "message": "User registered successfully",
            "user": {
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "roles": user.roles,
                "scopes": user.scopes,
            },
        }
    ), 201


@auth_bp.route("/keys", methods=["GET"])
@require_auth()
@require_scope("admin")
def get_public_keys():
    """
    Get public keys for token verification (JWKS endpoint)
    Returns the public keys used for token verification
    """
    import json

    authenticator = get_authenticator()

    # Get key manager's verification keys
    keys = authenticator._key_manager.get_verification_keys()

    # Format as JWKS
    jwks = {"keys": []}

    for kid, key in keys.items():
        jwks["keys"].append(
            {
                "kty": "oct",  # Symmetric key
                "alg": "HS512",
                "use": "sig",
                "kid": kid,
                "k": key,  # Base64url encoded key
            }
        )

    return jsonify(jwks)


# Error handlers
@auth_bp.errorhandler(400)
def bad_request(error):
    return jsonify(
        {"error": "bad_request", "error_description": str(error.description)}
    ), 400


@auth_bp.errorhandler(401)
def unauthorized(error):
    return jsonify(
        {"error": "unauthorized", "error_description": "Authentication required"}
    ), 401


@auth_bp.errorhandler(403)
def forbidden(error):
    return jsonify(
        {"error": "forbidden", "error_description": "Insufficient permissions"}
    ), 403


@auth_bp.errorhandler(500)
def internal_error(error):
    return jsonify(
        {
            "error": "internal_server_error",
            "error_description": "An internal error occurred",
        }
    ), 500


# Demo credentials helper
@auth_bp.route("/demo", methods=["GET"])
def get_demo_credentials():
    """
    Get demo credentials for testing
    WARNING: Only for development/testing!
    """
    return jsonify(
        {
            "message": "Demo credentials for testing",
            "users": [
                {
                    "username": "admin",
                    "password": "admin123",
                    "roles": ["admin"],
                    "scopes": ["read", "write", "execute", "admin", "metrics"],
                },
                {
                    "username": "operator",
                    "password": "operator123",
                    "roles": ["operator"],
                    "scopes": ["read", "write", "execute", "metrics"],
                },
                {
                    "username": "viewer",
                    "password": "viewer123",
                    "roles": ["viewer"],
                    "scopes": ["read"],
                },
            ],
            "note": "These credentials are for testing only. Change in production!",
        }
    )


# Export Blueprint
__all__ = ["auth_bp"]
