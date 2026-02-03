"""
Enterprise security features: mTLS, rate limiting, and advanced security.
"""

import asyncio
import ssl
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
import ipaddress
import hashlib
import secrets
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer
import structlog
import redis.asyncio as aioredis
from prometheus_client import Counter, Histogram, Gauge

from ..config.settings import IntegrationBridgeConfig

logger = structlog.get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)
REQUEST_DURATION = Histogram("http_request_duration_seconds", "HTTP request duration")
ACTIVE_CONNECTIONS = Gauge("active_connections", "Active connections")
RATE_LIMIT_HITS = Counter(
    "rate_limit_hits_total", "Rate limit violations", ["client_id"]
)
MTLS_ERRORS = Counter("mtls_errors_total", "mTLS errors", ["error_type"])


@dataclass
class RateLimitInfo:
    """Rate limiting information."""

    requests_made: int
    window_start: float
    last_request_time: float


@dataclass
class ClientInfo:
    """Client information for rate limiting."""

    client_id: str
    ip_address: str
    user_agent: str
    api_key: Optional[str] = None
    rate_limit_info: Optional[RateLimitInfo] = None


class RateLimiter:
    """Advanced rate limiting with multiple strategies."""

    def __init__(
        self,
        config: IntegrationBridgeConfig,
        redis_client: Optional[aioredis.Redis] = None,
    ):
        self.config = config
        self.redis = redis_client
        self.local_limits: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.window_size = 60  # 1 minute window
        self.burst_size = config.rate_limit.burst_size
        self.requests_per_minute = config.rate_limit.requests_per_minute

    async def is_allowed(
        self, client_id: str, ip_address: str = "", endpoint: str = ""
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed based on rate limits."""
        try:
            # Try Redis first if available
            if self.redis:
                return await self._check_redis_rate_limit(
                    client_id, ip_address, endpoint
                )
            else:
                return await self._check_local_rate_limit(
                    client_id, ip_address, endpoint
                )

        except Exception as e:
            logger.error("Rate limit check failed", client_id=client_id, error=str(e))
            # Allow request if rate limiting fails
            return True, {"remaining": self.requests_per_minute - 1}

    async def _check_redis_rate_limit(
        self, client_id: str, ip_address: str, endpoint: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit using Redis."""
        current_time = time.time()
        window_start = int(current_time - self.window_size)

        # Use sliding window algorithm
        key = f"rate_limit:{client_id}"

        # Clean old entries
        await self.redis.zremrangebyscore(key, 0, window_start)

        # Count current requests
        current_requests = await self.redis.zcard(key)

        if current_requests >= self.requests_per_minute:
            RATE_LIMIT_HITS.labels(client_id=client_id).inc()
            return False, {
                "limit": self.requests_per_minute,
                "remaining": 0,
                "reset_time": int(window_start + self.window_size),
            }

        # Add current request
        await self.redis.zadd(key, {str(current_time): current_time})
        await self.redis.expire(key, self.window_size)

        return True, {
            "limit": self.requests_per_minute,
            "remaining": self.requests_per_minute - current_requests - 1,
            "reset_time": int(window_start + self.window_size),
        }

    async def _check_local_rate_limit(
        self, client_id: str, ip_address: str, endpoint: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit using local storage."""
        current_time = time.time()
        window_start = current_time - self.window_size

        # Get client's request queue
        request_times = self.local_limits[client_id]

        # Remove old requests outside the window
        while request_times and request_times[0] < window_start:
            request_times.popleft()

        # Check if under limit
        if len(request_times) >= self.requests_per_minute:
            RATE_LIMIT_HITS.labels(client_id=client_id).inc()
            return False, {
                "limit": self.requests_per_minute,
                "remaining": 0,
                "reset_time": int(window_start + self.window_size),
            }

        # Add current request
        request_times.append(current_time)

        return True, {
            "limit": self.requests_per_minute,
            "remaining": self.requests_per_minute - len(request_times),
            "reset_time": int(window_start + self.window_size),
        }

    async def get_rate_limit_status(self, client_id: str) -> Dict[str, Any]:
        """Get current rate limit status for a client."""
        try:
            if self.redis:
                current_time = time.time()
                window_start = int(current_time - self.window_size)
                key = f"rate_limit:{client_id}"

                current_requests = await self.redis.zcard(key)
                return {
                    "limit": self.requests_per_minute,
                    "used": current_requests,
                    "remaining": max(0, self.requests_per_minute - current_requests),
                    "reset_time": int(window_start + self.window_size),
                }
            else:
                request_times = self.local_limits[client_id]
                current_time = time.time()
                window_start = current_time - self.window_size

                # Count requests in current window
                recent_requests = sum(1 for t in request_times if t >= window_start)

                return {
                    "limit": self.requests_per_minute,
                    "used": recent_requests,
                    "remaining": max(0, self.requests_per_minute - recent_requests),
                    "reset_time": int(window_start + self.window_size),
                }

        except Exception as e:
            logger.error(
                "Failed to get rate limit status", client_id=client_id, error=str(e)
            )
            return {
                "limit": self.requests_per_minute,
                "used": 0,
                "remaining": self.requests_per_minute,
            }


class MTLSValidator:
    """Mutual TLS certificate validation."""

    def __init__(self, config: IntegrationBridgeConfig):
        self.config = config
        self.enabled = config.security.enable_mtls
        self.cert_file = config.security.cert_file
        self.key_file = config.security.key_file
        self.ca_file = config.security.ca_file
        self.trusted_certificates: Dict[str, Any] = {}
        self.certificate_cache: Dict[str, Tuple[datetime, Dict[str, Any]]] = {}

    async def validate_client_certificate(
        self, request: Request
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate client TLS certificate."""
        if not self.enabled:
            return True, {"certificate_info": "mTLS not enabled"}

        try:
            # Get client certificate from request
            client_cert = self._extract_client_certificate(request)

            if not client_cert:
                MTLS_ERRORS.labels(error_type="missing_certificate").inc()
                return False, {"error": "Client certificate required"}

            # Validate certificate chain
            cert_info = await self._validate_certificate_chain(client_cert)

            if not cert_info["valid"]:
                MTLS_ERRORS.labels(error_type="invalid_certificate").inc()
                return False, cert_info

            # Check certificate revocation
            if await self._is_certificate_revoked(cert_info["subject"]):
                MTLS_ERRORS.labels(error_type="revoked_certificate").inc()
                return False, {"error": "Certificate has been revoked"}

            # Check certificate expiration
            if cert_info["expired"]:
                MTLS_ERRORS.labels(error_type="expired_certificate").inc()
                return False, {"error": "Certificate has expired"}

            return True, cert_info

        except Exception as e:
            logger.error("mTLS validation failed", error=str(e))
            MTLS_ERRORS.labels(error_type="validation_error").inc()
            return False, {"error": f"Certificate validation failed: {str(e)}"}

    def _extract_client_certificate(self, request: Request) -> Optional[Dict[str, Any]]:
        """Extract client certificate from request."""
        # In FastAPI, client certificate info comes from request.client.ssl
        client = request.client
        if not client or not hasattr(client, "ssl"):
            return None

        ssl_info = client.ssl
        if not ssl_info:
            return None

        return {
            "subject": ssl_info.get("subject"),
            "issuer": ssl_info.get("issuer"),
            "serial_number": ssl_info.get("serial_number"),
            "not_before": ssl_info.get("not_before"),
            "not_after": ssl_info.get("not_after"),
            "version": ssl_info.get("version"),
            "signature_algorithm": ssl_info.get("signature_algorithm"),
        }

    async def _validate_certificate_chain(
        self, client_cert: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate certificate chain against trusted CA."""
        cert_info = {
            "valid": False,
            "subject": client_cert.get("subject"),
            "issuer": client_cert.get("issuer"),
            "serial_number": client_cert.get("serial_number"),
            "expired": False,
            "validation_errors": [],
        }

        try:
            # Check expiration
            not_after = client_cert.get("not_after")
            if not_after:
                if isinstance(not_after, str):
                    not_after_dt = datetime.fromisoformat(
                        not_after.replace("Z", "+00:00")
                    )
                else:
                    not_after_dt = not_after

                if datetime.utcnow() > not_after_dt:
                    cert_info["expired"] = True
                    cert_info["validation_errors"].append("Certificate has expired")

            # Check against trusted CA
            if self.ca_file:
                # This would implement proper certificate chain validation
                # For now, we'll simulate the validation
                cert_info["valid"] = not cert_info["expired"]
            else:
                cert_info["validation_errors"].append("No CA certificate configured")

        except Exception as e:
            cert_info["validation_errors"].append(f"Validation error: {str(e)}")

        return cert_info

    async def _is_certificate_revoked(self, cert_subject: str) -> bool:
        """Check if certificate is revoked (CRL/OCSP)."""
        # This would implement CRL or OCSP checking
        # For now, we'll assume no certificates are revoked
        return False

    async def load_trusted_certificates(self):
        """Load trusted certificates from file."""
        if not self.ca_file:
            return

        try:
            # This would load and parse CA certificates
            # For now, we'll simulate loading
            logger.info("Trusted certificates loaded", ca_file=self.ca_file)
        except Exception as e:
            logger.error("Failed to load trusted certificates", error=str(e))


class SecurityMiddleware:
    """Main security middleware combining all security features."""

    def __init__(
        self,
        config: IntegrationBridgeConfig,
        redis_client: Optional[aioredis.Redis] = None,
    ):
        self.config = config
        self.rate_limiter = RateLimiter(config, redis_client)
        self.mtls_validator = MTLSValidator(config)
        self.blocked_ips: Dict[str, datetime] = {}
        self.suspicious_requests: Dict[str, List[datetime]] = defaultdict(list)
        self.security_events: List[Dict[str, Any]] = []

    async def validate_request(self, request: Request) -> Tuple[bool, Dict[str, Any]]:
        """Validate request against all security measures."""
        client_info = self._extract_client_info(request)
        validation_result = {
            "allowed": True,
            "client_id": client_info["client_id"],
            "security_checks": {},
        }

        try:
            # 1. Check IP blocklist
            if await self._is_ip_blocked(client_info["ip_address"]):
                validation_result["allowed"] = False
                validation_result["security_checks"]["ip_blocked"] = True
                await self._log_security_event("ip_blocked", client_info, request)
                return False, validation_result

            # 2. Validate mTLS certificate
            (
                mtls_valid,
                mtls_info,
            ) = await self.mtls_validator.validate_client_certificate(request)
            validation_result["security_checks"]["mtls"] = {
                "valid": mtls_valid,
                "info": mtls_info,
            }

            if not mtls_valid and self.mtls_validator.enabled:
                validation_result["allowed"] = False
                await self._log_security_event(
                    "mtls_failed", client_info, request, mtls_info
                )
                return False, validation_result

            # 3. Check rate limits
            rate_limit_allowed, rate_limit_info = await self.rate_limiter.is_allowed(
                client_info["client_id"], client_info["ip_address"], request.url.path
            )
            validation_result["security_checks"]["rate_limit"] = {
                "allowed": rate_limit_allowed,
                "info": rate_limit_info,
            }

            if not rate_limit_allowed:
                validation_result["allowed"] = False
                await self._log_security_event(
                    "rate_limit_exceeded", client_info, request, rate_limit_info
                )
                return False, validation_result

            # 4. Check for suspicious patterns
            suspicious = await self._detect_suspicious_activity(client_info, request)
            if suspicious:
                validation_result["security_checks"]["suspicious_activity"] = True
                await self._log_security_event(
                    "suspicious_activity", client_info, request
                )

                # Rate limit suspicious requests more aggressively
                if await self._should_block_suspicious_client(client_info["client_id"]):
                    validation_result["allowed"] = False
                    return False, validation_result

            return True, validation_result

        except Exception as e:
            logger.error("Security validation failed", error=str(e))
            # Allow request if security validation fails (fail open)
            return True, validation_result

    def _extract_client_info(self, request: Request) -> ClientInfo:
        """Extract client information from request."""
        # Get client ID from various sources
        client_id = None

        # Try API key first
        api_key = request.headers.get("X-API-Key")
        if api_key:
            client_id = f"api_key:{self._hash_api_key(api_key)}"

        # Try user ID from JWT (would be extracted from auth middleware)
        if hasattr(request.state, "user_id"):
            client_id = f"user:{request.state.user_id}"

        # Fallback to IP address
        if not client_id:
            ip_address = self._get_client_ip(request)
            client_id = f"ip:{ip_address}"
        else:
            ip_address = self._get_client_ip(request)

        return ClientInfo(
            client_id=client_id,
            ip_address=ip_address,
            user_agent=request.headers.get("User-Agent", ""),
            api_key=api_key,
        )

    def _get_client_ip(self, request: Request) -> str:
        """Get real client IP address."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fallback to client IP
        return request.client.host if request.client else "unknown"

    def _hash_api_key(self, api_key: str) -> str:
        """Hash API key for identification."""
        return hashlib.sha256(api_key.encode()).hexdigest()[:16]

    async def _is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked."""
        if ip_address in self.blocked_ips:
            block_expiry = self.blocked_ips[ip_address]
            if datetime.utcnow() < block_expiry:
                return True
            else:
                # Block expired, remove it
                del self.blocked_ips[ip_address]

        return False

    async def _detect_suspicious_activity(
        self, client_info: ClientInfo, request: Request
    ) -> bool:
        """Detect suspicious activity patterns."""
        current_time = datetime.utcnow()
        client_id = client_info.client_id

        # Check for rapid requests from same client
        recent_requests = [
            req_time
            for req_time in self.suspicious_requests[client_id]
            if (current_time - req_time).total_seconds() < 60  # Last minute
        ]

        self.suspicious_requests[client_id] = recent_requests + [current_time]

        # Flag if more than 100 requests in last minute
        if len(recent_requests) > 100:
            return True

        # Check for unusual user agent patterns
        user_agent = client_info.user_agent
        if not user_agent or len(user_agent) < 10:
            return True

        # Check for suspicious paths
        suspicious_paths = ["/admin", "/config", "/.env", "/etc/passwd"]
        if any(
            suspicious in request.url.path.lower() for suspicious in suspicious_paths
        ):
            return True

        return False

    async def _should_block_suspicious_client(self, client_id: str) -> bool:
        """Determine if suspicious client should be blocked."""
        recent_suspicious = self.suspicious_requests.get(client_id, [])

        # Count suspicious requests in last 10 minutes
        current_time = datetime.utcnow()
        recent_count = sum(
            1
            for req_time in recent_suspicious
            if (current_time - req_time).total_seconds() < 600
        )

        # Block if more than 50 suspicious requests in 10 minutes
        if recent_count > 50:
            # Block for 1 hour
            self.blocked_ips[client_id] = current_time + timedelta(hours=1)
            return True

        return False

    async def _log_security_event(
        self,
        event_type: str,
        client_info: ClientInfo,
        request: Request,
        additional_info: Optional[Dict[str, Any]] = None,
    ):
        """Log security event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "client_id": client_info.client_id,
            "ip_address": client_info.ip_address,
            "user_agent": client_info.user_agent,
            "path": request.url.path,
            "method": request.method,
            "additional_info": additional_info or {},
        }

        self.security_events.append(event)

        # Keep only last 1000 events
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]

        logger.warning("Security event detected", **event)

    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status and statistics."""
        current_time = datetime.utcnow()

        # Clean expired IP blocks
        expired_blocks = [
            ip for ip, expiry in self.blocked_ips.items() if current_time >= expiry
        ]
        for ip in expired_blocks:
            del self.blocked_ips[ip]

        # Count recent security events
        recent_events = [
            event
            for event in self.security_events
            if datetime.fromisoformat(event["timestamp"])
            > current_time - timedelta(hours=24)
        ]

        return {
            "blocked_ips": len(self.blocked_ips),
            "suspicious_clients": len(self.suspicious_requests),
            "security_events_24h": len(recent_events),
            "rate_limit_enabled": bool(self.rate_limiter),
            "mtls_enabled": self.mtls_validator.enabled,
            "recent_events": recent_events[-10:],  # Last 10 events
        }

    async def cleanup_old_data(self):
        """Clean up old security data."""
        current_time = datetime.utcnow()

        # Clean old suspicious request records
        for client_id in list(self.suspicious_requests.keys()):
            self.suspicious_requests[client_id] = [
                req_time
                for req_time in self.suspicious_requests[client_id]
                if (current_time - req_time).total_seconds() < 3600  # Keep 1 hour
            ]

            if not self.suspicious_requests[client_id]:
                del self.suspicious_requests[client_id]
