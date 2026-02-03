"""
Advanced security features: API signing, role-based rate limiting, threat detection.
"""

import time
import hmac
import hashlib
import secrets
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import re
import ipaddress
import structlog
import asyncio
from datetime import datetime, timedelta

from ..config.settings import IntegrationBridgeConfig
from ..auth.middleware import AuthenticationMiddleware
from ..utils.timeout_handler import TimeoutManager

logger = structlog.get_logger(__name__)


class SecurityLevel(Enum):
    """Security levels for rate limiting."""
    ANONYMOUS = "anonymous"
    AUTHENTICATED = "authenticated"
    PREMIUM = "premium"
    ADMIN = "admin"
    ENTERPRISE = "enterprise"


class ThreatType(Enum):
    """Types of security threats."""
    RATE_LIMIT = "rate_limit"
    MALICIOUS_REQUEST = "malicious_request"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    INJECTION_ATTEMPT = "injection_attempt"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    BRUTE_FORCE = "brute_force"


@dataclass
class APIKey:
    """API key for request signing."""
    
    key_id: str
    key_secret: str
    key_id_hex: str = field(init=False)
    permissions: List[str] = field(default_factory=list)
    rate_limits: Dict[str, int] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_used: datetime = field(default_factory=datetime.utcnow)
    usage_count: int = 0
    is_active: bool = True


@dataclass
class SecurityEvent:
    """Security event for monitoring."""
    
    timestamp: datetime
    threat_type: ThreatType
    source_ip: str
    user_agent: Optional[str] = None
    endpoint: str
    user_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "low"  # low, medium, high, critical
    blocked: bool = False
    risk_score: float = 0.0


class RequestSigner:
    """API request signing for integrity and non-repudiation."""
    
    def __init__(self, config: IntegrationBridgeConfig):
        self.config = config
        self.signing_secret = secrets.token_urlsafe(32)  # Generate secure secret
        
        # Initialize HMAC SHA256
        self.signing_algorithm = hashlib.sha256
        
    def _generate_signature(self, method: str, url: str, payload: str, timestamp: str) -> str:
        """Generate HMAC signature for API request."""
        # Create canonical string to sign
        canonical_string = f"{method}\n{url}\n{timestamp}\n{payload}"
        
        signature = self.signing_algorithm(
            self.signing_secret.encode('utf-8'),
            canonical_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        logger.debug("Generated signature", method=method, url=url[:50])
        return signature
    
    def sign_request(self, method: str, url: str, payload: Any, headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Sign an API request."""
        # Add timestamp if not provided
        timestamp = headers.get('X-API-Timestamp') if headers else str(int(time.time()))
        
        # Convert payload to string
        if isinstance(payload, dict):
            payload_str = json.dumps(payload, sort_keys=True)
        elif isinstance(payload, str):
            payload_str = payload
        else:
            payload_str = str(payload)
        
        signature = self._generate_signature(method, url, payload_str, timestamp)
        
        signed_headers = {
            'X-API-Signature': signature,
            'X-API-Timestamp': timestamp,
        }
        
        # Add signature to existing headers
        if headers:
            signed_headers.update(headers)
        
        return signed_headers
    
    def verify_signature(self, method: str, url: str, payload: Any, signature: str, timestamp: str) -> bool:
        """Verify an API request signature."""
        # Recreate canonical string
        canonical_string = f"{method}\n{url}\n{timestamp}\n{payload}"
        
        expected_signature = self.signing_algorithm(
            self.signing_secret.encode('utf-8'),
            canonical_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        # Use constant-time comparison to prevent timing attacks
        return hmac.compare_digest(expected_signature.encode(), signature.encode())
    
    def verify_request(self, method: str, url: str, payload: Any, headers: Dict[str, str]) -> Dict[str, str]:
        """Verify API request signature."""
        signature = headers.get('X-API-Signature', '')
        timestamp = headers.get('X-API-Timestamp', '')
        payload_str = json.dumps(payload, sort_keys=True) if isinstance(payload, dict) else str(payload)
        
        if not signature or not timestamp:
            return {"valid": False, "error": "Missing signature or timestamp"}
        
        # Check timestamp freshness (5 minutes)
        request_time = datetime.fromisoformat(timestamp)
        current_time = datetime.utcnow()
        
        if current_time - request_time > timedelta(minutes=5):
            return {"valid": False, "error": "Request too old"}
        
        # Verify signature
        is_valid = self.verify_signature(method, url, payload_str, signature, timestamp)
        
        return {"valid": is_valid, "error": "" if is_valid else "Invalid signature"}


class RoleBasedRateLimiter:
    """Rate limiting based on user roles and API keys."""
    
    def __init__(self, config: IntegrationBridgeConfig):
        self.config = config
        self.api_keys: Dict[str, APIKey] = {}
        self.request_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.blocked_keys: Set[str] = set()
        self.security_events: List[SecurityEvent] = []
        self.max_history = 10000
        
        # Initialize default limits
        self.default_limits = {
            SecurityLevel.ANONYMOUS: 100,
            SecurityLevel.AUTHENTICATED: 1000,
            SecurityLevel.PREMIUM: 5000,
            SecurityLevel.ADMIN: 10000
        }
    
    def load_api_key(self, key_id: str, key_secret: str, permissions: List[str]) -> APIKey:
        """Load or create an API key."""
        api_key = APIKey(
            key_id=key_id,
            key_secret=key_secret,
            key_id_hex=hashlib.sha256(key_id.encode()).hexdigest(),
            permissions=permissions,
            created_at=datetime.utcnow(),
            last_used=datetime.utcnow()
        )
        
        # Set rate limits based on security level
        if "admin" in permissions:
            api_key.rate_limits = {
                "requests_per_minute": self.default_limits[SecurityLevel.ADMIN],
                "requests_per_hour": 100000
            "requests_per_day": 1000000
            }
        elif "premium" in permissions:
            api_key.rate_limits = {
                "requests_per_minute": self.default_limits[SecurityLevel.PREMIUM],
                "requests_per_hour": 5000,
                "requests_per_day": 50000
            }
        else:
            api_key.rate_limits = {
                "requests_per_minute": self.default_limits[SecurityLevel.AUTHENTICATED],
                "requests_per_hour": 2000,
                "requests_per_day": 20000
            }
        
        self.api_keys[key_id] = api_key
        logger.info("API key loaded", key_id=key_id, permissions=permissions)
        
        return api_key
    
    def get_api_key(self, key_id: str) -> Optional[APIKey]:
        """Get API key by ID."""
        return self.api_keys.get(key_id)
    
    def check_rate_limit(self, key_id: str, endpoint: str) -> Dict[str, Any]:
        """Check rate limit for API key and endpoint."""
        api_key = self.api_keys.get(key_id)
        
        if not api_key:
            return {"allowed": False, "error": "Invalid API key"}
        
        # Check if key is blocked
        if key_id in self.blocked_keys:
            return {"allowed": False, "error": "API key blocked"}
        
        # Check if key is expired
        if api_key.expires_at and datetime.utcnow() > api_key.expires_at:
            return {"allowed": False, "error": "API key expired"}
        
        # Get current request count
        current_time = datetime.utcnow()
        request_counts = self.request_counts[key_id]
        
        # Clean old counts (older than 1 hour)
        cutoff_time = current_time - timedelta(hours=1)
        while request_counts and request_counts[0] < cutoff_time:
            request_counts.popleft()
        
        # Add current request
        request_counts.append(current_time)
        
        # Check rate limits
        rate_limit = api_key.rate_limits.get(f"endpoint:{endpoint}", 100)
        
        # Sliding window (1 minute)
        recent_requests = [
            req_time for req_time in request_counts
            if current_time - req_time < timedelta(minutes=1)
        ]
        
        if len(recent_requests) >= rate_limit:
            # Record rate limit violation
            self._record_security_event(
                ThreatType.RATE_LIMIT,
                "Unknown",  # Would get real IP
                endpoint,
                {"api_key": key_id, "rate_limit": rate_limit, "recent_requests": len(recent_requests)}
            )
            
            return {"allowed": False, "error": "Rate limit exceeded", "retry_after": str(60 - (current_time - recent_requests[0]).seconds if recent_requests else 60)}
        
        return {"allowed": True, "remaining": max(0, rate_limit - len(recent_requests))}
    
    def record_api_usage(self, key_id: str, endpoint: str, success: bool = True):
        """Record API key usage."""
        api_key = self.api_keys.get(key_id)
        
        if api_key:
            api_key.usage_count += 1
            api_key.last_used = datetime.utcnow()
            
            if not success:
                logger.warning("API key usage failed", key_id=key_id, endpoint=endpoint)
    
    def block_api_key(self, key_id: str, reason: str, duration_hours: int = 24):
        """Block an API key."""
        self.blocked_keys.add(key_id)
        logger.warning("API key blocked", key_id=key_id, reason=reason)
        
        # Schedule unblocking
        asyncio.create_task(self._unblock_api_key(key_id, duration_hours))
    
    async def _unblock_api_key(self, key_id: str, duration_hours: int):
        """Unblock an API key after specified duration."""
        await asyncio.sleep(duration_hours * 3600)  # Convert to seconds
        
        if key_id in self.blocked_keys:
            self.blocked_keys.remove(key_id)
            logger.info("API key unblocked", key_id=key_id)
    
    def _record_security_event(
        self, 
        threat_type: ThreatType,
        source_ip: str,
        endpoint: str,
        user_agent: Optional[str] = None,
        details: Dict[str, Any],
        severity: str = "medium"
    ):
        """Record a security event."""
        event = SecurityEvent(
            timestamp=datetime.utcnow(),
            threat_type=threat_type,
            source_ip=source_ip,
            user_agent=user_agent,
            endpoint=endpoint,
            details=details,
            severity=severity,
            blocked=False
        )
        
        # Calculate risk score
        if threat_type == ThreatType.BRUTE_FORCE:
            event.risk_score = 0.9
        elif threat_type == ThreatType.INJECTION_ATTEMPT:
            event.risk_score = 0.8
        elif threat_type == ThreatType.MALICIOUS_REQUEST:
            event.risk_score = 0.6
        elif threat_type == ThreatType.UNAUTHORIZED_ACCESS:
            event.risk_score = 0.5
        else:
            event.risk_score = 0.3
        
        # Block if high risk
        if event.risk_score > 0.7:
            event.blocked = True
            if event.source_ip and event.source_ip != "127.0.0.1":
                self._block_ip_temporarily(event.source_ip, hours=1)
        
        self.security_events.append(event)
        
        # Trim history
        if len(self.security_events) > self.max_history:
            self.security_events = self.security_events[-self.max_history:]
        
        logger.warning("Security event recorded", threat_type=threat_type.value, risk_score=event.risk_score)
    
    def _block_ip_temporarily(self, ip: str, hours: int = 1):
        """Block an IP address temporarily."""
        # This would integrate with firewall or load balancer
        logger.warning("IP temporarily blocked", ip=ip, hours=hours)
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status."""
        current_time = datetime.utcnow()
        
        # Analyze recent events
        recent_events = [
            event for event in self.security_events
            if current_time - event.timestamp < timedelta(hours=24)
        ]
        
        # Calculate statistics
        threat_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for event in recent_events:
            threat_counts[event.threat_type.value] += 1
            severity_counts[event.severity] += 1
        
        return {
            "recent_events_24h": len(recent_events),
            "threat_counts": dict(threat_counts),
            "severity_counts": dict(severity_counts),
            "blocked_ips_count": len(self.blocked_keys),
            "active_api_keys": len([k for k in self.api_keys if k.is_active]),
            "high_risk_events": len([e for e in recent_events if e.risk_score > 0.7])
        }


class ThreatDetection:
    """Advanced threat detection using patterns and ML."""
    
    def __init__(self):
        self.suspicious_patterns = {
            # SQL injection patterns
            "sql_injection": [
                r"(?i)*(\b(|\')|\\b|(?i)*(union|select|insert|delete|update|drop|create|alter))",
                r"(\b("|')(\w|;|\-\-)|(?i)*(\b(|\')|\\b|(?i))*(\b(|\')|\\b)(union|select|insert|delete|update|drop|create|alter))",
                r"(\b(|\')|\\b)(\b(|')|\\b)(union|select|insert|delete|update|drop|create|alter))",
            ],
            
            # XSS patterns
            "xss": [
                r"<script[^>]*>.*?</script>",
                r"javascript:|eval|vbscript|vbscript:",
                r"onerror\s*=.*[\"']",
                r"<iframe.*>",
            ],
            
            # Directory traversal
            "path_traversal": [
                r"\.\./|\.\.\\|/\.\.\.\\",
                r"\.\.[/]?",
                r"(\.(/\.\.|%2f){3,}.*",
            ],
            
            # Command injection
            "command_injection": [
                r"[;|&|`|$]",
                r"\|\|\|\\|([^\\w]|\\n\r)(cd|ls|pwd|echo|cat|grep|kill|)",
                r"(wget|curl|nc|netcat|ssh)",
            ],
        }
        
        self.request_patterns = defaultdict(int)
    
    def analyze_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a request for threats."""
        method = request.get("method", "GET")
        url = request.get("url", "")
        headers = request.get("headers", {})
        payload = request.get("payload", "")
        user_agent = headers.get("User-Agent", "")
        source_ip = request.get("source_ip", "127.0.0.1")  # Default for testing
        
        risk_score = 0.0
        threats = []
        
        # Check URL patterns
        for pattern_name, patterns in self.suspicious_patterns.items():
            for pattern in patterns:
                if pattern.search(url):
                    threats.append(f"Potential {pattern_name}: {url}")
                    risk_score += 0.2
        
        # Check payload for injection attempts
        if isinstance(payload, str):
            for pattern_name, patterns in self.suspicious_patterns.items():
                if pattern.search(payload):
                    threats.append(f"Potential {pattern_name} in payload")
                    risk_score += 0.3
        
        # Check User-Agent for suspicious patterns
        if user_agent:
            suspicious_agents = [
                "sqlmap", "sqlninja", "shell", "wget", "curl", "nc", "nmap"
            ]
            
            for agent in suspicious_agents:
                if agent.lower() in user_agent.lower():
                    threats.append(f"Suspicious user agent: {agent}")
                    risk_score += 0.2
        
        # Check for common attack patterns
        payload_str = str(payload).lower()
        attack_patterns = [
            "../../etc/passwd",
            "..\\windows\\system32\\config\\system",  # Windows
            "cd /tmp; rm -rf /",  # Unix commands
            "eval(", "exec(", "system(", "subprocess(",
        ]
        
        for pattern in attack_patterns:
            if pattern in payload_str:
                threats.append(f"Potential system command injection: {pattern}")
                risk_score += 0.4
        
        # Rate limiting detection
        client_ip = source_ip
        current_time = time.time()
        
        if client_ip in self.request_patterns:
            recent_requests = self.request_patterns[client_ip]
            request_count = sum(1 for _ in recent_requests if current_time - _ < 60)
            
            if request_count > 10:  # 10 requests per minute
                threats.append("High request rate")
                risk_score += 0.5
        
        # Final risk assessment
        if risk_score > 0.8:
            severity = "high"
        elif risk_score > 0.5:
            severity = "medium"
        else:
            severity = "low"
        
        return {
            "risk_score": risk_score,
            "severity": severity,
            "threats": threats,
            "recommendation": self._get_recommendation(risk_score, threats)
        }
    
    def _get_recommendation(self, risk_score: float, threats: List[str]) -> str:
        """Get security recommendation based on risk analysis."""
        if risk_score > 0.8:
            return "Block request and investigate"
        elif risk_score > 0.5:
            return "Manual review required"
        elif "sql_injection" in " ".join(threats):
            return "SQL injection detected"
        elif "xss" in " ".join(threats):
            return "XSS vulnerability detected"
        else:
            return "Continue monitoring"


class AdvancedSecurityMiddleware(AuthenticationMiddleware):
    """Advanced security middleware with threat detection and API signing."""
    
    def __init__(self, config: IntegrationBridgeConfig):
        super().__init__(config)
        self.request_signer = RequestSigner(config)
        self.role_based_limiter = RoleBasedRateLimiter(config)
        self.threat_detector = ThreatDetection()
        self.api_keys: Dict[str, APIKey] = {}
        
        # Load API keys
        self._load_api_keys()
        
        logger.info("Advanced security middleware initialized")
    
    async def authenticate_and_validate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate request with advanced validation."""
        # Basic authentication
        auth_result = await super().authenticate_request(request)
        
        if not auth_result["allowed"]:
            return auth_result
        
        # Additional API key validation if required
        api_key = request.get("headers", {}).get("X-API-Key")
        if api_key:
            key_validation = self.role_based_limiter.get_api_key(api_key)
            
            if not key_validation["allowed"]:
                return {"allowed": False, "error": "Invalid API key"}
        
        # Request signature verification
        signature_result = self.request_signer.verify_request(
            method=request.get("method", "GET"),
            url=request.get("url", ""),
            payload=request.get("payload", {}),
            headers=request.get("headers", {})
        )
        
        if not signature_result["valid"]:
            # Record signature verification failure
            self.threat_detector._record_security_event(
                ThreatType.MALICIOUS_REQUEST,
                request.get("source_ip", "unknown"),
                request.get("endpoint", "unknown"),
                request.get("user_agent", "unknown"),
                {"error": signature_result["error"], "validation": signature_result}
            )
            
            return {"allowed": False, "error": "Invalid signature"}
        
        # Threat detection
        threat_analysis = self.threat_detector.analyze_request(request)
        
        if threat_analysis["risk_score"] > 0.7:
            # High risk - block request
            self.threat_detector._record_security_event(
                threat_analysis["severity"],
                request.get("source_ip", "unknown"),
                request.get("endpoint", "unknown"),
                request.get("user_agent", "unknown"),
                {"threats": threat_analysis["threats"], "recommendation": threat_analysis["recommendation"]}
            )
            
            return {"allowed": False, "error": "Request blocked due to security threats"}
        
        # Enhanced authentication result
        return {
            **auth_result,
            "api_key_validation": key_validation if api_key else {"allowed": True},
            "signature_verification": signature_result,
            "threat_analysis": threat_analysis
        }
    
    def _load_api_keys(self):
        """Load API keys from configuration or database."""
        # This would load from a secure storage in production
        # For now, create a default admin key
        self.api_keys["admin_key"] = self.role_based_limiter.load_api_key(
            "admin_key",
            "development_admin_secret",
            ["*"]
        )
        logger.info("API keys loaded")


# Global advanced security manager
advanced_security_manager: Optional[AdvancedSecurityMiddleware] = None


async def get_advanced_security_manager(config: IntegrationBridgeConfig) -> AdvancedSecurityMiddleware:
    """Get or create global advanced security manager."""
    global advanced_security_manager
    
    if advanced_security_manager is None:
        advanced_security_manager = AdvancedSecurityMiddleware(config)
        logger.info("Advanced security manager initialized")
    
    return advanced_security_manager