"""
Authentication middleware for MCP HTTP server.

Supports multiple authentication modes: API Key, JWT, and Gateway.
"""

import logging
from typing import Optional

from fastapi import Header, HTTPException

from .config import QueryGenerationConfig

logger = logging.getLogger(__name__)


class AuthenticationMiddleware:
    """
    Authentication middleware for MCP HTTP endpoints.
    
    Supports three authentication modes:
    - api_key: Simple API key validation
    - jwt: JWT token validation
    - gateway: Trust upstream gateway (no validation)
    """
    
    def __init__(self, config: QueryGenerationConfig):
        """
        Initialize authentication middleware.
        
        Args:
            config: Server configuration
        """
        self.config = config
        self.enabled = config.mcp_auth_enabled
        self.mode = config.mcp_auth_mode
        
        if self.enabled:
            logger.info(f"Authentication enabled: mode={self.mode}")
            
            # Validate configuration based on mode
            if self.mode == "api_key" and not config.mcp_api_key:
                raise ValueError("MCP_API_KEY is required when using api_key auth mode")
            elif self.mode == "jwt" and not config.jwt_secret:
                raise ValueError("JWT_SECRET is required when using jwt auth mode")
        else:
            logger.warning("Authentication DISABLED - use only for development!")
    
    async def verify_api_key(self, x_api_key: Optional[str] = Header(None)) -> None:
        """
        Verify API key from X-API-Key header.
        
        Args:
            x_api_key: API key from request header
            
        Raises:
            HTTPException: If authentication fails
        """
        if not self.enabled:
            return  # Auth disabled
        
        if not x_api_key:
            logger.warning("API key authentication failed: missing X-API-Key header")
            raise HTTPException(status_code=401, detail="Missing X-API-Key header")
        
        if x_api_key != self.config.mcp_api_key:
            logger.warning("API key authentication failed: invalid key")
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        logger.debug("API key authentication successful")
    
    async def verify_jwt(self, authorization: Optional[str] = Header(None)) -> Optional[dict]:
        """
        Verify JWT token from Authorization header.
        
        Args:
            authorization: Authorization header value
            
        Returns:
            Decoded JWT payload if valid, None if auth disabled
            
        Raises:
            HTTPException: If authentication fails
        """
        if not self.enabled:
            return None  # Auth disabled
        
        if not authorization or not authorization.startswith("Bearer "):
            logger.warning("JWT authentication failed: missing or invalid Authorization header")
            raise HTTPException(
                status_code=401, 
                detail="Missing or invalid Authorization header. Expected: Bearer <token>"
            )
        
        token = authorization.split(" ")[1]
        
        try:
            import jwt as pyjwt
            
            payload = pyjwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=[self.config.jwt_algorithm]
            )
            
            logger.debug(f"JWT authentication successful for subject: {payload.get('sub', 'unknown')}")
            return payload
            
        except pyjwt.ExpiredSignatureError:
            logger.warning("JWT authentication failed: token expired")
            raise HTTPException(status_code=401, detail="Token expired")
        except pyjwt.InvalidTokenError as e:
            logger.warning(f"JWT authentication failed: {e}")
            raise HTTPException(status_code=401, detail="Invalid token")
        except ImportError:
            logger.error("PyJWT library not installed. Install with: pip install pyjwt")
            raise HTTPException(
                status_code=500, 
                detail="JWT authentication not available - PyJWT not installed"
            )
    
    async def verify_gateway(self) -> None:
        """
        Gateway authentication mode - trust upstream gateway.
        
        In this mode, we assume the upstream gateway (Kong, AWS API Gateway, etc.)
        has already validated the request. Optionally, you can add validation
        of gateway-specific headers here for defense-in-depth.
        """
        if not self.enabled:
            return  # Auth disabled
        
        # In gateway mode, we trust the upstream gateway
        # Optionally validate gateway signature headers here
        logger.debug("Gateway authentication: trusting upstream gateway")
    
    async def verify_request(
        self, 
        x_api_key: Optional[str] = Header(None),
        authorization: Optional[str] = Header(None)
    ) -> Optional[dict]:
        """
        Verify request based on configured authentication mode.
        
        Args:
            x_api_key: API key from X-API-Key header
            authorization: Authorization header for JWT
            
        Returns:
            Authentication payload (JWT claims) or None
            
        Raises:
            HTTPException: If authentication fails
        """
        if not self.enabled:
            return None
        
        if self.mode == "api_key":
            await self.verify_api_key(x_api_key)
            return None
        elif self.mode == "jwt":
            return await self.verify_jwt(authorization)
        elif self.mode == "gateway":
            await self.verify_gateway()
            return None
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Unknown authentication mode: {self.mode}"
            )

