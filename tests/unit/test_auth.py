"""
Unit tests for authentication middleware.
"""

import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pytest_mock.plugin import MockerFixture

from query_generation_agent.mcp.auth import AuthenticationMiddleware
from query_generation_agent.mcp.config import QueryGenerationConfig


def test_auth_middleware_disabled() -> None:
    """Test authentication middleware when disabled."""
    config = QueryGenerationConfig(
        project_id="test-project",
        bq_execution_project="test-project",
        gemini_api_key="test-key",
        mcp_auth_enabled=False
    )
    
    auth = AuthenticationMiddleware(config)
    
    assert auth.enabled is False
    assert auth.mode == "api_key"  # Default mode


def test_auth_middleware_api_key_mode() -> None:
    """Test authentication middleware with API key mode."""
    config = QueryGenerationConfig(
        project_id="test-project",
        bq_execution_project="test-project",
        gemini_api_key="test-key",
        mcp_auth_enabled=True,
        mcp_auth_mode="api_key",
        mcp_api_key="secret-key-123"
    )
    
    auth = AuthenticationMiddleware(config)
    
    assert auth.enabled is True
    assert auth.mode == "api_key"


def test_auth_middleware_api_key_missing() -> None:
    """Test authentication middleware raises error when API key is missing."""
    config = QueryGenerationConfig(
        project_id="test-project",
        bq_execution_project="test-project",
        gemini_api_key="test-key",
        mcp_auth_enabled=True,
        mcp_auth_mode="api_key",
        mcp_api_key=None
    )
    
    with pytest.raises(ValueError, match="MCP_API_KEY is required"):
        AuthenticationMiddleware(config)


def test_auth_middleware_jwt_mode() -> None:
    """Test authentication middleware with JWT mode."""
    config = QueryGenerationConfig(
        project_id="test-project",
        bq_execution_project="test-project",
        gemini_api_key="test-key",
        mcp_auth_enabled=True,
        mcp_auth_mode="jwt",
        jwt_secret="jwt-secret"
    )
    
    auth = AuthenticationMiddleware(config)
    
    assert auth.enabled is True
    assert auth.mode == "jwt"


def test_auth_middleware_jwt_secret_missing() -> None:
    """Test authentication middleware raises error when JWT secret is missing."""
    config = QueryGenerationConfig(
        project_id="test-project",
        bq_execution_project="test-project",
        gemini_api_key="test-key",
        mcp_auth_enabled=True,
        mcp_auth_mode="jwt",
        jwt_secret=None
    )
    
    with pytest.raises(ValueError, match="JWT_SECRET is required"):
        AuthenticationMiddleware(config)


def test_auth_middleware_gateway_mode() -> None:
    """Test authentication middleware with gateway mode."""
    config = QueryGenerationConfig(
        project_id="test-project",
        bq_execution_project="test-project",
        gemini_api_key="test-key",
        mcp_auth_enabled=True,
        mcp_auth_mode="gateway"
    )
    
    auth = AuthenticationMiddleware(config)
    
    assert auth.enabled is True
    assert auth.mode == "gateway"


@pytest.mark.asyncio
async def test_verify_api_key_success() -> None:
    """Test API key verification with valid key."""
    config = QueryGenerationConfig(
        project_id="test-project",
        bq_execution_project="test-project",
        gemini_api_key="test-key",
        mcp_auth_enabled=True,
        mcp_auth_mode="api_key",
        mcp_api_key="secret-key-123"
    )
    
    auth = AuthenticationMiddleware(config)
    
    # Should not raise exception
    await auth.verify_api_key(x_api_key="secret-key-123")


@pytest.mark.asyncio
async def test_verify_api_key_missing() -> None:
    """Test API key verification with missing key."""
    from fastapi import HTTPException
    
    config = QueryGenerationConfig(
        project_id="test-project",
        bq_execution_project="test-project",
        gemini_api_key="test-key",
        mcp_auth_enabled=True,
        mcp_auth_mode="api_key",
        mcp_api_key="secret-key-123"
    )
    
    auth = AuthenticationMiddleware(config)
    
    with pytest.raises(HTTPException) as exc_info:
        await auth.verify_api_key(x_api_key=None)
    
    assert exc_info.value.status_code == 401
    assert "Missing X-API-Key header" in exc_info.value.detail


@pytest.mark.asyncio
async def test_verify_api_key_invalid() -> None:
    """Test API key verification with invalid key."""
    from fastapi import HTTPException
    
    config = QueryGenerationConfig(
        project_id="test-project",
        bq_execution_project="test-project",
        gemini_api_key="test-key",
        mcp_auth_enabled=True,
        mcp_auth_mode="api_key",
        mcp_api_key="secret-key-123"
    )
    
    auth = AuthenticationMiddleware(config)
    
    with pytest.raises(HTTPException) as exc_info:
        await auth.verify_api_key(x_api_key="wrong-key")
    
    assert exc_info.value.status_code == 401
    assert "Invalid API key" in exc_info.value.detail


@pytest.mark.asyncio
async def test_verify_api_key_disabled() -> None:
    """Test API key verification when auth is disabled."""
    config = QueryGenerationConfig(
        project_id="test-project",
        bq_execution_project="test-project",
        gemini_api_key="test-key",
        mcp_auth_enabled=False
    )
    
    auth = AuthenticationMiddleware(config)
    
    # Should not raise exception when auth is disabled
    await auth.verify_api_key(x_api_key=None)


@pytest.mark.asyncio
async def test_verify_jwt_missing_header() -> None:
    """Test JWT verification with missing Authorization header."""
    from fastapi import HTTPException
    
    config = QueryGenerationConfig(
        project_id="test-project",
        bq_execution_project="test-project",
        gemini_api_key="test-key",
        mcp_auth_enabled=True,
        mcp_auth_mode="jwt",
        jwt_secret="jwt-secret"
    )
    
    auth = AuthenticationMiddleware(config)
    
    with pytest.raises(HTTPException) as exc_info:
        await auth.verify_jwt(authorization=None)
    
    assert exc_info.value.status_code == 401
    assert "Missing or invalid Authorization header" in exc_info.value.detail


@pytest.mark.asyncio
async def test_verify_jwt_invalid_format() -> None:
    """Test JWT verification with invalid Authorization format."""
    from fastapi import HTTPException
    
    config = QueryGenerationConfig(
        project_id="test-project",
        bq_execution_project="test-project",
        gemini_api_key="test-key",
        mcp_auth_enabled=True,
        mcp_auth_mode="jwt",
        jwt_secret="jwt-secret"
    )
    
    auth = AuthenticationMiddleware(config)
    
    with pytest.raises(HTTPException) as exc_info:
        await auth.verify_jwt(authorization="InvalidFormat")
    
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_verify_jwt_disabled() -> None:
    """Test JWT verification when auth is disabled."""
    config = QueryGenerationConfig(
        project_id="test-project",
        bq_execution_project="test-project",
        gemini_api_key="test-key",
        mcp_auth_enabled=False
    )
    
    auth = AuthenticationMiddleware(config)
    
    # Should return None when auth is disabled
    result = await auth.verify_jwt(authorization=None)
    assert result is None


@pytest.mark.asyncio
async def test_verify_jwt_valid_token(mocker: "MockerFixture") -> None:
    """Test JWT verification with valid token."""
    # Mock PyJWT module
    mock_jwt_module = mocker.MagicMock()
    mock_jwt_module.decode.return_value = {"sub": "user123", "exp": 9999999999}
    mocker.patch.dict("sys.modules", {"jwt": mock_jwt_module})
    
    config = QueryGenerationConfig(
        project_id="test-project",
        bq_execution_project="test-project",
        gemini_api_key="test-key",
        mcp_auth_enabled=True,
        mcp_auth_mode="jwt",
        jwt_secret="jwt-secret"
    )
    
    auth = AuthenticationMiddleware(config)
    
    result = await auth.verify_jwt(authorization="Bearer valid.jwt.token")
    
    assert result is not None
    assert result["sub"] == "user123"
    mock_jwt_module.decode.assert_called_once_with(
        "valid.jwt.token",
        "jwt-secret",
        algorithms=["HS256"]
    )


@pytest.mark.asyncio
async def test_verify_jwt_expired_token(mocker: "MockerFixture") -> None:
    """Test JWT verification with expired token."""
    from fastapi import HTTPException
    
    # Mock PyJWT module to raise ExpiredSignatureError
    mock_jwt_module = mocker.MagicMock()
    ExpiredSignatureError = type("ExpiredSignatureError", (Exception,), {})
    mock_jwt_module.ExpiredSignatureError = ExpiredSignatureError
    mock_jwt_module.decode.side_effect = ExpiredSignatureError("Token expired")
    mocker.patch.dict("sys.modules", {"jwt": mock_jwt_module})
    
    config = QueryGenerationConfig(
        project_id="test-project",
        bq_execution_project="test-project",
        gemini_api_key="test-key",
        mcp_auth_enabled=True,
        mcp_auth_mode="jwt",
        jwt_secret="jwt-secret"
    )
    
    auth = AuthenticationMiddleware(config)
    
    with pytest.raises(HTTPException) as exc_info:
        await auth.verify_jwt(authorization="Bearer expired.jwt.token")
    
    assert exc_info.value.status_code == 401
    assert "Token expired" in exc_info.value.detail


@pytest.mark.asyncio
async def test_verify_jwt_invalid_token(mocker: "MockerFixture") -> None:
    """Test JWT verification with invalid token."""
    from fastapi import HTTPException
    
    # Mock PyJWT module to raise InvalidTokenError
    mock_jwt_module = mocker.MagicMock()
    InvalidTokenError = type("InvalidTokenError", (Exception,), {})
    mock_jwt_module.InvalidTokenError = InvalidTokenError
    mock_jwt_module.ExpiredSignatureError = type("ExpiredSignatureError", (Exception,), {})
    mock_jwt_module.decode.side_effect = InvalidTokenError("Invalid token")
    mocker.patch.dict("sys.modules", {"jwt": mock_jwt_module})
    
    config = QueryGenerationConfig(
        project_id="test-project",
        bq_execution_project="test-project",
        gemini_api_key="test-key",
        mcp_auth_enabled=True,
        mcp_auth_mode="jwt",
        jwt_secret="jwt-secret"
    )
    
    auth = AuthenticationMiddleware(config)
    
    with pytest.raises(HTTPException) as exc_info:
        await auth.verify_jwt(authorization="Bearer invalid.jwt.token")
    
    assert exc_info.value.status_code == 401
    assert "Invalid token" in exc_info.value.detail


@pytest.mark.asyncio
async def test_verify_gateway() -> None:
    """Test gateway authentication (trusts upstream)."""
    config = QueryGenerationConfig(
        project_id="test-project",
        bq_execution_project="test-project",
        gemini_api_key="test-key",
        mcp_auth_enabled=True,
        mcp_auth_mode="gateway"
    )
    
    auth = AuthenticationMiddleware(config)
    
    # Should not raise exception (trusts upstream gateway)
    await auth.verify_gateway()


@pytest.mark.asyncio
async def test_verify_gateway_disabled() -> None:
    """Test gateway authentication when auth is disabled."""
    config = QueryGenerationConfig(
        project_id="test-project",
        bq_execution_project="test-project",
        gemini_api_key="test-key",
        mcp_auth_enabled=False
    )
    
    auth = AuthenticationMiddleware(config)
    
    # Should not raise exception when auth is disabled
    await auth.verify_gateway()


@pytest.mark.asyncio
async def test_verify_request_api_key_mode() -> None:
    """Test verify_request with API key mode."""
    config = QueryGenerationConfig(
        project_id="test-project",
        bq_execution_project="test-project",
        gemini_api_key="test-key",
        mcp_auth_enabled=True,
        mcp_auth_mode="api_key",
        mcp_api_key="secret-key-123"
    )
    
    auth = AuthenticationMiddleware(config)
    
    # Should not raise exception
    result = await auth.verify_request(x_api_key="secret-key-123", authorization=None)
    assert result is None


@pytest.mark.asyncio
async def test_verify_request_jwt_mode(mocker: "MockerFixture") -> None:
    """Test verify_request with JWT mode."""
    # Mock PyJWT module
    mock_jwt_module = mocker.MagicMock()
    mock_jwt_module.decode.return_value = {"sub": "user123"}
    mocker.patch.dict("sys.modules", {"jwt": mock_jwt_module})
    
    config = QueryGenerationConfig(
        project_id="test-project",
        bq_execution_project="test-project",
        gemini_api_key="test-key",
        mcp_auth_enabled=True,
        mcp_auth_mode="jwt",
        jwt_secret="jwt-secret"
    )
    
    auth = AuthenticationMiddleware(config)
    
    result = await auth.verify_request(x_api_key=None, authorization="Bearer token")
    assert result is not None
    assert result["sub"] == "user123"


@pytest.mark.asyncio
async def test_verify_request_gateway_mode() -> None:
    """Test verify_request with gateway mode."""
    config = QueryGenerationConfig(
        project_id="test-project",
        bq_execution_project="test-project",
        gemini_api_key="test-key",
        mcp_auth_enabled=True,
        mcp_auth_mode="gateway"
    )
    
    auth = AuthenticationMiddleware(config)
    
    # Should not raise exception
    result = await auth.verify_request(x_api_key=None, authorization=None)
    assert result is None


@pytest.mark.asyncio
async def test_verify_request_disabled() -> None:
    """Test verify_request when auth is disabled."""
    config = QueryGenerationConfig(
        project_id="test-project",
        bq_execution_project="test-project",
        gemini_api_key="test-key",
        mcp_auth_enabled=False
    )
    
    auth = AuthenticationMiddleware(config)
    
    # Should return None when auth is disabled
    result = await auth.verify_request(x_api_key=None, authorization=None)
    assert result is None

