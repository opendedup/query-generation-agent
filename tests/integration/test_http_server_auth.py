"""
Integration tests for HTTP server authentication.
"""

import os
import sys
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def mock_env_base(mocker: "MockerFixture") -> None:
    """Mock base environment variables for all tests."""
    mocker.patch.dict(os.environ, {
        "GCP_PROJECT_ID": "test-project",
        "GEMINI_API_KEY": "test-key",
        "MCP_TRANSPORT": "http",
        "MCP_HOST": "127.0.0.1",
        "MCP_PORT": "8081"
    }, clear=True)


@pytest.fixture
def mock_clients(mocker: "MockerFixture") -> None:
    """Mock BigQuery and Gemini clients to avoid real API calls."""
    mocker.patch("query_generation_agent.mcp.http_server.BigQueryClient")
    mocker.patch("query_generation_agent.mcp.http_server.GeminiClient")
    mocker.patch("query_generation_agent.mcp.http_server.MCPHandlers")


def test_http_server_auth_disabled(mock_env_base: None, mock_clients: None, mocker: "MockerFixture") -> None:
    """Test HTTP server with authentication disabled."""
    mocker.patch.dict(os.environ, {"MCP_AUTH_ENABLED": "false"}, clear=False)
    
    # Import after env is mocked
    from query_generation_agent.mcp.http_server import create_http_app
    
    app = create_http_app()
    client = TestClient(app)
    
    # Trigger startup event
    with client:
        # Health check should work
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        
        # Tools endpoint should work without authentication
        response = client.get("/mcp/tools")
        assert response.status_code == 200
        assert "tools" in response.json()


def test_http_server_api_key_auth_missing(mock_env_base: None, mock_clients: None, mocker: "MockerFixture") -> None:
    """Test HTTP server with API key auth - missing key."""
    mocker.patch.dict(os.environ, {
        "MCP_AUTH_ENABLED": "true",
        "MCP_AUTH_MODE": "api_key",
        "MCP_API_KEY": "test-secret-key"
    }, clear=False)
    
    from query_generation_agent.mcp.http_server import create_http_app
    
    app = create_http_app()
    client = TestClient(app)
    
    with client:
        # Tools endpoint should reject without API key
        response = client.get("/mcp/tools")
        assert response.status_code == 401
        assert "Missing X-API-Key header" in response.json()["detail"]


def test_http_server_api_key_auth_invalid(mock_env_base: None, mock_clients: None, mocker: "MockerFixture") -> None:
    """Test HTTP server with API key auth - invalid key."""
    mocker.patch.dict(os.environ, {
        "MCP_AUTH_ENABLED": "true",
        "MCP_AUTH_MODE": "api_key",
        "MCP_API_KEY": "test-secret-key"
    }, clear=False)
    
    from query_generation_agent.mcp.http_server import create_http_app
    
    app = create_http_app()
    client = TestClient(app)
    
    with client:
        # Tools endpoint should reject with wrong API key
        response = client.get("/mcp/tools", headers={"X-API-Key": "wrong-key"})
        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]


def test_http_server_api_key_auth_valid(mock_env_base: None, mock_clients: None, mocker: "MockerFixture") -> None:
    """Test HTTP server with API key auth - valid key."""
    mocker.patch.dict(os.environ, {
        "MCP_AUTH_ENABLED": "true",
        "MCP_AUTH_MODE": "api_key",
        "MCP_API_KEY": "test-secret-key"
    }, clear=False)
    
    from query_generation_agent.mcp.http_server import create_http_app
    
    app = create_http_app()
    client = TestClient(app)
    
    with client:
        # Tools endpoint should work with correct API key
        response = client.get("/mcp/tools", headers={"X-API-Key": "test-secret-key"})
        assert response.status_code == 200
        assert "tools" in response.json()


def test_http_server_jwt_auth_missing(mock_env_base: None, mock_clients: None, mocker: "MockerFixture") -> None:
    """Test HTTP server with JWT auth - missing token."""
    mocker.patch.dict(os.environ, {
        "MCP_AUTH_ENABLED": "true",
        "MCP_AUTH_MODE": "jwt",
        "JWT_SECRET": "jwt-secret"
    }, clear=False)
    
    from query_generation_agent.mcp.http_server import create_http_app
    
    app = create_http_app()
    client = TestClient(app)
    
    with client:
        # Tools endpoint should reject without token
        response = client.get("/mcp/tools")
        assert response.status_code == 401
        assert "Missing or invalid Authorization header" in response.json()["detail"]


def test_http_server_jwt_auth_invalid_format(mock_env_base: None, mock_clients: None, mocker: "MockerFixture") -> None:
    """Test HTTP server with JWT auth - invalid format."""
    mocker.patch.dict(os.environ, {
        "MCP_AUTH_ENABLED": "true",
        "MCP_AUTH_MODE": "jwt",
        "JWT_SECRET": "jwt-secret"
    }, clear=False)
    
    from query_generation_agent.mcp.http_server import create_http_app
    
    app = create_http_app()
    client = TestClient(app)
    
    with client:
        # Tools endpoint should reject with invalid format
        response = client.get("/mcp/tools", headers={"Authorization": "InvalidFormat"})
        assert response.status_code == 401


def test_http_server_jwt_auth_valid(mock_env_base: None, mock_clients: None, mocker: "MockerFixture") -> None:
    """Test HTTP server with JWT auth - valid token."""
    mocker.patch.dict(os.environ, {
        "MCP_AUTH_ENABLED": "true",
        "MCP_AUTH_MODE": "jwt",
        "JWT_SECRET": "jwt-secret"
    }, clear=False)
    
    # Mock PyJWT module
    mock_jwt_module = mocker.MagicMock()
    mock_jwt_module.decode.return_value = {"sub": "user123"}
    mocker.patch.dict("sys.modules", {"jwt": mock_jwt_module})
    
    from query_generation_agent.mcp.http_server import create_http_app
    
    app = create_http_app()
    client = TestClient(app)
    
    with client:
        # Tools endpoint should work with valid token
        response = client.get("/mcp/tools", headers={"Authorization": "Bearer valid.token"})
        assert response.status_code == 200
        assert "tools" in response.json()


def test_http_server_gateway_auth(mock_env_base: None, mock_clients: None, mocker: "MockerFixture") -> None:
    """Test HTTP server with gateway auth."""
    mocker.patch.dict(os.environ, {
        "MCP_AUTH_ENABLED": "true",
        "MCP_AUTH_MODE": "gateway"
    }, clear=False)
    
    from query_generation_agent.mcp.http_server import create_http_app
    
    app = create_http_app()
    client = TestClient(app)
    
    with client:
        # Tools endpoint should work (trusts upstream gateway)
        response = client.get("/mcp/tools")
        assert response.status_code == 200
        assert "tools" in response.json()


def test_http_server_health_always_public(mock_env_base: None, mock_clients: None, mocker: "MockerFixture") -> None:
    """Test that health endpoint is always public regardless of auth settings."""
    mocker.patch.dict(os.environ, {
        "MCP_AUTH_ENABLED": "true",
        "MCP_AUTH_MODE": "api_key",
        "MCP_API_KEY": "test-secret-key"
    }, clear=False)
    
    from query_generation_agent.mcp.http_server import create_http_app
    
    app = create_http_app()
    client = TestClient(app)
    
    with client:
        # Health endpoint should work without authentication
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


def test_http_server_call_tool_protected(mock_env_base: None, mock_clients: None, mocker: "MockerFixture") -> None:
    """Test that call-tool endpoint is protected by authentication."""
    mocker.patch.dict(os.environ, {
        "MCP_AUTH_ENABLED": "true",
        "MCP_AUTH_MODE": "api_key",
        "MCP_API_KEY": "test-secret-key"
    }, clear=False)
    
    from query_generation_agent.mcp.http_server import create_http_app
    
    app = create_http_app()
    client = TestClient(app)
    
    with client:
        # Call-tool endpoint should reject without API key
        response = client.post("/mcp/call-tool", json={"name": "generate_queries", "arguments": {}})
        assert response.status_code == 401
        
        # Call-tool endpoint should work with correct API key
        response = client.post(
            "/mcp/call-tool",
            json={"name": "generate_queries", "arguments": {}},
            headers={"X-API-Key": "test-secret-key"}
        )
        # Note: Will fail with 400 or 503 due to missing arguments/handlers, but auth passes
        assert response.status_code != 401


def test_http_server_root_endpoint_public(mock_env_base: None, mock_clients: None, mocker: "MockerFixture") -> None:
    """Test that root endpoint is public for service info."""
    mocker.patch.dict(os.environ, {
        "MCP_AUTH_ENABLED": "true",
        "MCP_AUTH_MODE": "api_key",
        "MCP_API_KEY": "test-secret-key"
    }, clear=False)
    
    from query_generation_agent.mcp.http_server import create_http_app
    
    app = create_http_app()
    client = TestClient(app)
    
    with client:
        # Root endpoint should work without authentication
        response = client.get("/")
        assert response.status_code == 200
        assert "service" in response.json()
        assert response.json()["service"] == "Query Generation Agent"

