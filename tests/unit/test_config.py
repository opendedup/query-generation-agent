"""
Unit tests for configuration management.
"""

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pytest_mock.plugin import MockerFixture

from query_generation_agent.mcp.config import QueryGenerationConfig, load_config


def test_config_defaults() -> None:
    """Test configuration with default values."""
    config = QueryGenerationConfig(
        project_id="test-project",
        bq_execution_project="test-project",
        gemini_api_key="test-key"
    )
    
    assert config.project_id == "test-project"
    assert config.bq_execution_project == "test-project"
    assert config.gemini_api_key == "test-key"
    assert config.bq_location == "US"
    assert config.max_query_iterations == 10
    assert config.max_queries_per_insight == 5
    assert config.alignment_threshold == 0.85


def test_config_validation_alignment_threshold() -> None:
    """Test alignment threshold validation."""
    # Valid threshold
    config = QueryGenerationConfig(
        project_id="test-project",
        bq_execution_project="test-project",
        gemini_api_key="test-key",
        alignment_threshold=0.9
    )
    assert config.alignment_threshold == 0.9
    
    # Invalid threshold (> 1)
    with pytest.raises(ValueError, match="alignment_threshold must be between 0 and 1"):
        QueryGenerationConfig(
            project_id="test-project",
            bq_execution_project="test-project",
            gemini_api_key="test-key",
            alignment_threshold=1.5
        )
    
    # Invalid threshold (< 0)
    with pytest.raises(ValueError, match="alignment_threshold must be between 0 and 1"):
        QueryGenerationConfig(
            project_id="test-project",
            bq_execution_project="test-project",
            gemini_api_key="test-key",
            alignment_threshold=-0.1
        )


def test_config_validation_transport() -> None:
    """Test transport mode validation."""
    # Valid transport
    config = QueryGenerationConfig(
        project_id="test-project",
        bq_execution_project="test-project",
        gemini_api_key="test-key",
        mcp_transport="http"
    )
    assert config.mcp_transport == "http"
    
    # Invalid transport
    with pytest.raises(ValueError, match="mcp_transport must be 'stdio' or 'http'"):
        QueryGenerationConfig(
            project_id="test-project",
            bq_execution_project="test-project",
            gemini_api_key="test-key",
            mcp_transport="invalid"
        )


def test_config_validation_log_level() -> None:
    """Test log level validation."""
    # Valid log level
    config = QueryGenerationConfig(
        project_id="test-project",
        bq_execution_project="test-project",
        gemini_api_key="test-key",
        log_level="DEBUG"
    )
    assert config.log_level == "DEBUG"
    
    # Invalid log level
    with pytest.raises(ValueError, match="log_level must be one of"):
        QueryGenerationConfig(
            project_id="test-project",
            bq_execution_project="test-project",
            gemini_api_key="test-key",
            log_level="INVALID"
        )


def test_load_config_missing_required(mocker: "MockerFixture") -> None:
    """Test load_config with missing required variables."""
    # Mock environment variables and disable .env file loading
    mocker.patch.dict(os.environ, {}, clear=True)
    mocker.patch("query_generation_agent.mcp.config.load_dotenv")
    
    # Should raise ValueError for missing GCP_PROJECT_ID
    with pytest.raises(ValueError, match="GCP_PROJECT_ID environment variable is required"):
        load_config()


def test_load_config_success(mocker: "MockerFixture") -> None:
    """Test successful config loading from environment."""
    # Mock environment variables
    mocker.patch.dict(os.environ, {
        "GCP_PROJECT_ID": "test-project",
        "BQ_EXECUTION_PROJECT": "exec-project",
        "GEMINI_API_KEY": "test-api-key",
        "BQ_LOCATION": "US",
        "MAX_QUERY_ITERATIONS": "15",
        "MAX_QUERIES_PER_INSIGHT": "5",
        "ALIGNMENT_THRESHOLD": "0.9",
        "MCP_TRANSPORT": "http",
        "MCP_PORT": "9000",
        "LOG_LEVEL": "DEBUG"
    }, clear=True)
    
    config = load_config()
    
    assert config.project_id == "test-project"
    assert config.bq_execution_project == "exec-project"
    assert config.gemini_api_key == "test-api-key"
    assert config.bq_location == "US"
    assert config.max_query_iterations == 15
    assert config.alignment_threshold == 0.9
    assert config.mcp_transport == "http"
    assert config.mcp_port == 9000
    assert config.log_level == "DEBUG"


def test_config_auth_defaults() -> None:
    """Test authentication configuration defaults."""
    config = QueryGenerationConfig(
        project_id="test-project",
        bq_execution_project="test-project",
        gemini_api_key="test-key"
    )
    
    assert config.mcp_auth_enabled is False
    assert config.mcp_auth_mode == "api_key"
    assert config.mcp_api_key is None
    assert config.jwt_secret is None
    assert config.jwt_algorithm == "HS256"


def test_config_auth_validation_mode() -> None:
    """Test authentication mode validation."""
    # Valid auth modes
    for mode in ["api_key", "jwt", "gateway"]:
        config = QueryGenerationConfig(
            project_id="test-project",
            bq_execution_project="test-project",
            gemini_api_key="test-key",
            mcp_auth_mode=mode
        )
        assert config.mcp_auth_mode == mode
    
    # Invalid auth mode
    with pytest.raises(ValueError, match="mcp_auth_mode must be one of"):
        QueryGenerationConfig(
            project_id="test-project",
            bq_execution_project="test-project",
            gemini_api_key="test-key",
            mcp_auth_mode="invalid"
        )


def test_config_auth_api_key() -> None:
    """Test API key authentication configuration."""
    config = QueryGenerationConfig(
        project_id="test-project",
        bq_execution_project="test-project",
        gemini_api_key="test-key",
        mcp_auth_enabled=True,
        mcp_auth_mode="api_key",
        mcp_api_key="secret-api-key-123"
    )
    
    assert config.mcp_auth_enabled is True
    assert config.mcp_auth_mode == "api_key"
    assert config.mcp_api_key == "secret-api-key-123"


def test_config_auth_jwt() -> None:
    """Test JWT authentication configuration."""
    config = QueryGenerationConfig(
        project_id="test-project",
        bq_execution_project="test-project",
        gemini_api_key="test-key",
        mcp_auth_enabled=True,
        mcp_auth_mode="jwt",
        jwt_secret="jwt-secret-key",
        jwt_algorithm="HS256"
    )
    
    assert config.mcp_auth_enabled is True
    assert config.mcp_auth_mode == "jwt"
    assert config.jwt_secret == "jwt-secret-key"
    assert config.jwt_algorithm == "HS256"


def test_load_config_auth_disabled(mocker: "MockerFixture") -> None:
    """Test load_config with authentication disabled."""
    mocker.patch.dict(os.environ, {
        "GCP_PROJECT_ID": "test-project",
        "GEMINI_API_KEY": "test-key",
        "MCP_AUTH_ENABLED": "false"
    }, clear=True)
    
    config = load_config()
    
    assert config.mcp_auth_enabled is False


def test_load_config_auth_api_key(mocker: "MockerFixture") -> None:
    """Test load_config with API key authentication."""
    mocker.patch.dict(os.environ, {
        "GCP_PROJECT_ID": "test-project",
        "GEMINI_API_KEY": "test-key",
        "MCP_AUTH_ENABLED": "true",
        "MCP_AUTH_MODE": "api_key",
        "MCP_API_KEY": "test-secret-key"
    }, clear=True)
    
    config = load_config()
    
    assert config.mcp_auth_enabled is True
    assert config.mcp_auth_mode == "api_key"
    assert config.mcp_api_key == "test-secret-key"


def test_load_config_auth_jwt(mocker: "MockerFixture") -> None:
    """Test load_config with JWT authentication."""
    mocker.patch.dict(os.environ, {
        "GCP_PROJECT_ID": "test-project",
        "GEMINI_API_KEY": "test-key",
        "MCP_AUTH_ENABLED": "true",
        "MCP_AUTH_MODE": "jwt",
        "JWT_SECRET": "jwt-secret",
        "JWT_ALGORITHM": "HS512"
    }, clear=True)
    
    config = load_config()
    
    assert config.mcp_auth_enabled is True
    assert config.mcp_auth_mode == "jwt"
    assert config.jwt_secret == "jwt-secret"
    assert config.jwt_algorithm == "HS512"


def test_load_config_auth_gateway(mocker: "MockerFixture") -> None:
    """Test load_config with gateway authentication."""
    mocker.patch.dict(os.environ, {
        "GCP_PROJECT_ID": "test-project",
        "GEMINI_API_KEY": "test-key",
        "MCP_AUTH_ENABLED": "true",
        "MCP_AUTH_MODE": "gateway"
    }, clear=True)
    
    config = load_config()
    
    assert config.mcp_auth_enabled is True
    assert config.mcp_auth_mode == "gateway"

