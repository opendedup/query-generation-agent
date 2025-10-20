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
    assert config.gemini_model == "gemini-2.5-pro-latest"


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
    # Mock environment variables
    mocker.patch.dict(os.environ, {}, clear=True)
    
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
        "BQ_LOCATION": "EU",
        "MAX_QUERY_ITERATIONS": "15",
        "MAX_QUERIES_PER_INSIGHT": "5",
        "ALIGNMENT_THRESHOLD": "0.9",
        "GEMINI_MODEL": "gemini-2.5-pro-latest",
        "MCP_TRANSPORT": "http",
        "MCP_PORT": "9000",
        "LOG_LEVEL": "DEBUG"
    }, clear=True)
    
    config = load_config()
    
    assert config.project_id == "test-project"
    assert config.bq_execution_project == "exec-project"
    assert config.gemini_api_key == "test-api-key"
    assert config.bq_location == "EU"
    assert config.max_query_iterations == 15
    assert config.alignment_threshold == 0.9
    assert config.mcp_transport == "http"
    assert config.mcp_port == 9000
    assert config.log_level == "DEBUG"

