"""
Configuration Management for Query Generation Agent

Loads and validates environment variables for the MCP service.
"""

import logging
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class QueryGenerationConfig(BaseSettings):
    """Configuration for Query Generation Agent."""

    # Pydantic settings
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Google Cloud Configuration
    project_id: str = Field(..., description="GCP Project ID", alias="GCP_PROJECT_ID")
    bq_execution_project: str = Field(..., description="Project for BigQuery execution", alias="BQ_EXECUTION_PROJECT")
    bq_location: str = Field(default="US", description="BigQuery location", alias="BQ_LOCATION")
    
    # API Keys
    gemini_api_key: str = Field(..., description="Gemini API key", alias="GEMINI_API_KEY")
    
    # Data Discovery Agent Configuration
    discovery_agent_url: str = Field(
        default="http://localhost:8080",
        description="Data Discovery Agent HTTP endpoint",
        alias="DISCOVERY_AGENT_URL",
    )
    
    # Query Generation Configuration
    max_query_iterations: int = Field(
        default=5,
        description="Max refinement iterations",
        alias="MAX_ITERATIONS",
    )
    max_queries_per_insight: int = Field(
        default=3,
        description="Max queries to generate",
        alias="MAX_QUERIES_PER_INSIGHT",
    )
    query_timeout_seconds: int = Field(
        default=120,
        description="Query execution timeout",
        alias="QUERY_TIMEOUT_SECONDS",
    )
    alignment_threshold: float = Field(
        default=0.85,
        description="Min alignment score",
        alias="ALIGNMENT_THRESHOLD",
    )
    query_naming_strategy: str = Field(
        default="rule_based",
        description="Query naming strategy (rule_based/llm/hybrid)",
        alias="QUERY_NAMING_STRATEGY",
    )
    
    # MCP Service Configuration
    mcp_server_name: str = Field(
        default="query-generation-agent",
        description="MCP server name",
        alias="MCP_SERVER_NAME",
    )
    mcp_server_version: str = Field(
        default="1.0.0",
        description="MCP server version",
        alias="MCP_SERVER_VERSION",
    )
    mcp_transport: str = Field(
        default="stdio",
        description="Transport mode (stdio/http)",
        alias="MCP_TRANSPORT",
    )
    mcp_host: str = Field(
        default="0.0.0.0",
        description="HTTP host address",
        alias="MCP_HOST",
    )
    mcp_port: int = Field(
        default=8081,
        description="HTTP port",
        alias="MCP_PORT",
    )
    
    # Authentication Configuration
    mcp_auth_enabled: bool = Field(
        default=False,
        description="Enable HTTP authentication",
        alias="MCP_AUTH_ENABLED",
    )
    mcp_auth_mode: str = Field(
        default="api_key",
        description="Auth mode (api_key/jwt/gateway)",
        alias="MCP_AUTH_MODE",
    )
    mcp_api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication",
        alias="MCP_API_KEY",
    )
    jwt_secret: Optional[str] = Field(
        default=None,
        description="JWT secret for token validation",
        alias="JWT_SECRET",
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT algorithm",
        alias="JWT_ALGORITHM",
    )
    
    # Optional Configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level",
        alias="LOG_LEVEL",
    )
    enable_validation_logging: bool = Field(
        default=True,
        description="Enable validation logs",
        alias="ENABLE_VALIDATION_LOGGING",
    )
    max_sample_rows: int = Field(
        default=10,
        description="Max sample rows from validation",
        alias="MAX_SAMPLE_ROWS",
    )


def load_config() -> QueryGenerationConfig:
    """
    Load configuration from environment variables.
    
    Returns:
        QueryGenerationConfig instance
        
    Raises:
        ValueError: If required environment variables are missing
    """
    try:
        config = QueryGenerationConfig()
        
        # If bq_execution_project is not set, use project_id
        if not config.bq_execution_project:
            config.bq_execution_project = config.project_id
            
        logger.info("Configuration loaded successfully")
        logger.info(f"  Project: {config.project_id}")
        logger.info(f"  Max Iterations: {config.max_query_iterations}")
        logger.info(f"  Alignment Threshold: {config.alignment_threshold}")
        logger.info(f"  Transport: {config.mcp_transport}")
        
        # Log authentication status
        if config.mcp_transport == "http":
            if config.mcp_auth_enabled:
                logger.info(f"  HTTP Authentication: Enabled (mode: {config.mcp_auth_mode})")
            else:
                logger.warning("  HTTP Authentication: DISABLED - Only use for development on localhost!")
        
        return config
    
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}", exc_info=True)
        raise

