"""
Configuration Management for Query Generation Agent

Loads and validates environment variables for the MCP service.
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class QueryGenerationConfig(BaseModel):
    """Configuration for Query Generation Agent."""
    
    # Google Cloud Configuration
    project_id: str = Field(..., description="GCP Project ID")
    bq_execution_project: str = Field(..., description="Project for BigQuery execution")
    bq_location: str = Field(default="US", description="BigQuery location")
    
    # API Keys
    gemini_api_key: str = Field(..., description="Gemini API key")
    
    # Query Generation Configuration
    max_query_iterations: int = Field(default=10, description="Max refinement iterations")
    max_queries_per_insight: int = Field(default=5, description="Max queries to generate")
    query_timeout_seconds: int = Field(default=120, description="Query execution timeout")
    gemini_model: str = Field(default="gemini-2.5-pro-latest", description="Gemini model")
    alignment_threshold: float = Field(default=0.85, description="Min alignment score")
    
    # MCP Service Configuration
    mcp_server_name: str = Field(default="query-generation-agent", description="MCP server name")
    mcp_server_version: str = Field(default="1.0.0", description="MCP server version")
    mcp_transport: str = Field(default="stdio", description="Transport mode (stdio/http)")
    mcp_host: str = Field(default="0.0.0.0", description="HTTP host address")
    mcp_port: int = Field(default=8081, description="HTTP port")
    
    # Authentication Configuration
    mcp_auth_enabled: bool = Field(default=False, description="Enable HTTP authentication")
    mcp_auth_mode: str = Field(default="api_key", description="Auth mode (api_key/jwt/gateway)")
    mcp_api_key: Optional[str] = Field(default=None, description="API key for authentication")
    jwt_secret: Optional[str] = Field(default=None, description="JWT secret for token validation")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    
    # Optional Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    enable_validation_logging: bool = Field(default=True, description="Enable validation logs")
    max_sample_rows: int = Field(default=10, description="Max sample rows from validation")
    
    @field_validator("alignment_threshold")
    @classmethod
    def validate_alignment_threshold(cls, v: float) -> float:
        """Validate alignment threshold is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("alignment_threshold must be between 0 and 1")
        return v
    
    @field_validator("mcp_transport")
    @classmethod
    def validate_transport(cls, v: str) -> str:
        """Validate transport mode."""
        if v not in ["stdio", "http"]:
            raise ValueError("mcp_transport must be 'stdio' or 'http'")
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of: {', '.join(valid_levels)}")
        return v.upper()
    
    @field_validator("mcp_auth_mode")
    @classmethod
    def validate_auth_mode(cls, v: str) -> str:
        """Validate authentication mode."""
        valid_modes = ["api_key", "jwt", "gateway"]
        if v not in valid_modes:
            raise ValueError(f"mcp_auth_mode must be one of: {', '.join(valid_modes)}")
        return v


def load_config() -> QueryGenerationConfig:
    """
    Load configuration from environment variables.
    
    Returns:
        QueryGenerationConfig instance
        
    Raises:
        ValueError: If required environment variables are missing
    """
    # Load from .env file if present
    load_dotenv()
    
    # Required variables
    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        raise ValueError("GCP_PROJECT_ID environment variable is required")
    
    bq_execution_project = os.getenv("BQ_EXECUTION_PROJECT", project_id)
    
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    
    # Build configuration
    config = QueryGenerationConfig(
        project_id=project_id,
        bq_execution_project=bq_execution_project,
        bq_location=os.getenv("BQ_LOCATION", "US"),
        gemini_api_key=gemini_api_key,
        max_query_iterations=int(os.getenv("MAX_QUERY_ITERATIONS", "10")),
        max_queries_per_insight=int(os.getenv("MAX_QUERIES_PER_INSIGHT", "5")),
        query_timeout_seconds=int(os.getenv("QUERY_TIMEOUT_SECONDS", "120")),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-pro-latest"),
        alignment_threshold=float(os.getenv("ALIGNMENT_THRESHOLD", "0.85")),
        mcp_server_name=os.getenv("MCP_SERVER_NAME", "query-generation-agent"),
        mcp_server_version=os.getenv("MCP_SERVER_VERSION", "1.0.0"),
        mcp_transport=os.getenv("MCP_TRANSPORT", "stdio"),
        mcp_host=os.getenv("MCP_HOST", "0.0.0.0"),
        mcp_port=int(os.getenv("MCP_PORT", "8081")),
        mcp_auth_enabled=os.getenv("MCP_AUTH_ENABLED", "false").lower() == "true",
        mcp_auth_mode=os.getenv("MCP_AUTH_MODE", "api_key"),
        mcp_api_key=os.getenv("MCP_API_KEY"),
        jwt_secret=os.getenv("JWT_SECRET"),
        jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        enable_validation_logging=os.getenv("ENABLE_VALIDATION_LOGGING", "true").lower() == "true",
        max_sample_rows=int(os.getenv("MAX_SAMPLE_ROWS", "10")),
    )
    
    logger.info(f"Configuration loaded successfully")
    logger.info(f"  Project: {config.project_id}")
    logger.info(f"  Gemini Model: {config.gemini_model}")
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

