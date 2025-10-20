"""
HTTP Server for Query Generation Agent

Provides FastAPI-based HTTP server for containerized deployment.
"""

import logging
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from ..clients.bigquery_client import BigQueryClient
from ..clients.gemini_client import GeminiClient
from .config import QueryGenerationConfig, load_config
from .handlers import MCPHandlers
from .tools import get_available_tools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
config_instance: QueryGenerationConfig | None = None
handlers_instance: MCPHandlers | None = None


def create_http_app() -> FastAPI:
    """
    Create FastAPI application for HTTP transport.
    
    Returns:
        Configured FastAPI app
    """
    global config_instance, handlers_instance
    
    app = FastAPI(
        title="Query Generation Agent",
        description="MCP service that generates and validates BigQuery SQL queries from data science insights",
        version="1.0.0"
    )
    
    @app.on_event("startup")
    async def startup_event() -> None:
        """Initialize server on startup."""
        global config_instance, handlers_instance
        
        logger.info("Initializing Query Generation Agent HTTP Server...")
        
        try:
            # Load configuration
            config_instance = load_config()
            
            # Set logging level
            logging.getLogger().setLevel(config_instance.log_level)
            
            # Initialize clients
            logger.info("Initializing BigQuery client...")
            bigquery_client = BigQueryClient(
                project_id=config_instance.bq_execution_project,
                location=config_instance.bq_location,
                timeout_seconds=config_instance.query_timeout_seconds
            )
            
            logger.info("Initializing Gemini client...")
            gemini_client = GeminiClient(
                api_key=config_instance.gemini_api_key,
                model_name=config_instance.gemini_model,
                temperature=0.2,
                max_retries=3
            )
            
            # Initialize handlers
            logger.info("Initializing MCP handlers...")
            handlers_instance = MCPHandlers(
                config=config_instance,
                bigquery_client=bigquery_client,
                gemini_client=gemini_client
            )
            
            logger.info("Server initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize server: {e}", exc_info=True)
            raise
    
    @app.get("/")
    async def root() -> Dict[str, Any]:
        """
        Root endpoint with service information.
        
        Returns:
            Service metadata
        """
        if not config_instance:
            raise HTTPException(status_code=503, detail="Server not initialized")
        
        return {
            "service": "Query Generation Agent",
            "version": config_instance.mcp_server_version,
            "status": "ready",
            "description": "MCP service for generating and validating BigQuery SQL queries",
            "endpoints": {
                "health": "/health",
                "tools": "/mcp/tools",
                "call_tool": "/mcp/call-tool"
            },
            "documentation": "https://github.com/your-org/query-generation-agent"
        }
    
    @app.get("/health")
    async def health_check() -> Dict[str, str]:
        """
        Health check endpoint.
        
        Returns:
            Health status
        """
        if not handlers_instance:
            raise HTTPException(status_code=503, detail="Server not fully initialized")
        
        return {"status": "healthy"}
    
    @app.get("/mcp/tools")
    async def list_tools() -> Dict[str, Any]:
        """
        List available MCP tools.
        
        Returns:
            List of available tools with their schemas
        """
        if not handlers_instance:
            raise HTTPException(status_code=503, detail="MCP server not initialized")
        
        try:
            tools = get_available_tools()
            
            return {
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.inputSchema
                    }
                    for tool in tools
                ]
            }
        except Exception as e:
            logger.error(f"Error listing tools: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/mcp/call-tool")
    async def call_tool(request: Request) -> Dict[str, Any]:
        """
        Execute an MCP tool.
        
        Request body:
            {
                "name": "tool_name",
                "arguments": {...}
            }
        
        Returns:
            Tool execution results
        """
        if not handlers_instance:
            raise HTTPException(status_code=503, detail="MCP server not initialized")
        
        try:
            body = await request.json()
            
            tool_name = body.get("name")
            arguments = body.get("arguments", {})
            
            if not tool_name:
                raise HTTPException(status_code=400, detail="Missing 'name' in request")
            
            logger.info(f"Tool called via HTTP: {tool_name}")
            
            # Call tool handler
            from .tools import GENERATE_QUERIES_TOOL
            
            if tool_name == GENERATE_QUERIES_TOOL:
                result = await handlers_instance.handle_generate_queries(arguments)
            else:
                raise HTTPException(status_code=404, detail=f"Unknown tool: {tool_name}")
            
            # Extract text from TextContent
            if result and len(result) > 0:
                import json
                response_text = result[0].text
                response_data = json.loads(response_text)
                return response_data
            else:
                return {"error": "No response from tool"}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error calling tool: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """
        Global exception handler.
        
        Args:
            request: Request that caused the exception
            exc: Exception that was raised
            
        Returns:
            JSON error response
        """
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc)
            }
        )
    
    return app


# Create app instance
app = create_http_app()


if __name__ == "__main__":
    import uvicorn
    
    # Load config for port
    config = load_config()
    
    logger.info(f"Starting HTTP server on {config.mcp_host}:{config.mcp_port}")
    
    uvicorn.run(
        app,
        host=config.mcp_host,
        port=config.mcp_port,
        log_level=config.log_level.lower()
    )

