"""
MCP Server for Query Generation Agent

Provides stdio-based MCP server for local development and subprocess communication.
"""

import asyncio
import logging
from typing import Any, Dict, Sequence

from mcp import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from ..clients.bigquery_client import BigQueryClient
from ..clients.gemini_client import GeminiClient
from .config import QueryGenerationConfig, load_config
from .handlers import MCPHandlers
from .tools import GENERATE_QUERIES_TOOL, GENERATE_VIEWS_TOOL, get_available_tools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_mcp_server(config: QueryGenerationConfig | None = None) -> Server:
    """
    Create and configure MCP server.
    
    Args:
        config: Configuration (loads from env if not provided)
        
    Returns:
        Configured MCP Server instance
    """
    # Load config if not provided
    if config is None:
        config = load_config()
    
    # Set logging level
    logging.getLogger().setLevel(config.log_level)
    
    # Initialize clients
    logger.info("Initializing BigQuery client...")
    bigquery_client = BigQueryClient(
        project_id=config.bq_execution_project,
        location=config.bq_location,
        timeout_seconds=config.query_timeout_seconds
    )
    
    logger.info("Initializing Gemini client...")
    gemini_client = GeminiClient(
        api_key=config.gemini_api_key,
        model_name=config.gemini_model,
        temperature=0.2,  # Lower temperature for more deterministic SQL generation
        max_retries=3
    )
    
    # Initialize handlers
    logger.info("Initializing MCP handlers...")
    handlers = MCPHandlers(
        config=config,
        bigquery_client=bigquery_client,
        gemini_client=gemini_client
    )
    
    # Create MCP server
    server = Server(config.mcp_server_name)
    
    logger.info(f"MCP Server '{config.mcp_server_name}' v{config.mcp_server_version} initialized")
    
    # Register list_tools handler
    @server.list_tools()
    async def handle_list_tools() -> list[Tool]:
        """
        List available MCP tools.
        
        Returns:
            List of available tools
        """
        logger.debug("Listing available tools")
        return get_available_tools()
    
    # Register call_tool handler
    @server.call_tool()
    async def handle_call_tool(
        name: str,
        arguments: Dict[str, Any] | None,
    ) -> Sequence[TextContent]:
        """
        Handle tool invocation.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            Sequence of TextContent responses
        """
        logger.info(f"Tool called: {name}")
        
        # Default to empty dict if no arguments
        if arguments is None:
            arguments = {}
        
        try:
            # Route to appropriate handler
            if name == GENERATE_QUERIES_TOOL:
                return await handlers.handle_generate_queries(arguments)
            
            elif name == GENERATE_VIEWS_TOOL:
                return await handlers.handle_generate_views(arguments)
            
            else:
                error_msg = f"Unknown tool: {name}"
                logger.error(error_msg)
                return [TextContent(type="text", text=f"Error: {error_msg}")]
        
        except Exception as e:
            logger.error(f"Error handling tool {name}: {e}", exc_info=True)
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    return server


async def main() -> None:
    """
    Main entry point for MCP server (stdio mode).
    
    Runs the server using stdio transport for local development
    and subprocess communication.
    """
    logger.info("Starting Query Generation Agent MCP Server (stdio mode)...")
    
    try:
        # Load configuration
        config = load_config()
        
        # Create server
        server = create_mcp_server(config)
        
        # Run server with stdio transport
        async with stdio_server() as (read_stream, write_stream):
            logger.info("Server ready and listening on stdio")
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
    
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())

