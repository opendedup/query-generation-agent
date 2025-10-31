"""
Entry point for running the MCP server via python -m query_generation_agent.mcp

Routes to either HTTP or stdio transport based on MCP_TRANSPORT environment variable.
"""

import logging
from .config import load_config


def run_server() -> None:
    """
    Main entry point for MCP server.
    
    Routes to either HTTP or stdio transport based on configuration.
    """
    config = load_config()
    
    # Apply logging level from configuration
    logging.getLogger().setLevel(config.log_level)
    
    # Suppress verbose SQLFluff logging
    logging.getLogger('sqlfluff').setLevel(logging.WARNING)
    
    if config.mcp_transport.lower() == "http":
        # Run HTTP server
        import uvicorn
        from .http_server import app
        
        print(f"Starting Query Generation Agent MCP Server (HTTP mode) on {config.mcp_host}:{config.mcp_port}...")
        
        uvicorn.run(
            app,
            host=config.mcp_host,
            port=config.mcp_port,
            log_level=config.log_level.lower()
        )
    else:
        # Run stdio server
        import asyncio
        from .server import main
        
        print("Starting Query Generation Agent MCP Server (stdio mode)...")
        asyncio.run(main())


if __name__ == "__main__":
    run_server()

