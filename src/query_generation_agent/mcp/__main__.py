"""
Entry point for running the MCP server via python -m query_generation_agent.mcp
"""

import asyncio

from .server import main

if __name__ == "__main__":
    asyncio.run(main())

