"""
HTTP Server for Query Generation Agent

Provides FastAPI-based HTTP server for containerized deployment with SSE support.
"""

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..clients.bigquery_client import BigQueryClient
from ..clients.gemini_client import GeminiClient
from .auth import AuthenticationMiddleware
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
auth_middleware: AuthenticationMiddleware | None = None
task_manager_instance: Any | None = None


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
        global config_instance, handlers_instance, auth_middleware, task_manager_instance
        
        logger.info("Initializing Query Generation Agent HTTP Server...")
        
        try:
            # Load configuration
            config_instance = load_config()
            
            # Set logging level
            logging.getLogger().setLevel(config_instance.log_level)
            
            # Initialize authentication middleware
            logger.info("Initializing authentication middleware...")
            auth_middleware = AuthenticationMiddleware(config_instance)
            
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
            
            # Initialize task manager for async operations
            logger.info("Initializing task manager...")
            from .task_manager import TaskManager
            task_manager_instance = TaskManager()
            
            # Start background task cleanup
            async def cleanup_loop():
                """Periodically cleanup old tasks."""
                while True:
                    await asyncio.sleep(300)  # Every 5 minutes
                    task_manager_instance.cleanup_old_tasks(max_age_seconds=3600)
            
            asyncio.create_task(cleanup_loop())
            logger.info("Task manager initialized with cleanup loop")
            
            logger.info("Server initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize server: {e}", exc_info=True)
            raise
    
    async def event_stream() -> AsyncGenerator[str, None]:
        """
        Generate Server-Sent Events stream.
        
        Yields formatted SSE messages for notifications and updates.
        """
        # Send initial connection message
        yield f"event: message\ndata: {json.dumps({'type': 'connected', 'service': 'query-generation-agent'})}\n\n"
        
        # Keep connection alive with periodic heartbeat
        try:
            while True:
                await asyncio.sleep(30)
                yield f"event: heartbeat\ndata: {json.dumps({'timestamp': asyncio.get_event_loop().time()})}\n\n"
        except asyncio.CancelledError:
            logger.info("SSE stream closed")
    
    @app.get("/", response_model=None)
    async def root(request: Request) -> Dict[str, Any] | StreamingResponse:
        """
        Root endpoint with service information or SSE endpoint.
        
        Args:
            request: HTTP request
        
        Returns:
            Service metadata (JSON) or SSE stream
        """
        if not config_instance:
            raise HTTPException(status_code=503, detail="Server not initialized")
        
        # Check if client wants SSE stream
        accept_header = request.headers.get("accept", "")
        if "text/event-stream" in accept_header:
            logger.info("Client requested SSE stream")
            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        
        # Return service information
        return {
            "service": "Query Generation Agent",
            "version": config_instance.mcp_server_version,
            "status": "ready",
            "transport": "http",
            "description": "MCP service for generating and validating BigQuery SQL queries",
            "endpoints": {
                "health": "/health",
                "tools": "/mcp/tools",
                "call_tool": "/mcp/call-tool",
                "sse": "/ (with Accept: text/event-stream)"
            },
            "documentation": "https://github.com/your-org/query-generation-agent"
        }
    
    @app.get("/health")
    async def health_check() -> Dict[str, str]:
        """
        Health check endpoint.
        
        Returns:
            Health status with service name and transport mode
        """
        if not handlers_instance:
            raise HTTPException(status_code=503, detail="Server not fully initialized")
        
        return {
            "status": "healthy",
            "service": "query-generation-agent",
            "transport": "http"
        }
    
    async def verify_auth(
        x_api_key: Optional[str] = Header(None),
        authorization: Optional[str] = Header(None)
    ) -> Optional[dict]:
        """
        Dependency for authentication verification.
        
        Args:
            x_api_key: API key from X-API-Key header
            authorization: Authorization header for JWT
            
        Returns:
            Authentication payload or None
        """
        if not auth_middleware:
            return None
        return await auth_middleware.verify_request(x_api_key, authorization)
    
    @app.get("/mcp/tools", dependencies=[Depends(verify_auth)])
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
    
    @app.post("/mcp/call-tool", dependencies=[Depends(verify_auth)])
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
            from .tools import GENERATE_QUERIES_TOOL, GENERATE_VIEWS_TOOL
            
            if tool_name == GENERATE_QUERIES_TOOL:
                result = await handlers_instance.handle_generate_queries(arguments)
            elif tool_name == GENERATE_VIEWS_TOOL:
                result = await handlers_instance.handle_generate_views(arguments)
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
    
    @app.post("/mcp/call-tool-async", dependencies=[Depends(verify_auth)])
    async def call_tool_async(request: Request) -> JSONResponse:
        """
        Execute an MCP tool asynchronously.
        
        Returns task ID immediately for long-running operations.
        Client polls status endpoint to check progress.
        
        Request body:
            {
                "name": "tool_name",
                "arguments": {...}
            }
        
        Returns:
            202 Accepted with task_id and status URLs
        """
        if not handlers_instance or not task_manager_instance:
            raise HTTPException(status_code=503, detail="MCP server not initialized")
        
        try:
            body = await request.json()
            
            tool_name = body.get("name")
            arguments = body.get("arguments", {})
            
            if not tool_name:
                raise HTTPException(status_code=400, detail="Missing 'name' in request")
            
            # Generate task ID
            task_id = str(uuid.uuid4())
            
            logger.info(f"Async tool called via HTTP: {tool_name}, task_id: {task_id}")
            
            # Create task
            task_manager_instance.create_task(task_id)
            
            # Start background execution
            from .tools import GENERATE_QUERIES_TOOL, GENERATE_VIEWS_TOOL
            
            if tool_name == GENERATE_QUERIES_TOOL:
                asyncio.create_task(
                    handlers_instance.handle_generate_queries_async(
                        task_id, arguments, task_manager_instance
                    )
                )
            elif tool_name == GENERATE_VIEWS_TOOL:
                # Generate views synchronously (usually fast) then update task
                async def generate_views_task():
                    try:
                        from .task_manager import TaskStatus
                        task_manager_instance.update_task_status(task_id, TaskStatus.RUNNING)
                        result = await handlers_instance.handle_generate_views(arguments)
                        task_manager_instance.update_task_status(
                            task_id, TaskStatus.COMPLETED, result=result
                        )
                    except Exception as e:
                        task_manager_instance.update_task_status(
                            task_id, TaskStatus.FAILED, error=str(e)
                        )
                asyncio.create_task(generate_views_task())
            else:
                raise HTTPException(status_code=404, detail=f"Unknown tool: {tool_name}")
            
            return JSONResponse(
                status_code=202,
                content={
                    "task_id": task_id,
                    "status": "pending",
                    "status_url": f"/mcp/tasks/{task_id}",
                    "result_url": f"/mcp/tasks/{task_id}/result",
                    "message": "Task started. Poll status_url to check progress."
                }
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error starting async tool: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/mcp/tasks/{task_id}", dependencies=[Depends(verify_auth)])
    async def get_task_status(task_id: str) -> Dict[str, Any]:
        """
        Get status of an async task.
        
        Args:
            task_id: Task identifier
        
        Returns:
            Task status and metadata
        """
        if not task_manager_instance:
            raise HTTPException(status_code=503, detail="Task manager not initialized")
        
        task = task_manager_instance.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        
        from .task_manager import TaskStatus
        
        response = {
            "task_id": task.id,
            "status": task.status,
            "created_at": task.created_at.isoformat(),
        }
        
        if task.status == TaskStatus.COMPLETED:
            response["result_url"] = f"/mcp/tasks/{task_id}/result"
            if task.completed_at:
                response["completed_at"] = task.completed_at.isoformat()
        elif task.status == TaskStatus.FAILED:
            response["error"] = task.error
            if task.completed_at:
                response["completed_at"] = task.completed_at.isoformat()
        
        return response
    
    @app.get("/mcp/tasks/{task_id}/result", dependencies=[Depends(verify_auth)])
    async def get_task_result(task_id: str) -> Dict[str, Any]:
        """
        Get result of a completed task.
        
        Args:
            task_id: Task identifier
        
        Returns:
            Task result
        """
        if not task_manager_instance:
            raise HTTPException(status_code=503, detail="Task manager not initialized")
        
        task = task_manager_instance.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        
        from .task_manager import TaskStatus
        
        if task.status == TaskStatus.RUNNING or task.status == TaskStatus.PENDING:
            raise HTTPException(
                status_code=409,
                detail=f"Task not yet completed. Current status: {task.status}"
            )
        
        if task.status == TaskStatus.FAILED:
            raise HTTPException(
                status_code=500,
                detail=f"Task failed: {task.error}"
            )
        
        # Extract text from result (which is Sequence[TextContent])
        if task.result and len(task.result) > 0:
            response_text = task.result[0].text
            response_data = json.loads(response_text)
            return {"result": response_data}
        else:
            return {"result": None}
    
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

