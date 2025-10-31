"""
JSON-RPC 2.0 utilities for MCP HTTP server.

Implements JSON-RPC 2.0 specification for protocol compliance.
Reference: https://www.jsonrpc.org/specification
"""

from typing import Any, Dict, Optional
from enum import IntEnum


class JsonRpcErrorCode(IntEnum):
    """Standard JSON-RPC 2.0 error codes."""
    
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    
    # Server errors (custom range -32000 to -32099)
    SERVER_NOT_INITIALIZED = -32001
    TOOL_EXECUTION_ERROR = -32002
    TIMEOUT_ERROR = -32003


def create_jsonrpc_request(
    method: str,
    params: Optional[Dict[str, Any]] = None,
    request_id: Optional[int | str] = None
) -> Dict[str, Any]:
    """
    Create a JSON-RPC 2.0 request.
    
    Args:
        method: Method name to call
        params: Parameters for the method
        request_id: Request identifier (omit for notifications)
    
    Returns:
        JSON-RPC 2.0 request object
    """
    request = {
        "jsonrpc": "2.0",
        "method": method,
    }
    
    if params is not None:
        request["params"] = params
    
    if request_id is not None:
        request["id"] = request_id
    
    return request


def create_jsonrpc_success_response(
    result: Any,
    request_id: int | str | None
) -> Dict[str, Any]:
    """
    Create a JSON-RPC 2.0 success response.
    
    Args:
        result: Result data
        request_id: ID from the original request
    
    Returns:
        JSON-RPC 2.0 success response
    """
    return {
        "jsonrpc": "2.0",
        "result": result,
        "id": request_id
    }


def create_jsonrpc_error_response(
    code: int,
    message: str,
    data: Optional[Any] = None,
    request_id: Optional[int | str] = None
) -> Dict[str, Any]:
    """
    Create a JSON-RPC 2.0 error response.
    
    Args:
        code: Error code (standard or custom)
        message: Human-readable error message
        data: Additional error data (optional)
        request_id: ID from the original request (None if request parsing failed)
    
    Returns:
        JSON-RPC 2.0 error response
    """
    error = {
        "code": code,
        "message": message
    }
    
    if data is not None:
        error["data"] = data
    
    return {
        "jsonrpc": "2.0",
        "error": error,
        "id": request_id
    }


def parse_jsonrpc_request(body: Dict[str, Any]) -> tuple[str, Dict[str, Any], int | str | None]:
    """
    Parse and validate a JSON-RPC 2.0 request.
    
    Args:
        body: Request body to parse
    
    Returns:
        Tuple of (method, params, request_id)
    
    Raises:
        ValueError: If request is invalid with error details
    """
    # Check jsonrpc version
    if body.get("jsonrpc") != "2.0":
        raise ValueError(
            create_jsonrpc_error_response(
                code=JsonRpcErrorCode.INVALID_REQUEST,
                message="Missing or invalid 'jsonrpc' field. Must be '2.0'",
                request_id=body.get("id")
            )
        )
    
    # Check method exists
    method = body.get("method")
    if not method or not isinstance(method, str):
        raise ValueError(
            create_jsonrpc_error_response(
                code=JsonRpcErrorCode.INVALID_REQUEST,
                message="Missing or invalid 'method' field",
                request_id=body.get("id")
            )
        )
    
    # Get params (optional)
    params = body.get("params", {})
    if not isinstance(params, dict):
        raise ValueError(
            create_jsonrpc_error_response(
                code=JsonRpcErrorCode.INVALID_REQUEST,
                message="'params' must be an object",
                request_id=body.get("id")
            )
        )
    
    # Get id (optional for notifications)
    request_id = body.get("id")
    
    return method, params, request_id


def create_validation_error_response(
    validator: str,
    message: str,
    path: str,
    validator_value: Any,
    request_id: Optional[int | str] = None
) -> Dict[str, Any]:
    """
    Create a JSON-RPC 2.0 error response for validation failures.
    
    Args:
        validator: Type of validation that failed
        message: Validation error message
        path: Path to the invalid field
        validator_value: Expected validator value
        request_id: Request ID
    
    Returns:
        JSON-RPC 2.0 error response with validation details
    """
    # Build helpful context based on error type
    help_text = None
    if validator == "required":
        help_text = f"Missing required field(s): {validator_value}"
    elif validator == "minLength":
        help_text = f"Field must be at least {validator_value} characters long"
    elif validator == "minimum":
        help_text = f"Value must be at least {validator_value}"
    elif validator == "maximum":
        help_text = f"Value must be at most {validator_value}"
    elif validator == "type":
        help_text = f"Expected type '{validator_value}'"
    elif validator == "minItems":
        help_text = f"Array must contain at least {validator_value} item(s)"
    
    error_data = {
        "validator": validator,
        "path": path,
        "validator_value": validator_value,
    }
    
    if help_text:
        error_data["help"] = help_text
    
    return create_jsonrpc_error_response(
        code=JsonRpcErrorCode.INVALID_PARAMS,
        message=f"Invalid params: {message}",
        data=error_data,
        request_id=request_id
    )

