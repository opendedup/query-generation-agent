"""
MCP Tool Definitions

Defines available tools for the Query Generation Agent MCP server.
"""

from mcp.types import Tool

# Tool names
GENERATE_QUERIES_TOOL = "generate_queries"


def get_available_tools() -> list[Tool]:
    """
    Get list of available MCP tools.
    
    Returns:
        List of Tool definitions
    """
    return [
        Tool(
            name=GENERATE_QUERIES_TOOL,
            description=(
                "Generate and validate BigQuery SQL queries from a data science insight. "
                "Takes an insight (question) and dataset metadata, generates multiple query candidates, "
                "validates them through iterative refinement, and returns validated queries with "
                "alignment scores and descriptions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "insight": {
                        "type": "string",
                        "description": "The data science insight or question to answer",
                        "minLength": 10
                    },
                    "datasets": {
                        "type": "array",
                        "description": "Array of dataset metadata from data discovery",
                        "items": {
                            "type": "object",
                            "properties": {
                                "project_id": {
                                    "type": "string",
                                    "description": "GCP project ID"
                                },
                                "dataset_id": {
                                    "type": "string",
                                    "description": "BigQuery dataset ID"
                                },
                                "table_id": {
                                    "type": "string",
                                    "description": "BigQuery table ID"
                                },
                                "asset_type": {
                                    "type": "string",
                                    "description": "Asset type (table, view, etc.)"
                                },
                                "row_count": {
                                    "type": ["integer", "null"],
                                    "description": "Number of rows"
                                },
                                "size_bytes": {
                                    "type": ["integer", "null"],
                                    "description": "Size in bytes"
                                },
                                "column_count": {
                                    "type": ["integer", "null"],
                                    "description": "Number of columns"
                                },
                                "schema_fields": {
                                    "type": "array",
                                    "description": "Schema fields with names, types, and descriptions",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "type": {"type": "string"},
                                            "description": {"type": "string"}
                                        }
                                    }
                                },
                                "full_markdown": {
                                    "type": "string",
                                    "description": "Complete markdown documentation"
                                },
                                "has_pii": {
                                    "type": "boolean",
                                    "description": "Contains PII data"
                                },
                                "has_phi": {
                                    "type": "boolean",
                                    "description": "Contains PHI data"
                                },
                                "environment": {
                                    "type": ["string", "null"],
                                    "description": "Environment (prod, staging, dev)"
                                },
                                "owner_email": {
                                    "type": ["string", "null"],
                                    "description": "Dataset owner email"
                                },
                                "tags": {
                                    "type": "array",
                                    "description": "Dataset tags",
                                    "items": {"type": "string"}
                                }
                            },
                            "required": ["project_id", "dataset_id", "table_id", "asset_type", "schema_fields", "full_markdown"]
                        },
                        "minItems": 1
                    },
                    "max_queries": {
                        "type": "integer",
                        "description": "Maximum number of queries to generate (default: 3)",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 3
                    },
                    "max_iterations": {
                        "type": "integer",
                        "description": "Maximum refinement iterations per query (default: 10)",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 10
                    },
                    "require_alignment_check": {
                        "type": "boolean",
                        "description": "Require LLM alignment validation (default: true)",
                        "default": True
                    },
                    "allow_cross_dataset": {
                        "type": "boolean",
                        "description": "Allow joins across datasets (default: true)",
                        "default": True
                    }
                },
                "required": ["insight", "datasets"]
            }
        )
    ]

