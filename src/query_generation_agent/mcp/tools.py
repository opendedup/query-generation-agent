"""
MCP Tool Definitions

Defines available tools for the Query Generation Agent MCP server.

Note: The datasets parameter accepts the BigQuery writer schema format
(DiscoveredAssetDict) from the data-discovery-agent for seamless integration.
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
                "alignment scores and descriptions.\n\n"
                "Note: For long-running operations (>30s), use the async endpoint "
                "/mcp/call-tool-async which returns immediately with a task_id for status polling. "
                "This prevents client timeouts on complex query generation tasks."
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
                        "description": (
                            "Array of dataset metadata from data discovery in BigQuery writer schema format "
                            "(DiscoveredAssetDict). Output from get_datasets_for_query_generation tool can be "
                            "passed directly to this parameter."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                # Core identifiers
                                "table_id": {
                                    "type": "string",
                                    "description": "BigQuery table ID"
                                },
                                "project_id": {
                                    "type": "string",
                                    "description": "GCP project ID"
                                },
                                "dataset_id": {
                                    "type": "string",
                                    "description": "BigQuery dataset ID"
                                },
                                # Metadata
                                "description": {
                                    "type": ["string", "null"],
                                    "description": "Table description"
                                },
                                "table_type": {
                                    "type": ["string", "null"],
                                    "description": "Table type (TABLE, VIEW, MATERIALIZED_VIEW, etc.)"
                                },
                                "asset_type": {
                                    "type": "string",
                                    "description": "Asset type (table, view, etc.) - same as table_type"
                                },
                                # Timestamps
                                "created": {
                                    "type": ["string", "null"],
                                    "description": "Creation timestamp (ISO format)"
                                },
                                "last_modified": {
                                    "type": ["string", "null"],
                                    "description": "Last modified timestamp (ISO format)"
                                },
                                "last_accessed": {
                                    "type": ["string", "null"],
                                    "description": "Last accessed timestamp (ISO format)"
                                },
                                # Statistics
                                "row_count": {
                                    "type": ["integer", "null"],
                                    "description": "Number of rows"
                                },
                                "column_count": {
                                    "type": ["integer", "null"],
                                    "description": "Number of columns"
                                },
                                "size_bytes": {
                                    "type": ["integer", "null"],
                                    "description": "Size in bytes"
                                },
                                # Security & Governance
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
                                    "description": "Environment (PROD, DEV, STAGING, etc.)"
                                },
                                # Labels and tags
                                "labels": {
                                    "type": "array",
                                    "description": "Key-value labels",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "key": {"type": "string"},
                                            "value": {"type": "string"}
                                        }
                                    }
                                },
                                # Schema
                                "schema": {
                                    "type": "array",
                                    "description": "Schema fields with names, types, modes, descriptions, and sample values",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "type": {"type": "string"},
                                            "mode": {"type": ["string", "null"]},
                                            "description": {"type": ["string", "null"]},
                                            "sample_values": {
                                                "type": ["array", "null"],
                                                "items": {"type": "string"}
                                            }
                                        }
                                    }
                                },
                                # AI-generated insights
                                "analytical_insights": {
                                    "type": "array",
                                    "description": "AI-generated analytical insights",
                                    "items": {"type": "string"}
                                },
                                # Lineage
                                "lineage": {
                                    "type": "array",
                                    "description": "Lineage relationships",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "source": {"type": "string"},
                                            "target": {"type": "string"}
                                        }
                                    }
                                },
                                # Profiling
                                "column_profiles": {
                                    "type": "array",
                                    "description": "Column profiling statistics",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "column_name": {"type": "string"},
                                            "profile_type": {"type": ["string", "null"]},
                                            "min_value": {"type": ["string", "null"]},
                                            "max_value": {"type": ["string", "null"]},
                                            "avg_value": {"type": ["string", "null"]},
                                            "distinct_count": {"type": ["integer", "null"]},
                                            "null_percentage": {"type": ["number", "null"]}
                                        }
                                    }
                                },
                                # Metrics
                                "key_metrics": {
                                    "type": "array",
                                    "description": "Key metrics (completeness, freshness, cost, etc.)",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "metric_name": {"type": "string"},
                                            "metric_value": {"type": "string"}
                                        }
                                    }
                                },
                                # Run metadata
                                "run_timestamp": {
                                    "type": "string",
                                    "description": "Run timestamp (ISO format)"
                                },
                                "insert_timestamp": {
                                    "type": "string",
                                    "description": "Insert timestamp (ISO format or 'AUTO')"
                                },
                                # Additional MCP fields
                                "full_markdown": {
                                    "type": ["string", "null"],
                                    "description": "Complete markdown documentation"
                                },
                                "owner_email": {
                                    "type": ["string", "null"],
                                    "description": "Asset owner email"
                                },
                                "tags": {
                                    "type": "array",
                                    "description": "Additional tags",
                                    "items": {"type": "string"}
                                }
                            },
                            "required": ["table_id", "project_id", "dataset_id", "asset_type", "schema"]
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
                    },
                    "discovery_metadata": {
                        "type": "object",
                        "description": (
                            "Optional metadata from PRP discovery process. Includes query execution details, "
                            "refinements made, and summary statistics. Passed through from data-discovery-agent "
                            "for full traceability of dataset selection."
                        ),
                        "properties": {
                            "queries_executed": {
                                "type": "array",
                                "description": "List of queries executed during discovery",
                                "items": {"type": "object"}
                            },
                            "refinements_made": {
                                "type": "array",
                                "description": "Query refinements applied during discovery",
                                "items": {"type": "object"}
                            },
                            "summary": {
                                "type": "object",
                                "description": "Summary statistics of discovery process"
                            }
                        }
                    }
                },
                "required": ["insight", "datasets"]
            }
        )
    ]

