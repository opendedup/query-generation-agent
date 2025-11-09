"""
MCP Tool Definitions

Defines available tools for the Query Generation Agent MCP server.

Note: The datasets parameter accepts the BigQuery writer schema format
(DiscoveredAssetDict) from the data-discovery-agent for seamless integration.
"""

from mcp.types import Tool

# Tool names
GENERATE_QUERIES_TOOL = "generate_queries"
GENERATE_VIEWS_TOOL = "generate_views"


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
                    "dataset_ids": {
                        "type": "array",
                        "description": "Array of dataset identifiers to fetch from data-discovery-agent",
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
                                }
                            },
                            "required": ["project_id", "dataset_id", "table_id"]
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
                        "description": "Maximum refinement iterations per query (default: 5)",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5
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
                    "llm_mode": {
                        "type": "string",
                        "description": "LLM model mode: 'fast_llm' uses gemini-2.5-flash, 'detailed_llm' uses gemini-2.5-pro (default: fast_llm)",
                        "enum": ["fast_llm", "detailed_llm"],
                        "default": "fast_llm"
                    },
                    "stop_on_first_valid": {
                        "type": "boolean",
                        "description": "Stop validation after first valid query is found (default: false). When true, generates all candidates first, then validates one-by-one until first success.",
                        "default": True
                    },
                    "target_table_name": {
                        "type": "string",
                        "description": "Name of target table/view these queries populate (used for naming)"
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
                "required": ["insight", "dataset_ids"]
            }
        ),
        Tool(
            name=GENERATE_VIEWS_TOOL,
            description=(
                "Generate CREATE VIEW DDL statements from PRP data requirements (Section 9). "
                "Parses target schemas from PRP markdown, generates SQL that transforms "
                "source tables into required format. Returns validated DDL statements in "
                "the same JSON format as generate_queries."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prp_markdown": {
                        "type": "string",
                        "description": "PRP markdown containing Section 9: Data Requirements with table schemas",
                        "minLength": 50
                    },
                    "source_datasets": {
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
                                "asset_type": {
                                    "type": "string",
                                    "description": "Asset type (table, view, etc.)"
                                },
                                # Schema
                                "schema": {
                                    "type": "array",
                                    "description": "Schema fields with names, types, modes, descriptions",
                                    "items": {"type": "object"}
                                },
                                # Optional metadata
                                "description": {"type": ["string", "null"]},
                                "row_count": {"type": ["integer", "null"]},
                                "column_count": {"type": ["integer", "null"]}
                            },
                            "required": ["table_id", "project_id", "dataset_id", "asset_type", "schema"]
                        }
                    },
                    "target_project": {
                        "type": "string",
                        "description": "GCP project ID where views should be created (optional)"
                    },
                    "target_dataset": {
                        "type": "string",
                        "description": "BigQuery dataset where views should be created (optional)"
                    },
                    "llm_mode": {
                        "type": "string",
                        "description": "LLM model mode: 'fast_llm' uses gemini-2.5-flash, 'detailed_llm' uses gemini-2.5-pro (default: fast_llm)",
                        "enum": ["fast_llm", "detailed_llm"],
                        "default": "fast_llm"
                    },
                    "stop_on_first_valid": {
                        "type": "boolean",
                        "description": "Stop validation after first valid view is found (default: true). When true, generates all view DDL first, then validates one-by-one until first success.",
                        "default": True
                    }
                },
                "required": ["prp_markdown", "source_datasets"]
            }
        )
    ]

