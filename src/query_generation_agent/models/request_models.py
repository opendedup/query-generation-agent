"""
Request Models for Query Generation Agent

Defines input data structures for query generation requests.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DatasetMetadata(BaseModel):
    """
    Metadata for a discovered dataset from data-discovery-agent.
    
    Contains schema information and documentation needed for query generation.
    """
    
    # Identity
    project_id: str = Field(..., description="GCP project ID")
    dataset_id: str = Field(..., description="BigQuery dataset ID")
    table_id: str = Field(..., description="BigQuery table ID")
    asset_type: str = Field(..., description="Asset type (table, view, etc.)")
    
    # Size metrics
    row_count: Optional[int] = Field(None, description="Number of rows in table")
    size_bytes: Optional[int] = Field(None, description="Size in bytes")
    column_count: Optional[int] = Field(None, description="Number of columns")
    
    # Schema information
    schema_fields: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Field names, types, and descriptions"
    )
    
    # Documentation
    full_markdown: str = Field(..., description="Complete markdown documentation")
    
    # Optional metadata
    has_pii: bool = Field(default=False, description="Contains PII data")
    has_phi: bool = Field(default=False, description="Contains PHI data")
    environment: Optional[str] = Field(None, description="Environment (prod, staging, dev)")
    owner_email: Optional[str] = Field(None, description="Dataset owner")
    tags: List[str] = Field(default_factory=list, description="Dataset tags")
    
    def get_full_table_id(self) -> str:
        """
        Get the fully qualified table ID.
        
        Returns:
            Fully qualified table ID in format project.dataset.table
        """
        return f"{self.project_id}.{self.dataset_id}.{self.table_id}"
    
    def get_schema_summary(self) -> str:
        """
        Get a human-readable schema summary.
        
        Returns:
            Formatted string with field names and types
        """
        if not self.schema_fields:
            return "Schema not available"
        
        lines = [f"Table: {self.get_full_table_id()}", "Fields:"]
        for field in self.schema_fields:
            field_name = field.get("name", "unknown")
            field_type = field.get("type", "unknown")
            field_desc = field.get("description", "")
            
            line = f"  - {field_name} ({field_type})"
            if field_desc:
                line += f": {field_desc}"
            lines.append(line)
        
        return "\n".join(lines)
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "project_id": "my-project",
                "dataset_id": "sales",
                "table_id": "transactions",
                "asset_type": "table",
                "row_count": 1000000,
                "size_bytes": 536870912,
                "column_count": 12,
                "schema_fields": [
                    {
                        "name": "transaction_id",
                        "type": "STRING",
                        "description": "Unique transaction identifier"
                    },
                    {
                        "name": "customer_id",
                        "type": "STRING",
                        "description": "Customer identifier"
                    },
                    {
                        "name": "amount",
                        "type": "FLOAT64",
                        "description": "Transaction amount in USD"
                    }
                ],
                "full_markdown": "# Transactions Table\n...",
                "has_pii": True,
                "environment": "prod"
            }
        }


class GenerateQueriesRequest(BaseModel):
    """
    Request to generate SQL queries for a data science insight.
    
    Takes an insight (question) and relevant datasets, generates and validates
    multiple SQL queries that answer the insight.
    """
    
    insight: str = Field(
        ...,
        description="The data science question or insight to answer",
        min_length=10
    )
    
    datasets: List[DatasetMetadata] = Field(
        ...,
        description="Discovered datasets to query",
        min_length=1
    )
    
    max_queries: int = Field(
        default=3,
        description="Maximum number of queries to generate",
        ge=1,
        le=10
    )
    
    max_iterations: int = Field(
        default=10,
        description="Maximum refinement iterations per query",
        ge=1,
        le=20
    )
    
    require_alignment_check: bool = Field(
        default=True,
        description="Require LLM alignment validation"
    )
    
    allow_cross_dataset: bool = Field(
        default=True,
        description="Allow joins across datasets"
    )
    
    def get_datasets_summary(self) -> str:
        """
        Get summary of all datasets in the request.
        
        Returns:
            Formatted string with dataset information
        """
        lines = [f"Insight: {self.insight}", f"\nDatasets ({len(self.datasets)}):"]
        
        for i, dataset in enumerate(self.datasets, 1):
            lines.append(f"\n{i}. {dataset.get_full_table_id()}")
            lines.append(f"   Rows: {dataset.row_count:,}" if dataset.row_count else "   Rows: unknown")
            lines.append(f"   Columns: {dataset.column_count}" if dataset.column_count else "   Columns: unknown")
        
        return "\n".join(lines)
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "insight": "What is the average transaction value by payment method for Q4 2024?",
                "datasets": [
                    {
                        "project_id": "my-project",
                        "dataset_id": "sales",
                        "table_id": "transactions",
                        "asset_type": "table",
                        "row_count": 1000000,
                        "schema_fields": [],
                        "full_markdown": "# Transactions Table\n..."
                    }
                ],
                "max_queries": 3,
                "max_iterations": 10
            }
        }

