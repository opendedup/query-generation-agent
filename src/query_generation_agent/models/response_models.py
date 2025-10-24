"""
Response Models for Query Generation Agent

Defines output data structures for query generation responses.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    """
    Result of query validation.
    
    Contains validation status, errors, execution stats, and sample results.
    """
    
    is_valid: bool = Field(..., description="Whether the query is valid")
    error_message: Optional[str] = Field(None, description="Error message if validation failed")
    error_type: Optional[str] = Field(None, description="Type of error (syntax, execution, alignment)")
    
    # Execution statistics
    execution_stats: Optional[Dict[str, Any]] = Field(
        None,
        description="Query execution statistics"
    )
    
    # Sample results from query execution
    sample_results: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="First N rows from query execution"
    )
    
    result_schema: Optional[List[Dict[str, str]]] = Field(
        None,
        description="Schema of result columns"
    )
    
    # Validation details
    syntax_valid: bool = Field(default=False, description="SQL syntax is valid")
    dryrun_valid: bool = Field(default=False, description="BigQuery dry-run succeeded")
    execution_valid: bool = Field(default=False, description="Query executed successfully")
    alignment_valid: bool = Field(default=False, description="Results align with insight")
    
    # Alignment details
    alignment_score: Optional[float] = Field(None, description="Alignment score 0-1")
    alignment_reasoning: Optional[str] = Field(None, description="LLM reasoning for alignment")
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "is_valid": True,
                "error_message": None,
                "execution_stats": {
                    "total_bytes_processed": 1024000,
                    "estimated_cost_usd": 0.005
                },
                "sample_results": [
                    {"payment_method": "credit_card", "avg_amount": 125.50},
                    {"payment_method": "paypal", "avg_amount": 98.25}
                ],
                "result_schema": [
                    {"name": "payment_method", "type": "STRING"},
                    {"name": "avg_amount", "type": "FLOAT64"}
                ],
                "syntax_valid": True,
                "dryrun_valid": True,
                "execution_valid": True,
                "alignment_valid": True,
                "alignment_score": 0.92,
                "alignment_reasoning": "Query correctly calculates average by payment method"
            }
        }


class QueryResult(BaseModel):
    """
    A single generated and validated query result.
    
    Contains the SQL query, description, validation status, and metadata.
    """
    
    # Query
    sql: str = Field(..., description="The BigQuery SQL query")
    description: str = Field(..., description="Natural language description")
    source_tables: List[str] = Field(
        ...,
        description="Fully qualified table names used in query (project.dataset.table)"
    )
    
    # Validation
    validation_status: Literal["valid", "failed"] = Field(
        ...,
        description="Overall validation status"
    )
    validation_details: ValidationResult = Field(..., description="Detailed validation results")
    
    # Alignment
    alignment_score: float = Field(
        ...,
        description="How well query addresses insight (0-1)",
        ge=0,
        le=1
    )
    
    # Metadata
    iterations: int = Field(..., description="Number of refinement iterations", ge=0)
    generation_time_ms: float = Field(..., description="Time to generate and validate", ge=0)
    
    # Optional fields
    estimated_cost_usd: Optional[float] = Field(None, description="Estimated execution cost")
    estimated_bytes_processed: Optional[int] = Field(None, description="Estimated bytes processed")
    
    def is_valid(self) -> bool:
        """
        Check if query is valid.
        
        Returns:
            True if validation_status is 'valid'
        """
        return self.validation_status == "valid"
    
    def get_summary(self) -> str:
        """
        Get human-readable summary of the query result.
        
        Returns:
            Formatted summary string
        """
        lines = [
            f"Status: {self.validation_status}",
            f"Alignment Score: {self.alignment_score:.2f}",
            f"Iterations: {self.iterations}",
            f"Generation Time: {self.generation_time_ms:.0f}ms"
        ]
        
        if self.estimated_cost_usd:
            lines.append(f"Estimated Cost: ${self.estimated_cost_usd:.4f}")
        
        if self.validation_status == "valid":
            lines.append(f"\nDescription: {self.description}")
            lines.append(f"\nSQL:\n{self.sql}")
        else:
            error_msg = self.validation_details.error_message or "Unknown error"
            lines.append(f"\nError: {error_msg}")
        
        return "\n".join(lines)
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "sql": "SELECT payment_method, AVG(amount) as avg_amount FROM `project.sales.transactions` WHERE DATE(timestamp) BETWEEN '2024-10-01' AND '2024-12-31' GROUP BY payment_method",
                "description": "Calculate average transaction amount by payment method for Q4 2024",
                "source_tables": ["project.sales.transactions"],
                "validation_status": "valid",
                "validation_details": {
                    "is_valid": True,
                    "syntax_valid": True,
                    "dryrun_valid": True,
                    "execution_valid": True,
                    "alignment_valid": True,
                    "alignment_score": 0.92
                },
                "alignment_score": 0.92,
                "iterations": 2,
                "generation_time_ms": 3500.0,
                "estimated_cost_usd": 0.005,
                "estimated_bytes_processed": 1024000
            }
        }


class GenerateQueriesResponse(BaseModel):
    """
    Response containing generated and validated queries.
    
    Contains array of query results with metadata about the generation process.
    """
    
    # Results
    queries: List[QueryResult] = Field(
        default_factory=list,
        description="Generated and validated queries"
    )
    
    # Metadata
    total_attempted: int = Field(..., description="Total queries attempted", ge=0)
    total_validated: int = Field(..., description="Number of successfully validated queries", ge=0)
    execution_time_ms: float = Field(..., description="Total execution time", ge=0)
    
    # Request context
    insight: str = Field(..., description="Original insight")
    dataset_count: int = Field(..., description="Number of datasets used", ge=0)
    
    # Optional summary
    summary: Optional[str] = Field(None, description="Human-readable summary")
    warnings: List[str] = Field(default_factory=list, description="Warnings during generation")
    
    def get_valid_queries(self) -> List[QueryResult]:
        """
        Get only the valid queries.
        
        Returns:
            List of QueryResult with validation_status='valid'
        """
        return [q for q in self.queries if q.is_valid()]
    
    def get_best_query(self) -> Optional[QueryResult]:
        """
        Get the query with the highest alignment score.
        
        Returns:
            QueryResult with highest alignment score, or None if no valid queries
        """
        valid_queries = self.get_valid_queries()
        if not valid_queries:
            return None
        return max(valid_queries, key=lambda q: q.alignment_score)
    
    def get_summary_text(self) -> str:
        """
        Get human-readable summary of response.
        
        Returns:
            Formatted summary string
        """
        lines = [
            f"Insight: {self.insight}",
            f"Datasets: {self.dataset_count}",
            f"Queries Attempted: {self.total_attempted}",
            f"Queries Validated: {self.total_validated}",
            f"Execution Time: {self.execution_time_ms:.0f}ms"
        ]
        
        if self.warnings:
            lines.append(f"\nWarnings: {len(self.warnings)}")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
        
        valid_queries = self.get_valid_queries()
        if valid_queries:
            lines.append(f"\nValid Queries: {len(valid_queries)}")
            for i, query in enumerate(valid_queries, 1):
                lines.append(f"\n{i}. {query.description}")
                lines.append(f"   Alignment: {query.alignment_score:.2f}")
                lines.append(f"   Iterations: {query.iterations}")
        else:
            lines.append("\nNo valid queries generated")
        
        return "\n".join(lines)
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "queries": [
                    {
                        "sql": "SELECT payment_method, AVG(amount) FROM transactions GROUP BY 1",
                        "description": "Average transaction by payment method",
                        "source_tables": ["project.sales.transactions"],
                        "validation_status": "valid",
                        "validation_details": {"is_valid": True},
                        "alignment_score": 0.92,
                        "iterations": 2,
                        "generation_time_ms": 3500.0
                    }
                ],
                "total_attempted": 3,
                "total_validated": 1,
                "execution_time_ms": 8500.0,
                "insight": "What is the average transaction value by payment method?",
                "dataset_count": 1
            }
        }

