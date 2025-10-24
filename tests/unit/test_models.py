"""
Unit tests for data models.
"""

import pytest

from query_generation_agent.models.request_models import DatasetMetadata, GenerateQueriesRequest
from query_generation_agent.models.response_models import (
    GenerateQueriesResponse,
    QueryResult,
    ValidationResult,
)


def test_dataset_metadata_creation() -> None:
    """Test DatasetMetadata creation and methods."""
    dataset = DatasetMetadata(
        project_id="my-project",
        dataset_id="sales",
        table_id="transactions",
        asset_type="table",
        row_count=1000000,
        schema_fields=[
            {"name": "id", "type": "STRING", "description": "ID"},
            {"name": "amount", "type": "FLOAT64", "description": "Amount"}
        ],
        full_markdown="# Transactions\nTest table",
        has_pii=True
    )
    
    assert dataset.get_full_table_id() == "my-project.sales.transactions"
    assert dataset.has_pii is True
    assert len(dataset.schema_fields) == 2
    
    schema_summary = dataset.get_schema_summary()
    assert "my-project.sales.transactions" in schema_summary
    assert "id (STRING)" in schema_summary
    assert "amount (FLOAT64)" in schema_summary


def test_generate_queries_request_validation() -> None:
    """Test GenerateQueriesRequest validation."""
    dataset = DatasetMetadata(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
        asset_type="table",
        schema_fields=[],
        full_markdown="Test"
    )
    
    # Valid request
    request = GenerateQueriesRequest(
        insight="What is the average transaction value?",
        datasets=[dataset],
        max_queries=3,
        max_iterations=10
    )
    
    assert request.insight == "What is the average transaction value?"
    assert len(request.datasets) == 1
    assert request.max_queries == 3
    assert request.max_iterations == 10
    
    # Invalid: insight too short
    with pytest.raises(ValueError):
        GenerateQueriesRequest(
            insight="Short",
            datasets=[dataset],
            max_queries=3
        )
    
    # Invalid: no datasets
    with pytest.raises(ValueError):
        GenerateQueriesRequest(
            insight="What is the average transaction value?",
            datasets=[],
            max_queries=3
        )
    
    # Invalid: max_queries out of range
    with pytest.raises(ValueError):
        GenerateQueriesRequest(
            insight="What is the average transaction value?",
            datasets=[dataset],
            max_queries=15  # Max is 10
        )


def test_validation_result() -> None:
    """Test ValidationResult model."""
    result = ValidationResult(
        is_valid=True,
        syntax_valid=True,
        dryrun_valid=True,
        execution_valid=True,
        alignment_valid=True,
        alignment_score=0.92
    )
    
    assert result.is_valid is True
    assert result.alignment_score == 0.92
    assert result.error_message is None


def test_query_result() -> None:
    """Test QueryResult model and methods."""
    validation = ValidationResult(
        is_valid=True,
        syntax_valid=True,
        dryrun_valid=True,
        execution_valid=True,
        alignment_valid=True,
        alignment_score=0.92
    )
    
    query = QueryResult(
        sql="SELECT * FROM table",
        description="Select all rows",
        source_tables=["project.dataset.table"],
        validation_status="valid",
        validation_details=validation,
        alignment_score=0.92,
        iterations=2,
        generation_time_ms=1500.0,
        estimated_cost_usd=0.005
    )
    
    assert query.is_valid() is True
    assert query.alignment_score == 0.92
    assert query.source_tables == ["project.dataset.table"]
    
    summary = query.get_summary()
    assert "Status: valid" in summary
    assert "Alignment Score: 0.92" in summary
    assert "SELECT * FROM table" in summary


def test_generate_queries_response() -> None:
    """Test GenerateQueriesResponse model and methods."""
    validation = ValidationResult(
        is_valid=True,
        syntax_valid=True,
        dryrun_valid=True,
        execution_valid=True,
        alignment_valid=True,
        alignment_score=0.92
    )
    
    query1 = QueryResult(
        sql="SELECT * FROM table1",
        description="Query 1",
        source_tables=["project.dataset.table1"],
        validation_status="valid",
        validation_details=validation,
        alignment_score=0.92,
        iterations=2,
        generation_time_ms=1500.0
    )
    
    query2 = QueryResult(
        sql="SELECT * FROM table2",
        description="Query 2",
        source_tables=["project.dataset.table2"],
        validation_status="failed",
        validation_details=ValidationResult(
            is_valid=False,
            error_message="Syntax error",
            syntax_valid=False,
            dryrun_valid=False,
            execution_valid=False,
            alignment_valid=False
        ),
        alignment_score=0.0,
        iterations=5,
        generation_time_ms=3000.0
    )
    
    response = GenerateQueriesResponse(
        queries=[query1, query2],
        total_attempted=2,
        total_validated=1,
        execution_time_ms=5000.0,
        insight="Test insight",
        dataset_count=1
    )
    
    assert len(response.queries) == 2
    assert response.total_attempted == 2
    assert response.total_validated == 1
    
    valid_queries = response.get_valid_queries()
    assert len(valid_queries) == 1
    assert valid_queries[0].sql == "SELECT * FROM table1"
    
    best_query = response.get_best_query()
    assert best_query is not None
    assert best_query.alignment_score == 0.92
    
    summary = response.get_summary_text()
    assert "Test insight" in summary
    assert "Queries Attempted: 2" in summary
    assert "Queries Validated: 1" in summary

