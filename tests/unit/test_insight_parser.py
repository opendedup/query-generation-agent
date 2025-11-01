"""
Unit tests for InsightParser.

Tests LLM-based extraction of structured context from insight text.
"""

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pytest_mock.plugin import MockerFixture

from query_generation_agent.generation.insight_parser import (
    ExtractionResponseSchema,
    InsightParser,
)
from query_generation_agent.models.request_models import InsightContext


def test_extraction_response_schema_validation() -> None:
    """Test ExtractionResponseSchema validation."""
    # Valid schema
    schema = ExtractionResponseSchema(
        cleaned_insight="What is the average transaction value by region?",
        example_queries=["SELECT AVG(amount) FROM transactions"],
        referenced_datasets=["transactions"],
        pattern_keywords=["aggregation"],
        primary_intent="aggregation",
        reasoning="User wants to calculate average transaction values",
        confidence=0.9
    )
    
    assert schema.cleaned_insight == "What is the average transaction value by region?"
    assert len(schema.example_queries) == 1
    assert schema.confidence == 0.9
    
    # Test default values
    schema_minimal = ExtractionResponseSchema(
        cleaned_insight="Test insight",
        reasoning="Test reasoning"
    )
    
    assert schema_minimal.example_queries == []
    assert schema_minimal.referenced_datasets == []
    assert schema_minimal.confidence == 0.5


def test_insight_parser_initialization(mocker: "MockerFixture") -> None:
    """Test InsightParser initialization."""
    mock_gemini_client = mocker.Mock()
    parser = InsightParser(mock_gemini_client)
    
    assert parser.gemini_client == mock_gemini_client
    assert len(parser.INTENT_TYPES) > 0
    assert len(parser.PATTERN_KEYWORDS) > 0
    assert "cohort_analysis" in parser.INTENT_TYPES
    assert "aggregation" in parser.PATTERN_KEYWORDS


def test_insight_parser_build_extraction_prompt() -> None:
    """Test extraction prompt building."""
    from query_generation_agent.clients.gemini_client import GeminiClient
    
    # Create parser with mock client
    mock_client = type('MockClient', (), {})()
    parser = InsightParser(mock_client)  # type: ignore
    
    insight = "I want to analyze customer cohorts similar to this query"
    prompt = parser._build_extraction_prompt(insight)
    
    assert "INSIGHT TEXT:" in prompt
    assert insight in prompt
    assert "Extract SQL Examples" in prompt
    assert "Identify Dataset References" in prompt
    assert "Detect Query Patterns" in prompt
    assert "Classify Primary Intent" in prompt
    assert "cohort" in prompt
    assert "aggregation" in prompt


def test_insight_parser_fallback_context(mocker: "MockerFixture") -> None:
    """Test fallback context creation when LLM fails."""
    mock_gemini_client = mocker.Mock()
    parser = InsightParser(mock_gemini_client)
    
    insight = "What is the average transaction value?"
    context = parser._create_fallback_context(insight)
    
    assert isinstance(context, InsightContext)
    assert context.original_text == insight
    assert context.cleaned_text == insight
    assert context.example_queries == []
    assert context.referenced_datasets == []
    assert context.confidence == 0.0
    assert "fallback" in context.reasoning.lower()


def test_insight_parser_parse_with_llm_success(mocker: "MockerFixture") -> None:
    """Test successful insight parsing with LLM."""
    mock_gemini_client = mocker.Mock()
    
    # Mock successful extraction
    mock_gemini_client.extract_insight_context.return_value = (
        True,  # success
        None,  # error_msg
        {  # extracted_data
            "cleaned_insight": "Calculate average transaction value by payment method",
            "example_queries": ["SELECT AVG(amount) FROM transactions GROUP BY payment_method"],
            "referenced_datasets": ["transactions"],
            "pattern_keywords": ["aggregation", "group_by"],
            "primary_intent": "aggregation",
            "reasoning": "User wants aggregation analysis grouped by payment method",
            "confidence": 0.85
        },
        {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}  # usage
    )
    
    parser = InsightParser(mock_gemini_client)
    
    insight = "What is the average transaction value by payment method?"
    context = parser.parse(insight, llm_mode="fast_llm")
    
    # Verify extraction was called
    assert mock_gemini_client.extract_insight_context.called
    
    # Verify context structure
    assert isinstance(context, InsightContext)
    assert context.original_text == insight
    assert context.cleaned_text == "Calculate average transaction value by payment method"
    assert len(context.example_queries) == 1
    assert "transactions" in context.referenced_datasets
    assert "aggregation" in context.pattern_keywords
    assert context.inferred_intent == "aggregation"
    assert context.confidence == 0.85
    assert "aggregation analysis" in context.reasoning


def test_insight_parser_parse_with_llm_failure(mocker: "MockerFixture") -> None:
    """Test insight parsing when LLM fails."""
    mock_gemini_client = mocker.Mock()
    
    # Mock failed extraction
    mock_gemini_client.extract_insight_context.return_value = (
        False,  # success
        "API Error",  # error_msg
        None,  # extracted_data
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}  # usage
    )
    
    parser = InsightParser(mock_gemini_client)
    
    insight = "What is the average transaction value?"
    context = parser.parse(insight, llm_mode="fast_llm")
    
    # Should return fallback context
    assert isinstance(context, InsightContext)
    assert context.confidence == 0.0
    assert context.original_text == insight
    assert context.cleaned_text == insight
    assert context.example_queries == []


def test_insight_parser_parse_with_sql_examples(mocker: "MockerFixture") -> None:
    """Test parsing insight that contains SQL examples."""
    mock_gemini_client = mocker.Mock()
    
    # Mock extraction with SQL examples
    mock_gemini_client.extract_insight_context.return_value = (
        True,
        None,
        {
            "cleaned_insight": "Analyze customer cohorts [SQL Example Extracted]",
            "example_queries": [
                "SELECT DATE_TRUNC(signup_date, MONTH) as cohort, COUNT(*) as users FROM users GROUP BY cohort"
            ],
            "referenced_datasets": ["users"],
            "pattern_keywords": ["cohort", "aggregation", "time_series"],
            "primary_intent": "cohort_analysis",
            "reasoning": "User wants cohort analysis with temporal grouping",
            "confidence": 0.95
        },
        {"prompt_tokens": 150, "completion_tokens": 75, "total_tokens": 225}
    )
    
    parser = InsightParser(mock_gemini_client)
    
    insight = """I want to analyze customer cohorts similar to:
    ```sql
    SELECT DATE_TRUNC(signup_date, MONTH) as cohort, 
           COUNT(*) as users 
    FROM users 
    GROUP BY cohort
    ```
    """
    
    context = parser.parse(insight, llm_mode="fast_llm")
    
    assert len(context.example_queries) == 1
    assert "DATE_TRUNC" in context.example_queries[0]
    assert "cohort_analysis" == context.inferred_intent
    assert context.confidence == 0.95


def test_insight_parser_parse_with_dataset_references(mocker: "MockerFixture") -> None:
    """Test parsing insight with explicit dataset references."""
    mock_gemini_client = mocker.Mock()
    
    # Mock extraction with multiple dataset references
    mock_gemini_client.extract_insight_context.return_value = (
        True,
        None,
        {
            "cleaned_insight": "Calculate customer lifetime value using customer and order data",
            "example_queries": [],
            "referenced_datasets": ["customers", "orders", "transactions"],
            "pattern_keywords": ["join", "aggregation"],
            "primary_intent": "joining",
            "reasoning": "Query requires joining customers with orders to calculate LTV",
            "confidence": 0.8
        },
        {"prompt_tokens": 120, "completion_tokens": 60, "total_tokens": 180}
    )
    
    parser = InsightParser(mock_gemini_client)
    
    insight = "Using the customers and orders tables, calculate customer lifetime value"
    context = parser.parse(insight, llm_mode="fast_llm")
    
    assert len(context.referenced_datasets) >= 2
    assert "customers" in context.referenced_datasets
    assert "orders" in context.referenced_datasets
    assert "joining" == context.inferred_intent


def test_insight_parser_low_confidence_warning(mocker: "MockerFixture", caplog: pytest.LogCaptureFixture) -> None:
    """Test that low confidence extraction logs warning."""
    import logging
    
    caplog.set_level(logging.INFO)
    
    mock_gemini_client = mocker.Mock()
    
    # Mock extraction with low confidence
    mock_gemini_client.extract_insight_context.return_value = (
        True,
        None,
        {
            "cleaned_insight": "Do something with data",
            "example_queries": [],
            "referenced_datasets": [],
            "pattern_keywords": [],
            "primary_intent": None,
            "reasoning": "Insight is very vague and unclear",
            "confidence": 0.2
        },
        {"prompt_tokens": 80, "completion_tokens": 40, "total_tokens": 120}
    )
    
    parser = InsightParser(mock_gemini_client)
    
    insight = "Do something with data"
    context = parser.parse(insight, llm_mode="fast_llm")
    
    assert context.confidence == 0.2
    # Check that info logs were captured during parsing
    assert any("Parsing insight" in record.message for record in caplog.records)


def test_insight_context_dataclass() -> None:
    """Test InsightContext dataclass creation."""
    context = InsightContext(
        original_text="Original insight",
        cleaned_text="Cleaned insight",
        example_queries=["SELECT * FROM table"],
        referenced_datasets=["table1", "table2"],
        pattern_keywords=["aggregation"],
        inferred_intent="aggregation",
        reasoning="Test reasoning",
        confidence=0.75,
        metadata={"extraction_method": "llm"}
    )
    
    assert context.original_text == "Original insight"
    assert context.cleaned_text == "Cleaned insight"
    assert len(context.example_queries) == 1
    assert len(context.referenced_datasets) == 2
    assert context.inferred_intent == "aggregation"
    assert context.confidence == 0.75
    assert context.metadata["extraction_method"] == "llm"

