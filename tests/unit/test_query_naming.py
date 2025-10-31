"""
Tests for descriptive query naming functionality.

Tests the rule-based, LLM-based, and hybrid query naming strategies.
"""

from typing import TYPE_CHECKING

import pytest

from query_generation_agent.generation.query_refiner import QueryRefiner
from query_generation_agent.models.validation_models import QueryValidationHistory

if TYPE_CHECKING:
    from pytest_mock.plugin import MockerFixture


class TestRuleBasedQueryNaming:
    """Test rule-based query name generation."""

    def test_simple_inner_join_description(self, mocker: "MockerFixture") -> None:
        """Test extraction from simple inner join description."""
        # Mock dependencies
        mock_gemini = mocker.Mock()
        mock_dryrun = mocker.Mock()
        mock_alignment = mocker.Mock()
        
        refiner = QueryRefiner(
            gemini_client=mock_gemini,
            dryrun_validator=mock_dryrun,
            alignment_validator=mock_alignment,
            query_naming_strategy="rule_based"
        )
        
        description = "Brief description: [Simple and direct approach using an INNER JOIN between the predictions and post-game results tables.]"
        sql = "SELECT * FROM table1 INNER JOIN table2 ON table1.id = table2.id"
        
        name = refiner._generate_name_rule_based(description, sql, query_index=0)
        
        assert "inner_join" in name
        assert "simple" in name or "direct" in name
        assert len(name) <= 60
        assert "_" in name  # Should be snake_case

    def test_cte_modular_approach(self, mocker: "MockerFixture") -> None:
        """Test extraction from CTE modular approach description."""
        mock_gemini = mocker.Mock()
        mock_dryrun = mocker.Mock()
        mock_alignment = mocker.Mock()
        
        refiner = QueryRefiner(
            gemini_client=mock_gemini,
            dryrun_validator=mock_dryrun,
            alignment_validator=mock_alignment,
            query_naming_strategy="rule_based"
        )
        
        description = "Brief description: [Modular approach using Common Table Expressions (CTEs) to separate the logic for game outcomes and model bets.]"
        sql = """
        WITH GameData AS (
            SELECT * FROM games
        ),
        ModelBets AS (
            SELECT * FROM bets
        )
        SELECT * FROM GameData JOIN ModelBets ON GameData.id = ModelBets.game_id
        """
        
        name = refiner._generate_name_rule_based(description, sql, query_index=1)
        
        assert "cte" in name or "modular" in name
        assert "approach" in name
        assert len(name) <= 60

    def test_window_function_description(self, mocker: "MockerFixture") -> None:
        """Test extraction from window function description."""
        mock_gemini = mocker.Mock()
        mock_dryrun = mocker.Mock()
        mock_alignment = mocker.Mock()
        
        refiner = QueryRefiner(
            gemini_client=mock_gemini,
            dryrun_validator=mock_dryrun,
            alignment_validator=mock_alignment,
            query_naming_strategy="rule_based"
        )
        
        description = "Uses window functions to rank results by score"
        sql = """
        SELECT 
            ROW_NUMBER() OVER (PARTITION BY team ORDER BY score DESC) as rank,
            team, 
            score
        FROM results
        """
        
        name = refiner._generate_name_rule_based(description, sql, query_index=0)
        
        assert "window" in name
        assert len(name) <= 60

    def test_aggregate_with_group_by(self, mocker: "MockerFixture") -> None:
        """Test extraction from aggregate query description."""
        mock_gemini = mocker.Mock()
        mock_dryrun = mocker.Mock()
        mock_alignment = mocker.Mock()
        
        refiner = QueryRefiner(
            gemini_client=mock_gemini,
            dryrun_validator=mock_dryrun,
            alignment_validator=mock_alignment,
            query_naming_strategy="rule_based"
        )
        
        description = "Comprehensive analysis using aggregations and grouping"
        sql = """
        SELECT 
            team,
            COUNT(*) as games,
            AVG(score) as avg_score,
            SUM(wins) as total_wins
        FROM results
        GROUP BY team
        """
        
        name = refiner._generate_name_rule_based(description, sql, query_index=0)
        
        assert "aggregate" in name
        assert len(name) <= 60

    def test_subquery_description(self, mocker: "MockerFixture") -> None:
        """Test extraction from subquery description."""
        mock_gemini = mocker.Mock()
        mock_dryrun = mocker.Mock()
        mock_alignment = mocker.Mock()
        
        refiner = QueryRefiner(
            gemini_client=mock_gemini,
            dryrun_validator=mock_dryrun,
            alignment_validator=mock_alignment,
            query_naming_strategy="rule_based"
        )
        
        description = "Uses a nested subquery to filter results"
        sql = """
        SELECT * FROM games
        WHERE team_id IN (
            SELECT id FROM teams WHERE division = 'NFC'
        )
        """
        
        name = refiner._generate_name_rule_based(description, sql, query_index=0)
        
        assert "subquery" in name
        assert len(name) <= 60

    def test_fallback_to_description_words(self, mocker: "MockerFixture") -> None:
        """Test fallback to extracting words from description."""
        mock_gemini = mocker.Mock()
        mock_dryrun = mocker.Mock()
        mock_alignment = mocker.Mock()
        
        refiner = QueryRefiner(
            gemini_client=mock_gemini,
            dryrun_validator=mock_dryrun,
            alignment_validator=mock_alignment,
            query_naming_strategy="rule_based"
        )
        
        description = "Brief description: Calculate team performance metrics by season"
        sql = "SELECT team, season, score FROM results"
        
        name = refiner._generate_name_rule_based(description, sql, query_index=0)
        
        # Should extract some words from description
        assert len(name) > 0
        assert len(name) <= 60
        assert "_" in name

    def test_ultimate_fallback(self, mocker: "MockerFixture") -> None:
        """Test ultimate fallback to index-based naming."""
        mock_gemini = mocker.Mock()
        mock_dryrun = mocker.Mock()
        mock_alignment = mocker.Mock()
        
        refiner = QueryRefiner(
            gemini_client=mock_gemini,
            dryrun_validator=mock_dryrun,
            alignment_validator=mock_alignment,
            query_naming_strategy="rule_based"
        )
        
        description = ""  # Empty description
        sql = "SELECT 1"  # Simple SQL with no patterns
        
        name = refiner._generate_name_rule_based(description, sql, query_index=5)
        
        assert name == "query_5"

    def test_multiple_patterns_priority(self, mocker: "MockerFixture") -> None:
        """Test that multiple patterns are combined appropriately."""
        mock_gemini = mocker.Mock()
        mock_dryrun = mocker.Mock()
        mock_alignment = mocker.Mock()
        
        refiner = QueryRefiner(
            gemini_client=mock_gemini,
            dryrun_validator=mock_dryrun,
            alignment_validator=mock_alignment,
            query_naming_strategy="rule_based"
        )
        
        description = "Complex approach using CTEs and window functions with aggregations"
        sql = """
        WITH RankedData AS (
            SELECT 
                ROW_NUMBER() OVER (PARTITION BY team ORDER BY score DESC) as rank,
                team,
                COUNT(*) as games,
                AVG(score) as avg_score
            FROM results
            GROUP BY team
        )
        SELECT * FROM RankedData
        """
        
        name = refiner._generate_name_rule_based(description, sql, query_index=0)
        
        # Should contain multiple relevant terms
        assert any(term in name for term in ["cte", "window", "aggregate", "complex"])
        assert len(name) <= 60


class TestLLMQueryNaming:
    """Test LLM-based query name generation."""

    def test_llm_naming_success(self, mocker: "MockerFixture") -> None:
        """Test successful LLM-based name generation."""
        # Mock Gemini client
        mock_response = mocker.Mock()
        mock_response.text = "cte_confidence_scoring_approach"
        
        mock_gemini = mocker.Mock()
        mock_gemini.client.generate_content.return_value = mock_response
        
        mock_dryrun = mocker.Mock()
        mock_alignment = mocker.Mock()
        
        refiner = QueryRefiner(
            gemini_client=mock_gemini,
            dryrun_validator=mock_dryrun,
            alignment_validator=mock_alignment,
            query_naming_strategy="llm"
        )
        
        description = "Uses CTEs to score predictions with confidence intervals"
        sql = "WITH scores AS (SELECT * FROM predictions) SELECT * FROM scores"
        insight = "How confident are our predictions?"
        
        name = refiner._generate_name_with_llm(description, sql, insight, query_index=0)
        
        assert name == "cte_confidence_scoring_approach"
        mock_gemini.client.generate_content.assert_called_once()

    def test_llm_naming_sanitization(self, mocker: "MockerFixture") -> None:
        """Test that LLM output is properly sanitized."""
        # Mock Gemini client with response containing special characters
        mock_response = mocker.Mock()
        mock_response.text = "CTE-Based Scoring (Approach!)"
        
        mock_gemini = mocker.Mock()
        mock_gemini.client.generate_content.return_value = mock_response
        
        mock_dryrun = mocker.Mock()
        mock_alignment = mocker.Mock()
        
        refiner = QueryRefiner(
            gemini_client=mock_gemini,
            dryrun_validator=mock_dryrun,
            alignment_validator=mock_alignment,
            query_naming_strategy="llm"
        )
        
        description = "Test description"
        sql = "SELECT * FROM test"
        insight = "Test insight"
        
        name = refiner._generate_name_with_llm(description, sql, insight, query_index=0)
        
        # Should be sanitized to snake_case
        assert name == "cte_based_scoring_approach"
        assert name.islower()
        assert " " not in name
        assert "-" not in name
        assert "(" not in name
        assert ")" not in name
        assert "!" not in name

    def test_llm_naming_length_validation(self, mocker: "MockerFixture") -> None:
        """Test that overly long names are rejected."""
        # Mock Gemini client with very long response
        mock_response = mocker.Mock()
        mock_response.text = "a" * 100  # Too long
        
        mock_gemini = mocker.Mock()
        mock_gemini.client.generate_content.return_value = mock_response
        
        mock_dryrun = mocker.Mock()
        mock_alignment = mocker.Mock()
        
        refiner = QueryRefiner(
            gemini_client=mock_gemini,
            dryrun_validator=mock_dryrun,
            alignment_validator=mock_alignment,
            query_naming_strategy="llm"
        )
        
        description = "Test description"
        sql = "SELECT * FROM test"
        insight = "Test insight"
        
        name = refiner._generate_name_with_llm(description, sql, insight, query_index=0)
        
        # Should reject and return None
        assert name is None

    def test_llm_naming_too_short(self, mocker: "MockerFixture") -> None:
        """Test that very short names are rejected."""
        mock_response = mocker.Mock()
        mock_response.text = "ab"  # Too short
        
        mock_gemini = mocker.Mock()
        mock_gemini.client.generate_content.return_value = mock_response
        
        mock_dryrun = mocker.Mock()
        mock_alignment = mocker.Mock()
        
        refiner = QueryRefiner(
            gemini_client=mock_gemini,
            dryrun_validator=mock_dryrun,
            alignment_validator=mock_alignment,
            query_naming_strategy="llm"
        )
        
        description = "Test description"
        sql = "SELECT * FROM test"
        insight = "Test insight"
        
        name = refiner._generate_name_with_llm(description, sql, insight, query_index=0)
        
        assert name is None

    def test_llm_naming_exception_handling(self, mocker: "MockerFixture") -> None:
        """Test that exceptions are handled gracefully."""
        # Mock Gemini client to raise exception
        mock_gemini = mocker.Mock()
        mock_gemini.client.generate_content.side_effect = Exception("API Error")
        
        mock_dryrun = mocker.Mock()
        mock_alignment = mocker.Mock()
        
        refiner = QueryRefiner(
            gemini_client=mock_gemini,
            dryrun_validator=mock_dryrun,
            alignment_validator=mock_alignment,
            query_naming_strategy="llm"
        )
        
        description = "Test description"
        sql = "SELECT * FROM test"
        insight = "Test insight"
        
        name = refiner._generate_name_with_llm(description, sql, insight, query_index=0)
        
        # Should return None on exception
        assert name is None


class TestHybridQueryNaming:
    """Test hybrid query naming strategy."""

    def test_hybrid_llm_success(self, mocker: "MockerFixture") -> None:
        """Test hybrid strategy when LLM succeeds."""
        # Mock successful LLM response
        mock_response = mocker.Mock()
        mock_response.text = "optimized_cte_strategy"
        
        mock_gemini = mocker.Mock()
        mock_gemini.client.generate_content.return_value = mock_response
        
        mock_dryrun = mocker.Mock()
        mock_alignment = mocker.Mock()
        
        refiner = QueryRefiner(
            gemini_client=mock_gemini,
            dryrun_validator=mock_dryrun,
            alignment_validator=mock_alignment,
            query_naming_strategy="hybrid"
        )
        
        description = "Optimized query using CTEs"
        sql = "WITH data AS (SELECT * FROM table1) SELECT * FROM data"
        insight = "Get optimized results"
        
        name = refiner._generate_descriptive_query_name(
            description=description,
            sql=sql,
            insight=insight,
            query_index=0,
            naming_strategy="hybrid"
        )
        
        # Should use LLM result
        assert name == "optimized_cte_strategy"
        mock_gemini.client.generate_content.assert_called_once()

    def test_hybrid_llm_failure_fallback(self, mocker: "MockerFixture") -> None:
        """Test hybrid strategy falls back to rule-based when LLM fails."""
        # Mock failed LLM response
        mock_gemini = mocker.Mock()
        mock_gemini.client.generate_content.side_effect = Exception("API Error")
        
        mock_dryrun = mocker.Mock()
        mock_alignment = mocker.Mock()
        
        refiner = QueryRefiner(
            gemini_client=mock_gemini,
            dryrun_validator=mock_dryrun,
            alignment_validator=mock_alignment,
            query_naming_strategy="hybrid"
        )
        
        description = "Simple approach using INNER JOIN between tables"
        sql = "SELECT * FROM table1 INNER JOIN table2 ON table1.id = table2.id"
        insight = "Join data sources"
        
        name = refiner._generate_descriptive_query_name(
            description=description,
            sql=sql,
            insight=insight,
            query_index=0,
            naming_strategy="hybrid"
        )
        
        # Should fallback to rule-based extraction
        assert "inner_join" in name or "simple" in name
        assert name != "query_0"  # Should not be ultimate fallback

    def test_hybrid_no_insight_uses_rule_based(self, mocker: "MockerFixture") -> None:
        """Test hybrid strategy uses rule-based when no insight provided."""
        mock_gemini = mocker.Mock()
        mock_dryrun = mocker.Mock()
        mock_alignment = mocker.Mock()
        
        refiner = QueryRefiner(
            gemini_client=mock_gemini,
            dryrun_validator=mock_dryrun,
            alignment_validator=mock_alignment,
            query_naming_strategy="hybrid"
        )
        
        description = "Uses window functions for ranking"
        sql = "SELECT ROW_NUMBER() OVER (ORDER BY score) FROM results"
        insight = ""  # Empty insight
        
        name = refiner._generate_descriptive_query_name(
            description=description,
            sql=sql,
            insight=insight,
            query_index=0,
            naming_strategy="hybrid"
        )
        
        # Should use rule-based without calling LLM
        assert "window" in name
        mock_gemini.client.generate_content.assert_not_called()


class TestQueryNamingIntegration:
    """Test integration of query naming with build_query_result."""

    def test_query_name_with_target_table(self, mocker: "MockerFixture") -> None:
        """Test query naming with target table prefix."""
        mock_gemini = mocker.Mock()
        mock_dryrun = mocker.Mock()
        mock_alignment = mocker.Mock()
        
        refiner = QueryRefiner(
            gemini_client=mock_gemini,
            dryrun_validator=mock_dryrun,
            alignment_validator=mock_alignment,
            query_naming_strategy="rule_based"
        )
        
        # Create mock history
        history = QueryValidationHistory(
            candidate_id="test",
            initial_sql="SELECT * FROM table1 INNER JOIN table2",
            insight="Join data for analysis"
        )
        history.final_valid = True
        history.final_alignment_score = 0.95
        
        # Mock iteration state
        mock_iter = mocker.Mock()
        mock_iter.syntax_passed = True
        mock_iter.dryrun_passed = True
        mock_iter.execution_passed = True
        mock_iter.alignment_passed = True
        mock_iter.execution_stats = {"total_bytes_processed": 1000}
        mock_iter.sample_results = [{"col1": "val1"}]
        mock_iter.result_schema = []
        history.iterations = [mock_iter]
        
        result = refiner._build_query_result(
            sql="SELECT * FROM table1 INNER JOIN table2",
            description="Simple INNER JOIN approach",
            history=history,
            total_time_ms=1000.0,
            source_tables=["project.dataset.table1"],
            target_table_name="performance_analysis",
            query_index=0
        )
        
        # Should have target table prefix
        assert result.query_name.startswith("performance_analysis_")
        assert "inner_join" in result.query_name or "simple" in result.query_name

    def test_query_name_without_target_table(self, mocker: "MockerFixture") -> None:
        """Test query naming without target table prefix."""
        mock_gemini = mocker.Mock()
        mock_dryrun = mocker.Mock()
        mock_alignment = mocker.Mock()
        
        refiner = QueryRefiner(
            gemini_client=mock_gemini,
            dryrun_validator=mock_dryrun,
            alignment_validator=mock_alignment,
            query_naming_strategy="rule_based"
        )
        
        # Create mock history
        history = QueryValidationHistory(
            candidate_id="test",
            initial_sql="WITH data AS (SELECT * FROM table1) SELECT * FROM data",
            insight="Use CTE for modularity"
        )
        history.final_valid = True
        history.final_alignment_score = 0.90
        
        # Mock iteration state
        mock_iter = mocker.Mock()
        mock_iter.syntax_passed = True
        mock_iter.dryrun_passed = True
        mock_iter.execution_passed = True
        mock_iter.alignment_passed = True
        mock_iter.execution_stats = {"total_bytes_processed": 1000}
        mock_iter.sample_results = [{"col1": "val1"}]
        mock_iter.result_schema = []
        history.iterations = [mock_iter]
        
        result = refiner._build_query_result(
            sql="WITH data AS (SELECT * FROM table1) SELECT * FROM data",
            description="Modular approach using CTE",
            history=history,
            total_time_ms=1000.0,
            source_tables=["project.dataset.table1"],
            target_table_name=None,
            query_index=1
        )
        
        # Should not have prefix, just the descriptive name
        assert "cte" in result.query_name or "modular" in result.query_name
        assert not result.query_name.startswith("performance_")

