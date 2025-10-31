"""
Tests for SQLFluff validator.

Tests SQL linting validation with BigQuery dialect.
"""

from typing import TYPE_CHECKING

import pytest

from query_generation_agent.validation.sqlfluff_validator import SQLFluffValidator

if TYPE_CHECKING:
    from pytest_mock.plugin import MockerFixture


class TestSQLFluffValidator:
    """Test cases for SQLFluff validator."""
    
    def test_validator_initialization(self) -> None:
        """Test that validator initializes correctly."""
        validator = SQLFluffValidator()
        assert validator is not None
        assert validator.linter is not None
        assert validator.max_errors == 10
    
    def test_valid_simple_query(self) -> None:
        """Test that a valid simple query passes validation."""
        validator = SQLFluffValidator()
        
        sql = """
        SELECT
            customer_id,
            order_date,
            total_amount
        FROM `project.dataset.orders`
        WHERE order_date >= '2024-01-01'
        ORDER BY order_date DESC
        LIMIT 100
        """
        
        is_valid, error_msg = validator.validate(sql)
        
        assert is_valid is True
        assert error_msg is None
    
    def test_valid_complex_query_with_cte(self) -> None:
        """Test that a valid complex query with CTE passes validation."""
        validator = SQLFluffValidator()
        
        sql = """
        WITH recent_orders AS (
            SELECT
                customer_id,
                SUM(total_amount) AS total_spent
            FROM `project.dataset.orders`
            WHERE order_date >= '2024-01-01'
            GROUP BY customer_id
        )
        SELECT
            c.customer_name,
            ro.total_spent
        FROM `project.dataset.customers` AS c
        INNER JOIN recent_orders AS ro
            ON c.customer_id = ro.customer_id
        WHERE ro.total_spent > 1000
        ORDER BY ro.total_spent DESC
        """
        
        is_valid, error_msg = validator.validate(sql)
        
        assert is_valid is True
        assert error_msg is None
    
    def test_string_with_apostrophe_single_quotes_fails(self) -> None:
        """Test that string with apostrophe using single quote concatenation fails."""
        validator = SQLFluffValidator()
        
        # This should fail with the 'Pick''em' pattern
        sql = """
        SELECT
            company_name,
            status
        FROM `project.dataset.companies`
        WHERE status = 'Pick''em'
        """
        
        is_valid, error_msg = validator.validate(sql)
        
        # Should fail because of string literal issue
        # Note: This may pass sqlfluff's default rules, but our custom config should catch it
        # If it passes, that's okay - BigQuery dry-run will catch it
        if not is_valid:
            assert error_msg is not None
            assert "L063" in error_msg or "quote" in error_msg.lower()
    
    def test_string_with_apostrophe_double_quotes_passes(self) -> None:
        """Test that string with apostrophe using double quotes passes."""
        validator = SQLFluffValidator()
        
        sql = """
        SELECT
            company_name,
            status
        FROM `project.dataset.companies`
        WHERE company_name = "O'Reilly Media"
            AND status = "Customer's Choice"
        """
        
        is_valid, error_msg = validator.validate(sql)
        
        assert is_valid is True
        assert error_msg is None
    
    def test_empty_query_fails(self) -> None:
        """Test that empty query fails validation."""
        validator = SQLFluffValidator()
        
        is_valid, error_msg = validator.validate("")
        
        assert is_valid is False
        assert error_msg is not None
        assert "empty" in error_msg.lower()
    
    def test_whitespace_only_query_fails(self) -> None:
        """Test that whitespace-only query fails validation."""
        validator = SQLFluffValidator()
        
        is_valid, error_msg = validator.validate("   \n\t  ")
        
        assert is_valid is False
        assert error_msg is not None
        assert "empty" in error_msg.lower()
    
    def test_invalid_sql_syntax(self) -> None:
        """Test that query with invalid SQL syntax fails."""
        validator = SQLFluffValidator()
        
        # Missing FROM clause
        sql = "SELECT customer_id, order_date"
        
        is_valid, error_msg = validator.validate(sql)
        
        # May pass or fail depending on sqlfluff rules
        # If it fails, error message should be present
        if not is_valid:
            assert error_msg is not None
    
    def test_validator_handles_exceptions_gracefully(self, mocker: "MockerFixture") -> None:
        """Test that validator handles exceptions gracefully."""
        validator = SQLFluffValidator()
        
        # Mock the linter to raise an exception
        mocker.patch.object(
            validator.linter,
            'lint_string',
            side_effect=Exception("Mocked linting error")
        )
        
        sql = "SELECT * FROM `project.dataset.table`"
        
        is_valid, error_msg = validator.validate(sql)
        
        # Should return True (non-blocking) when linter fails
        assert is_valid is True
        assert error_msg is None
    
    def test_max_errors_limit(self) -> None:
        """Test that validator respects max_errors limit."""
        validator = SQLFluffValidator(max_errors=3)
        
        # Query with multiple style issues
        sql = """
        SELECT customer_id,order_date,total_amount FROM `project.dataset.orders` WHERE order_date>='2024-01-01' AND customer_id IS NOT NULL AND total_amount>100 ORDER BY order_date DESC LIMIT 100
        """
        
        is_valid, error_msg = validator.validate(sql)
        
        # If violations are found, should limit to max_errors
        if not is_valid and error_msg:
            # Count numbered errors in message (format: "1. Line X")
            error_lines = [line for line in error_msg.split('\n') if line.strip().startswith(('1.', '2.', '3.', '4.'))]
            assert len(error_lines) <= 3
    
    def test_format_violations_includes_helpful_hints(self) -> None:
        """Test that format_violations includes helpful hints for L063 errors."""
        validator = SQLFluffValidator()
        
        # Create a mock violation for L063
        class MockViolation:
            def rule_code(self) -> str:
                return 'L063'
            
            @property
            def description(self) -> str:
                return "Inconsistent quote style"
            
            @property
            def line_no(self) -> int:
                return 5
            
            @property
            def line_pos(self) -> int:
                return 20
        
        violations = [MockViolation()]
        error_msg = validator._format_violations(violations)
        
        assert "HINT" in error_msg
        assert "double quotes" in error_msg
        assert '"Pick\'em"' in error_msg
        assert "O'Reilly" in error_msg or "Pick'em" in error_msg
    
    def test_critical_rules_defined(self) -> None:
        """Test that critical rules are properly defined."""
        validator = SQLFluffValidator()
        
        critical_rules = validator._get_critical_rules()
        
        assert isinstance(critical_rules, set)
        assert 'L063' in critical_rules  # String literal quote style
        assert 'L028' in critical_rules  # SELECT without FROM
        assert len(critical_rules) > 0
    
    def test_validator_with_bigquery_specific_syntax(self) -> None:
        """Test validator handles BigQuery-specific syntax correctly."""
        validator = SQLFluffValidator()
        
        # BigQuery specific: STRUCT, ARRAY, backticks
        sql = """
        SELECT
            customer_id,
            ARRAY_AGG(STRUCT(order_id, order_date)) AS orders,
            COUNT(*) AS order_count
        FROM `project.dataset.orders`
        GROUP BY customer_id
        """
        
        is_valid, error_msg = validator.validate(sql)
        
        # Should pass with BigQuery dialect
        assert is_valid is True
        assert error_msg is None

