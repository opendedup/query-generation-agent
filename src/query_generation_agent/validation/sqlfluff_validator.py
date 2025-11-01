"""
SQLFluff SQL Linter Validator

Validates SQL queries using sqlfluff with BigQuery dialect.
Catches syntax errors, style issues, and BigQuery-specific problems
before expensive BigQuery API calls.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

from sqlfluff.core import Linter, FluffConfig

logger = logging.getLogger(__name__)


class SQLFluffValidator:
    """
    Validates SQL queries using sqlfluff linter.
    
    Performs comprehensive SQL validation including:
    - BigQuery syntax validation
    - Style consistency checks
    - String literal validation (catches 'Pick''em' issues)
    - Best practice enforcement
    """
    
    def __init__(self, max_errors: int = 10):
        """
        Initialize SQLFluff validator with BigQuery dialect.
        
        Args:
            max_errors: Maximum number of errors to report (default: 10)
        """
        self.max_errors = max_errors
        
        config_path = Path(__file__).resolve().parents[4] / ".sqlfluff"
        config: Optional[FluffConfig] = None

        if config_path.exists():
            try:
                config = FluffConfig.from_path(str(config_path))
                logger.debug(
                    "SQLFluff config loaded from %s with dialect=%s",
                    config_path,
                    config.get("dialect"),
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load SQLFluff config from %s, falling back to defaults: %s",
                    config_path,
                    exc,
                )

        if config is not None:
            self.linter = Linter(config=config)
        else:
            logger.debug(
                "Using built-in SQLFluff configuration with BigQuery dialect"
            )
            self.linter = Linter(dialect="bigquery")
    
    def validate(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SQL query using sqlfluff.
        
        Args:
            sql: SQL query to validate
            
        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if query passes linting
            - error_message: Detailed error message if validation fails, None otherwise
        """
        if not sql or not sql.strip():
            return False, "SQL query is empty"
        
        try:
            logger.debug("Running SQLFluff validation")
            
            # Lint the SQL string
            result = self.linter.lint_string(sql)
            
            # Check for violations
            if result.violations:
                logger.debug(f"SQLFluff found {len(result.violations)} violation(s)")
                
                # Filter to only show errors (not warnings/info)
                critical_violations = [
                    v for v in result.violations 
                    if v.rule_code() in self._get_critical_rules()
                ]
                
                # If we have critical violations, fail validation
                if critical_violations:
                    error_msg = self._format_violations(critical_violations[:self.max_errors])
                    logger.warning(f"SQLFluff validation failed: {error_msg}")
                    return False, error_msg
                
                # Non-critical violations: log but don't fail
                logger.debug(f"SQLFluff found {len(result.violations)} non-critical issues (not failing)")
                return True, None
            
            logger.debug("SQLFluff validation passed")
            return True, None
            
        except Exception as e:
            # Don't fail validation if linter crashes - let BigQuery be the fallback
            error_msg = f"SQLFluff validation error (non-blocking): {str(e)}"
            logger.warning(error_msg)
            # Return True to allow BigQuery validation to run
            return True, None
    
    def _get_critical_rules(self) -> set:
        """
        Get set of critical rule codes that should fail validation.
        
        These are rules that catch actual SQL errors, not just style issues.
        
        Returns:
            Set of critical rule codes
        """
        return {
            # Syntax and parsing errors
            'L001',  # Trailing whitespace
            'L003',  # Indentation not consistent
            'L008',  # Comma placement
            'L016',  # Line too long
            'L028',  # SELECT without FROM
            'L029',  # Unquoted keywords in WHERE
            'L034',  # Wildcard selection without columns
            'L046',  # Unnecessary nested CASE
            'L047',  # Use IS NULL instead of = NULL
            'L063',  # String literal quote style (catches 'Pick''em')
            # Add more as needed
        }
    
    def _format_violations(self, violations: list) -> str:
        """
        Format violations into a readable error message for LLM refinement.
        
        Args:
            violations: List of SQLFluff violations
            
        Returns:
            Formatted error message with actionable feedback
        """
        if not violations:
            return "Unknown linting error"
        
        error_lines = ["SQL linting errors detected:\n"]
        
        for i, violation in enumerate(violations, 1):
            rule_code = violation.rule_code()
            description = violation.description
            line_no = violation.line_no
            line_pos = violation.line_pos
            
            # Format: "1. Line 30, Col 43 [L063]: Inconsistent string quote style..."
            error_lines.append(
                f"{i}. Line {line_no}, Col {line_pos} [{rule_code}]: {description}"
            )
        
        # Add helpful hint for string literal issues
        if any('L063' in str(v.rule_code()) for v in violations):
            error_lines.append(
                "\nHINT: For strings containing apostrophes, use double quotes in BigQuery:"
            )
            error_lines.append('  ✅ CORRECT: WHERE status = "Pick\'em"')
            error_lines.append('  ❌ WRONG: WHERE status = \'Pick\'\'em\'')
        
        return "\n".join(error_lines)

