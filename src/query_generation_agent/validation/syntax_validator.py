"""
SQL Syntax Validator

Basic SQL syntax validation using sqlparse.
"""

import logging
import re
from typing import Optional, Tuple

import sqlparse
from sqlparse import tokens as T

logger = logging.getLogger(__name__)


class SyntaxValidator:
    """
    Validates SQL syntax using sqlparse.
    
    Performs basic syntax checks and common error detection.
    """
    
    def __init__(self):
        """Initialize syntax validator."""
        pass
    
    def validate(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SQL syntax.
        
        Args:
            sql: SQL query to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not sql or not sql.strip():
            return False, "SQL query is empty"
        
        try:
            # Parse the SQL
            parsed = sqlparse.parse(sql)
            
            if not parsed:
                return False, "Failed to parse SQL query"
            
            # Check for multiple statements (should be single query)
            if len(parsed) > 1:
                return False, "Multiple SQL statements detected. Please provide a single query."
            
            statement = parsed[0]
            
            # Check if it's a SELECT statement (we only generate SELECT queries)
            if not self._is_select_statement(statement):
                return False, "Query must be a SELECT statement"
            
            # Check for common syntax errors
            error = self._check_common_errors(sql, statement)
            if error:
                return False, error
            
            logger.debug("SQL syntax validation passed")
            return True, None
            
        except Exception as e:
            error_msg = f"Syntax validation error: {str(e)}"
            logger.warning(error_msg)
            return False, error_msg
    
    def _is_select_statement(self, statement: sqlparse.sql.Statement) -> bool:
        """
        Check if statement is a SELECT query.
        
        Args:
            statement: Parsed SQL statement
            
        Returns:
            True if SELECT statement
        """
        # Get first token (ignoring whitespace)
        first_token = statement.token_first(skip_ws=True, skip_cm=True)
        
        if first_token and first_token.ttype is T.Keyword.DML:
            return first_token.value.upper() == "SELECT"
        
        return False
    
    def _check_common_errors(self, sql: str, statement: sqlparse.sql.Statement) -> Optional[str]:
        """
        Check for common SQL errors.
        
        Args:
            sql: Raw SQL string
            statement: Parsed statement
            
        Returns:
            Error message if found, None otherwise
        """
        sql_upper = sql.upper()
        
        # Check for SELECT without FROM (except for simple literals)
        if "SELECT" in sql_upper and "FROM" not in sql_upper:
            # Allow simple SELECT statements like "SELECT 1"
            if not re.search(r'SELECT\s+[\d\'"]+', sql_upper):
                return "SELECT query missing FROM clause"
        
        # Check for unmatched parentheses
        if sql.count("(") != sql.count(")"):
            return "Unmatched parentheses in query"
        
        # Check for unmatched quotes
        # Simple check: count single and double quotes
        single_quotes = sql.count("'") - sql.count("\\'")
        double_quotes = sql.count('"') - sql.count('\\"')
        
        if single_quotes % 2 != 0:
            return "Unmatched single quotes in query"
        
        if double_quotes % 2 != 0:
            return "Unmatched double quotes in query"
        
        # Check for common keyword misspellings or issues
        if " FORM " in sql_upper:
            return "Possible typo: 'FORM' should be 'FROM'"
        
        if " SELCT " in sql_upper or " SLECT " in sql_upper:
            return "Possible typo: 'SELECT' is misspelled"
        
        # Check for GROUP BY without SELECT aggregation
        if "GROUP BY" in sql_upper and "FROM" in sql_upper:
            # This is a simplified check - ideally would parse more carefully
            has_aggregation = any(
                agg in sql_upper
                for agg in ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN(", "ARRAY_AGG("]
            )
            if not has_aggregation:
                # Could be valid if all non-grouped columns are in GROUP BY
                # We'll let BigQuery validate this properly
                pass
        
        # Check for incomplete WHERE clause
        if " WHERE " in sql_upper:
            where_pos = sql_upper.find(" WHERE ")
            after_where = sql_upper[where_pos + 7:].strip()
            
            # Check if WHERE is followed immediately by keywords that shouldn't be there
            if any(after_where.startswith(kw) for kw in ["ORDER BY", "GROUP BY", "LIMIT", "UNION"]):
                return "Incomplete WHERE clause"
        
        # Check for JOIN without ON
        join_keywords = ["JOIN", "LEFT JOIN", "RIGHT JOIN", "INNER JOIN", "OUTER JOIN", "FULL JOIN"]
        for join_kw in join_keywords:
            if f" {join_kw} " in sql_upper and " ON " not in sql_upper and " USING " not in sql_upper:
                # Could be CROSS JOIN which doesn't need ON
                if "CROSS" not in sql_upper:
                    return f"{join_kw} clause missing ON or USING condition"
        
        return None
    
    def format_sql(self, sql: str) -> str:
        """
        Format SQL for better readability.
        
        Args:
            sql: SQL query to format
            
        Returns:
            Formatted SQL
        """
        try:
            formatted = sqlparse.format(
                sql,
                reindent=True,
                keyword_case="upper",
                indent_width=2
            )
            return formatted
        except Exception as e:
            logger.warning(f"Failed to format SQL: {e}")
            return sql
    
    def extract_table_references(self, sql: str) -> list[str]:
        """
        Extract table references from SQL query.
        
        Args:
            sql: SQL query
            
        Returns:
            List of table references (project.dataset.table or dataset.table)
        """
        tables = []
        
        try:
            parsed = sqlparse.parse(sql)
            if not parsed:
                return tables
            
            statement = parsed[0]
            
            # Look for table names after FROM and JOIN
            tokens = list(statement.flatten())
            
            for i, token in enumerate(tokens):
                if token.ttype is T.Keyword and token.value.upper() in ["FROM", "JOIN"]:
                    # Next non-whitespace token should be the table name
                    for j in range(i + 1, len(tokens)):
                        if tokens[j].ttype not in [T.Whitespace, T.Newline]:
                            if tokens[j].ttype in [T.Name, None]:
                                table_name = tokens[j].value.strip("`'\"")
                                if table_name and not table_name.upper() in ["AS", "ON", "USING"]:
                                    tables.append(table_name)
                            break
            
            return tables
            
        except Exception as e:
            logger.warning(f"Failed to extract table references: {e}")
            return tables

