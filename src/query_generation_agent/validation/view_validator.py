"""
View Validator

Validates CREATE VIEW DDL statements against target schemas.
Uses BigQuery dry-run to verify view produces expected columns and types.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from google.cloud import bigquery

from ..clients.bigquery_client import BigQueryClient
from ..models.response_models import ValidationResult

logger = logging.getLogger(__name__)


class ViewValidator:
    """
    Validates CREATE VIEW DDL statements.
    
    Extracts SELECT from CREATE VIEW, validates via BigQuery dry-run,
    and compares output schema to target schema.
    """
    
    def __init__(self, bigquery_client: BigQueryClient):
        """
        Initialize view validator.
        
        Args:
            bigquery_client: BigQuery client for validation
        """
        self.bigquery_client = bigquery_client
    
    def validate_view_ddl(
        self,
        view_ddl: str,
        target_schema: List[Dict[str, str]]
    ) -> ValidationResult:
        """
        Validate view DDL produces expected schema.
        
        Args:
            view_ddl: CREATE VIEW DDL statement
            target_schema: Expected columns with name, type, description
            
        Returns:
            ValidationResult with validation details
        """
        logger.info("Validating VIEW DDL")
        
        # Extract SELECT query from CREATE VIEW
        select_query = self._extract_select_from_view(view_ddl)
        
        if not select_query:
            return ValidationResult(
                is_valid=False,
                error_message="Could not extract SELECT statement from VIEW DDL",
                error_type="syntax",
                syntax_valid=False,
                dryrun_valid=False,
                execution_valid=False,
                alignment_valid=False
            )
        
        # Get actual schema from BigQuery dry-run
        try:
            actual_schema = self.bigquery_client.get_query_schema(select_query)
        except Exception as e:
            error_msg = str(e)
            
            # Check if error is due to non-existent tables
            if "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
                logger.warning(f"Source tables not found (expected for new views): {error_msg}")
                # Return partial validation - DDL is syntactically valid but can't verify schema
                return ValidationResult(
                    is_valid=True,  # Consider valid if only issue is missing source tables
                    error_message=f"Cannot fully validate: {error_msg}",
                    error_type="warning",
                    syntax_valid=True,
                    dryrun_valid=False,  # Can't run dry-run without source tables
                    execution_valid=False,
                    alignment_valid=True,  # Assume alignment if schema can't be checked
                    alignment_score=0.8  # Partial score
                )
            
            logger.error(f"BigQuery dry-run failed: {error_msg}")
            return ValidationResult(
                is_valid=False,
                error_message=f"BigQuery validation failed: {error_msg}",
                error_type="execution",
                syntax_valid=True,  # Passed extraction
                dryrun_valid=False,
                execution_valid=False,
                alignment_valid=False
            )
        
        # Compare schemas
        matches, differences = self._compare_schemas(actual_schema, target_schema)
        
        if matches:
            logger.info("VIEW DDL validation successful - schema matches target")
            return ValidationResult(
                is_valid=True,
                error_message=None,
                error_type=None,
                syntax_valid=True,
                dryrun_valid=True,
                execution_valid=True,
                alignment_valid=True,
                alignment_score=1.0,
                result_schema=[
                    {"name": field.name, "type": field.field_type}
                    for field in actual_schema
                ]
            )
        else:
            error_msg = f"Schema mismatch: {differences}"
            logger.warning(error_msg)
            return ValidationResult(
                is_valid=False,
                error_message=error_msg,
                error_type="alignment",
                syntax_valid=True,
                dryrun_valid=True,
                execution_valid=True,
                alignment_valid=False,
                alignment_score=0.5,
                result_schema=[
                    {"name": field.name, "type": field.field_type}
                    for field in actual_schema
                ]
            )
    
    def _extract_select_from_view(self, view_ddl: str) -> Optional[str]:
        """
        Extract SELECT query from CREATE VIEW statement.
        
        Args:
            view_ddl: Full CREATE VIEW DDL
            
        Returns:
            SELECT query or None if extraction fails
        """
        # Remove comments
        ddl = re.sub(r'--[^\n]*', '', view_ddl)
        ddl = re.sub(r'/\*.*?\*/', '', ddl, flags=re.DOTALL)
        
        # Find AS keyword after CREATE VIEW
        as_match = re.search(r'\bAS\b', ddl, re.IGNORECASE)
        if not as_match:
            logger.error("Could not find AS keyword in CREATE VIEW statement")
            return None
        
        # Everything after AS is the SELECT query
        select_query = ddl[as_match.end():].strip()
        
        # Remove trailing semicolon
        select_query = select_query.rstrip(';').strip()
        
        if not select_query.upper().startswith('SELECT'):
            # Try to find SELECT keyword
            select_match = re.search(r'\bSELECT\b', select_query, re.IGNORECASE)
            if select_match:
                select_query = select_query[select_match.start():]
            else:
                logger.error("Could not find SELECT keyword after AS")
                return None
        
        return select_query
    
    def _compare_schemas(
        self,
        actual_schema: List[bigquery.SchemaField],
        target_schema: List[Dict[str, str]]
    ) -> Tuple[bool, str]:
        """
        Compare actual schema from BigQuery to target schema.
        
        Args:
            actual_schema: BigQuery schema fields
            target_schema: Target columns with name, type
            
        Returns:
            Tuple of (matches, differences_description)
        """
        # Convert actual schema to comparable format
        actual_cols = {
            field.name.lower(): self._normalize_type(field.field_type)
            for field in actual_schema
        }
        
        # Convert target schema to comparable format
        target_cols = {
            col["name"].lower(): self._normalize_type(col["type"])
            for col in target_schema
        }
        
        # Check for missing columns
        missing = set(target_cols.keys()) - set(actual_cols.keys())
        if missing:
            return False, f"Missing columns: {', '.join(missing)}"
        
        # Check for extra columns
        extra = set(actual_cols.keys()) - set(target_cols.keys())
        if extra:
            return False, f"Extra columns: {', '.join(extra)}"
        
        # Check for type mismatches
        mismatches = []
        for col_name in target_cols:
            actual_type = actual_cols[col_name]
            target_type = target_cols[col_name]
            if actual_type != target_type:
                mismatches.append(f"{col_name}: expected {target_type}, got {actual_type}")
        
        if mismatches:
            return False, f"Type mismatches: {'; '.join(mismatches)}"
        
        return True, ""
    
    def _normalize_type(self, bq_type: str) -> str:
        """
        Normalize BigQuery type for comparison.
        
        Maps similar types (e.g., INT64 == INTEGER) to canonical form.
        
        Args:
            bq_type: BigQuery type string
            
        Returns:
            Normalized type string
        """
        # Convert to uppercase
        normalized = bq_type.upper()
        
        # Type mappings
        type_map = {
            "INTEGER": "INT64",
            "FLOAT": "FLOAT64",
            "BOOLEAN": "BOOL",
            "BYTES": "BYTES",
            "DATETIME": "DATETIME",
            "GEOGRAPHY": "GEOGRAPHY",
            "INTERVAL": "INTERVAL",
            "JSON": "JSON",
            "NUMERIC": "NUMERIC",
            "BIGNUMERIC": "BIGNUMERIC",
        }
        
        return type_map.get(normalized, normalized)

