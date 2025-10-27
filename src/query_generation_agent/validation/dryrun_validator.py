"""
BigQuery Dry-Run Validator

Validates queries using BigQuery dry-run and sample execution.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from ..clients.bigquery_client import BigQueryClient

logger = logging.getLogger(__name__)


class DryRunValidator:
    """
    Validates queries using BigQuery dry-run and limited execution.
    
    Performs:
    1. Dry-run validation (syntax and schema check)
    2. Sample execution (with LIMIT) to get actual results
    """
    
    def __init__(self, bigquery_client: BigQueryClient, max_sample_rows: int = 10):
        """
        Initialize dry-run validator.
        
        Args:
            bigquery_client: BigQuery client instance
            max_sample_rows: Maximum rows to fetch for sample execution
        """
        self.bigquery_client = bigquery_client
        self.max_sample_rows = max_sample_rows
    
    def validate_dryrun(self, sql: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Validate query using BigQuery dry-run.
        
        Args:
            sql: SQL query to validate
            
        Returns:
            Tuple of (is_valid, error_message, execution_stats)
        """
        logger.debug("Running BigQuery dry-run validation")
        
        is_valid, error_msg, stats = self.bigquery_client.validate_syntax_and_schema(sql)
        
        if is_valid:
            logger.info("Dry-run validation passed")
        else:
            logger.warning(f"Dry-run validation failed: {error_msg}")
        
        return is_valid, error_msg, stats
    
    def execute_sample(
        self,
        sql: str,
        source_tables: Optional[List[str]] = None
    ) -> Tuple[bool, Optional[str], Optional[List[Dict[str, Any]]], Optional[List[Dict[str, str]]]]:
        """
        Execute query with LIMIT to get sample results.
        
        Args:
            sql: SQL query to execute
            source_tables: Optional list of source table IDs
            
        Returns:
            Tuple of (success, error_message, sample_rows, result_schema)
        """
        logger.debug(f"Executing query with LIMIT {self.max_sample_rows}")
        
        success, error_msg, sample_rows, schema = self.bigquery_client.execute_with_limit(
            sql,
            limit=self.max_sample_rows,
            source_tables=source_tables
        )
        
        if success:
            row_count = len(sample_rows) if sample_rows else 0
            logger.info(f"Sample execution successful. Retrieved {row_count} rows")
        else:
            logger.warning(f"Sample execution failed: {error_msg}")
        
        return success, error_msg, sample_rows, schema
    
    def validate_full(
        self,
        sql: str
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Perform full validation: dry-run + sample execution.
        
        Args:
            sql: SQL query to validate
            
        Returns:
            Tuple of (is_valid, error_message, validation_results)
        """
        validation_results = {
            "dryrun_valid": False,
            "execution_valid": False,
            "execution_stats": None,
            "sample_results": None,
            "result_schema": None
        }
        
        # Step 1: Dry-run validation
        dryrun_valid, dryrun_error, stats = self.validate_dryrun(sql)
        validation_results["dryrun_valid"] = dryrun_valid
        validation_results["execution_stats"] = stats
        
        if not dryrun_valid:
            return False, f"Dry-run failed: {dryrun_error}", validation_results
        
        # Step 2: Sample execution
        exec_success, exec_error, sample_rows, schema = self.execute_sample(sql)
        validation_results["execution_valid"] = exec_success
        validation_results["sample_results"] = sample_rows
        validation_results["result_schema"] = schema
        
        if not exec_success:
            return False, f"Execution failed: {exec_error}", validation_results
        
        # Both validations passed
        logger.info("Full validation passed (dry-run + execution)")
        return True, None, validation_results
    
    def estimate_cost(self, sql: str) -> Optional[float]:
        """
        Estimate query execution cost.
        
        Args:
            sql: SQL query
            
        Returns:
            Estimated cost in USD, or None if estimation failed
        """
        stats = self.bigquery_client.get_execution_stats(sql)
        if stats:
            return stats.get("estimated_cost_usd")
        return None
    
    def check_tables_exist(self, table_refs: List[str]) -> Dict[str, bool]:
        """
        Check if tables exist in BigQuery.
        
        Args:
            table_refs: List of table references (project.dataset.table)
            
        Returns:
            Dictionary mapping table_ref to existence boolean
        """
        results = {}
        
        for table_ref in table_refs:
            parts = table_ref.strip("`'\"").split(".")
            
            if len(parts) == 3:
                project_id, dataset_id, table_id = parts
                exists = self.bigquery_client.validate_table_exists(project_id, dataset_id, table_id)
                results[table_ref] = exists
            else:
                # Invalid format
                results[table_ref] = False
        
        return results

