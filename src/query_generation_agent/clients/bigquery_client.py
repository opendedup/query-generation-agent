"""
BigQuery Client for Query Validation

Handles BigQuery dry-run validation and sample query execution.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError

logger = logging.getLogger(__name__)


class BigQueryClient:
    """
    Client for BigQuery query validation and execution.
    
    Provides methods for:
    - Dry-run validation (syntax and schema check without execution)
    - Limited execution for sample results
    - Query cost estimation
    """
    
    def __init__(
        self,
        project_id: str,
        location: str = "US",
        timeout_seconds: int = 120
    ):
        """
        Initialize BigQuery client.
        
        Args:
            project_id: GCP project ID for query execution
            location: BigQuery location (default: US)
            timeout_seconds: Query timeout in seconds
        """
        self.project_id = project_id
        self.location = location
        self.timeout_seconds = timeout_seconds
        
        # Initialize BigQuery client
        self.client = bigquery.Client(project=project_id)
        
        logger.info(f"BigQuery client initialized for project: {project_id}")
    
    def validate_syntax_and_schema(self, sql: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Validate SQL syntax and schema using BigQuery dry-run.
        
        Args:
            sql: SQL query to validate
            
        Returns:
            Tuple of (is_valid, error_message, stats)
        """
        try:
            job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
            
            logger.debug(f"Running dry-run validation for query")
            query_job = self.client.query(sql, job_config=job_config)
            
            # Dry-run succeeded
            stats = {
                "total_bytes_processed": query_job.total_bytes_processed,
                "total_bytes_billed": query_job.total_bytes_billed or 0,
                "estimated_cost_usd": self._estimate_cost(query_job.total_bytes_processed),
                "schema_valid": True
            }
            
            logger.info(f"Dry-run validation passed. Estimated bytes: {stats['total_bytes_processed']:,}")
            return True, None, stats
            
        except GoogleCloudError as e:
            error_msg = str(e)
            logger.warning(f"Dry-run validation failed: {error_msg}")
            return False, error_msg, None
            
        except Exception as e:
            error_msg = f"Unexpected error during dry-run: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg, None
    
    def execute_with_limit(
        self,
        sql: str,
        limit: int = 10
    ) -> Tuple[bool, Optional[str], Optional[List[Dict[str, Any]]], Optional[List[Dict[str, str]]]]:
        """
        Execute query with LIMIT to get sample results.
        
        Args:
            sql: SQL query to execute
            limit: Maximum number of rows to return
            
        Returns:
            Tuple of (success, error_message, sample_rows, schema)
        """
        try:
            # Wrap query with LIMIT if not already present
            limited_sql = self._add_limit_if_needed(sql, limit)
            
            logger.debug(f"Executing query with limit {limit}")
            
            job_config = bigquery.QueryJobConfig(
                use_query_cache=False,
                maximum_bytes_billed=10 * 1024 * 1024 * 1024  # 10 GB limit for safety
            )
            
            query_job = self.client.query(limited_sql, job_config=job_config, timeout=self.timeout_seconds)
            
            # Wait for query to complete
            results = query_job.result()
            
            # Extract sample rows
            sample_rows = []
            for row in results:
                sample_rows.append(dict(row.items()))
            
            # Extract schema
            schema = []
            if query_job.schema:
                for field in query_job.schema:
                    schema.append({
                        "name": field.name,
                        "type": field.field_type,
                        "mode": field.mode
                    })
            
            logger.info(f"Query executed successfully. Returned {len(sample_rows)} rows")
            return True, None, sample_rows, schema
            
        except GoogleCloudError as e:
            error_msg = str(e)
            logger.warning(f"Query execution failed: {error_msg}")
            return False, error_msg, None, None
            
        except Exception as e:
            error_msg = f"Unexpected error during execution: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg, None, None
    
    def get_execution_stats(self, sql: str) -> Optional[Dict[str, Any]]:
        """
        Get execution statistics without actually running the query.
        
        Uses dry-run to estimate costs and bytes processed.
        
        Args:
            sql: SQL query to analyze
            
        Returns:
            Dictionary with execution stats or None if failed
        """
        is_valid, _, stats = self.validate_syntax_and_schema(sql)
        return stats if is_valid else None
    
    def _add_limit_if_needed(self, sql: str, limit: int) -> str:
        """
        Add LIMIT clause if not already present.
        
        Args:
            sql: Original SQL query
            limit: Limit value to add
            
        Returns:
            SQL with LIMIT clause
        """
        # Simple check - look for LIMIT in query (case-insensitive)
        sql_upper = sql.upper().strip()
        
        if "LIMIT" in sql_upper:
            # Already has LIMIT
            return sql
        
        # Add LIMIT
        return f"{sql.rstrip(';')} LIMIT {limit}"
    
    def _estimate_cost(self, bytes_processed: int) -> float:
        """
        Estimate query cost based on bytes processed.
        
        BigQuery charges $6.25 per TB processed (on-demand pricing).
        
        Args:
            bytes_processed: Number of bytes that will be processed
            
        Returns:
            Estimated cost in USD
        """
        if not bytes_processed:
            return 0.0
        
        # Convert bytes to TB
        tb_processed = bytes_processed / (1024 ** 4)
        
        # On-demand pricing: $6.25 per TB
        cost = tb_processed * 6.25
        
        return cost
    
    def validate_table_exists(self, project_id: str, dataset_id: str, table_id: str) -> bool:
        """
        Check if a table exists.
        
        Args:
            project_id: GCP project ID
            dataset_id: Dataset ID
            table_id: Table ID
            
        Returns:
            True if table exists
        """
        try:
            table_ref = f"{project_id}.{dataset_id}.{table_id}"
            self.client.get_table(table_ref)
            return True
        except Exception:
            return False
    
    def get_table_schema(self, project_id: str, dataset_id: str, table_id: str) -> Optional[List[Dict[str, str]]]:
        """
        Get table schema.
        
        Args:
            project_id: GCP project ID
            dataset_id: Dataset ID
            table_id: Table ID
            
        Returns:
            List of field dictionaries or None if table not found
        """
        try:
            table_ref = f"{project_id}.{dataset_id}.{table_id}"
            table = self.client.get_table(table_ref)
            
            schema = []
            for field in table.schema:
                schema.append({
                    "name": field.name,
                    "type": field.field_type,
                    "mode": field.mode,
                    "description": field.description or ""
                })
            
            return schema
            
        except Exception as e:
            logger.warning(f"Failed to get table schema: {e}")
            return None

