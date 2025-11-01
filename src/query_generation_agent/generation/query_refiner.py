"""
Query Refiner

Iteratively refines and validates SQL queries until they pass all checks.
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Set

from ..clients.gemini_client import GeminiClient
from ..models.request_models import DatasetMetadata
from ..models.response_models import QueryResult, ValidationResult
from ..models.validation_models import (
    IterationState,
    QueryValidationHistory,
    ValidationStage,
)
from ..validation.alignment_validator import AlignmentValidator
from ..validation.dryrun_validator import DryRunValidator
from ..validation.sqlfluff_validator import SQLFluffValidator

logger = logging.getLogger(__name__)


class QueryRefiner:
    """
    Refines and validates queries through iterative feedback loop.
    
    Pipeline:
    1. Dry-run validation (BigQuery) - validates syntax and semantics
    2. Sample execution
    3. Alignment validation (LLM)
    4. If any step fails, refine and retry using BigQuery error messages
    """
    
    def __init__(
        self,
        gemini_client: GeminiClient,
        dryrun_validator: DryRunValidator,
        alignment_validator: AlignmentValidator,
        max_iterations: int = 10,
        query_naming_strategy: str = "rule_based"
    ):
        """
        Initialize query refiner.
        
        Args:
            gemini_client: Gemini client for refinement
            dryrun_validator: Dry-run validator
            alignment_validator: Alignment validator
            max_iterations: Maximum refinement iterations per query
            query_naming_strategy: Strategy for generating query names (rule_based/llm/hybrid)
        """
        self.gemini_client = gemini_client
        self.sqlfluff_validator = SQLFluffValidator()
        self.dryrun_validator = dryrun_validator
        self.alignment_validator = alignment_validator
        self.max_iterations = max_iterations
        self.query_naming_strategy = query_naming_strategy
    
    def refine_and_validate(
        self,
        candidate_id: str,
        initial_sql: str,
        description: str,
        insight: str,
        datasets: List[DatasetMetadata],
        target_table_name: Optional[str] = None,
        query_index: int = 0,
        llm_mode: str = "fast_llm"
    ) -> QueryResult:
        """
        Refine and validate a query candidate through iterative feedback.
        
        Args:
            candidate_id: Unique identifier for this candidate
            initial_sql: Initial SQL query
            description: Query description
            insight: Original insight
            datasets: Available datasets
            target_table_name: Name of target table for query naming
            query_index: Index of this query for unique naming
            llm_mode: LLM model mode ('fast_llm' or 'detailed_llm')
            
        Returns:
            QueryResult with final validation status
        """
        start_time = time.time()
        
        logger.info(f"Starting refinement for candidate {candidate_id}")
        logger.info(f"Initial SQL:\n{initial_sql}")
        logger.info("=" * 80)
        
        # Extract source tables from datasets
        source_tables = [dataset.get_full_table_id() for dataset in datasets]
        
        # Initialize validation history
        history = QueryValidationHistory(
            candidate_id=candidate_id,
            initial_sql=initial_sql,
            insight=insight
        )
        
        # Convert datasets to dict format
        dataset_dicts = [self._dataset_to_dict(d) for d in datasets]
        
        current_sql = initial_sql
        iteration = 0
        consecutive_zero_rows = 0
        
        # Track token usage for this query
        query_refinement_tokens = 0
        query_alignment_tokens = 0
        query_llm_calls = 0
        
        while iteration < self.max_iterations:
            iteration_start = time.time()
            
            logger.info(f"Iteration {iteration + 1}/{self.max_iterations}")
            logger.info(f"Testing SQL:\n{current_sql}")
            logger.info("-" * 80)
            
            # Create iteration state
            iter_state = IterationState(
                iteration_number=iteration,
                sql=current_sql
            )
            
            # Run validation pipeline
            is_valid, final_score, validation_details, align_usage = self._run_validation_pipeline(
                sql=current_sql,
                insight=insight,
                iter_state=iter_state,
                datasets=datasets,
                llm_mode=llm_mode
            )
            
            # Track alignment token usage
            query_alignment_tokens += align_usage.get("total_tokens", 0)
            query_llm_calls += 1
            
            # Update iteration timing
            iter_state.iteration_time_ms = (time.time() - iteration_start) * 1000
            
            # Add iteration to history
            history.add_iteration(iter_state)
            
            # Check for consecutive zero-row results (fail fast)
            sample_results = validation_details.get("sample_results")
            if sample_results is not None and len(sample_results) == 0:
                consecutive_zero_rows += 1
                if consecutive_zero_rows >= 3:
                    logger.warning(
                        f"Candidate failed 3 consecutive times with 0 rows. "
                        f"This suggests incompatible tables or missing JOIN keys. "
                        f"Abandoning this candidate."
                    )
                    break
            else:
                consecutive_zero_rows = 0
            
            if is_valid:
                # Query passed all validations!
                logger.info(f"Query validated successfully after {iteration + 1} iteration(s)")
                history.final_valid = True
                history.final_sql = current_sql
                history.final_alignment_score = final_score
                
                # Generate field descriptions NOW for the final valid query (only once)
                if iter_state.result_schema:
                    enriched_schema = self._generate_field_descriptions_for_final_query(
                        sql=current_sql,
                        schema=iter_state.result_schema,
                        insight=insight,
                        datasets=datasets,
                        llm_mode=llm_mode
                    )
                    # Update iter_state with enriched schema
                    iter_state.result_schema = enriched_schema
                    logger.info("Field descriptions added to final query schema")
                
                break
            
            # Check if we should continue
            if iteration + 1 >= self.max_iterations:
                logger.warning(f"Reached maximum iterations ({self.max_iterations}) without success")
                break
            
            # Refine query based on feedback
            feedback = history.get_feedback_for_refinement()
            logger.info(f"Refining query with feedback: {feedback[:200]}...")
            
            success, error_msg, refined_sql, refine_usage = self.gemini_client.refine_query(
                original_sql=current_sql,
                feedback=feedback,
                insight=insight,
                datasets=dataset_dicts,
                llm_mode=llm_mode
            )
            
            # Track refinement token usage
            query_refinement_tokens += refine_usage.get("total_tokens", 0)
            query_llm_calls += 1
            
            if not success or not refined_sql:
                logger.error(f"Failed to refine query: {error_msg}")
                # Can't continue without refined query
                break
            
            logger.info(f"Refined SQL (iteration {iteration + 2}):\n{refined_sql}")
            logger.info("=" * 80)
            
            current_sql = refined_sql
            iteration += 1
        
        # Build final QueryResult
        total_time_ms = (time.time() - start_time) * 1000
        
        return self._build_query_result(
            sql=history.final_sql or current_sql,
            description=description,
            history=history,
            total_time_ms=total_time_ms,
            source_tables=source_tables,
            target_table_name=target_table_name,
            query_index=query_index,
            query_refinement_tokens=query_refinement_tokens,
            query_alignment_tokens=query_alignment_tokens,
            query_llm_calls=query_llm_calls
        )
    
    def _run_validation_pipeline(
        self,
        sql: str,
        insight: str,
        iter_state: IterationState,
        datasets: List[DatasetMetadata],
        llm_mode: str = "fast_llm"
    ) -> tuple[bool, Optional[float], Optional[Dict[str, Any]], Dict[str, int]]:
        """
        Run the full validation pipeline on a query.
        
        Args:
            sql: SQL query to validate
            insight: Original insight
            iter_state: Iteration state to update
            datasets: Available datasets (for field description generation)
            llm_mode: LLM model mode ('fast_llm' or 'detailed_llm')
            
        Returns:
            Tuple of (is_valid, alignment_score, validation_details, usage_metadata)
        """
        validation_details = {}
        empty_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        # Stage 0: SQLFluff validation (fast fail on obvious issues)
        logger.info("Stage 0: SQLFluff linting")
        iter_state.current_stage = ValidationStage.SQLFLUFF
        sqlfluff_valid, sqlfluff_error = self.sqlfluff_validator.validate(sql)
        iter_state.sqlfluff_passed = sqlfluff_valid
        validation_details["sqlfluff_valid"] = sqlfluff_valid
        
        if not sqlfluff_valid:
            logger.warning(f"SQLFluff validation failed: {sqlfluff_error}")
            logger.info(f"Query that failed linting:\n{sql}")
            iter_state.add_error(
                stage=ValidationStage.SQLFLUFF,
                error_type="sqlfluff_error",
                message=sqlfluff_error or "Query failed SQLFluff linting",
                details={"sql": sql[:500]}  # Include first 500 chars for context
            )
            validation_details["error_message"] = sqlfluff_error
            validation_details["error_type"] = "sqlfluff_error"
            return False, None, validation_details, empty_usage
        logger.info("✓ SQLFluff validation passed")
        
        # Skip custom syntax validation - let BigQuery be the source of truth
        logger.info("Skipping custom syntax validation - BigQuery will validate")
        iter_state.syntax_passed = True  # Mark as passed since we're not checking
        validation_details["syntax_valid"] = True
        
        # Stage 1: Dry-run validation (BigQuery) - this validates syntax and semantics
        logger.info("Stage 1: Dry-run validation (BigQuery)")
        iter_state.current_stage = ValidationStage.DRYRUN
        dryrun_valid, dryrun_error, stats = self.dryrun_validator.validate_dryrun(sql)
        iter_state.dryrun_passed = dryrun_valid
        iter_state.execution_stats = stats
        validation_details["dryrun_valid"] = dryrun_valid
        validation_details["execution_stats"] = stats
        
        if not dryrun_valid:
            logger.warning(f"BigQuery validation failed: {dryrun_error}")
            logger.info(f"Query that failed:\n{sql}")
            iter_state.add_error(
                stage=ValidationStage.DRYRUN,
                error_type="bigquery_error",
                message=dryrun_error or "Query failed BigQuery validation",
                details={"sql": sql[:500]}  # Include first 500 chars for context
            )
            return False, None, validation_details, empty_usage
        logger.info(f"✓ BigQuery validation passed (estimated {stats.get('total_bytes_processed', 0) / (1024*1024):.2f} MB)")
        
        # Stage 2: Sample execution
        logger.info("Stage 2: Sample execution")
        iter_state.current_stage = ValidationStage.EXECUTION
        
        # Extract source tables from datasets
        source_tables = self._extract_source_tables(sql, datasets)
        
        exec_success, exec_error, sample_rows, schema = self.dryrun_validator.execute_sample(
            sql, source_tables=source_tables
        )
        iter_state.execution_passed = exec_success
        iter_state.sample_results = sample_rows
        iter_state.result_schema = schema
        validation_details["execution_valid"] = exec_success
        validation_details["sample_results"] = sample_rows
        validation_details["result_schema"] = schema
        
        # Debug logging for schema
        if schema:
            logger.debug(f"Schema captured with {len(schema)} fields: {[f.get('name') for f in schema]}")
        else:
            logger.warning("Schema is empty or None after execute_sample")
        
        if not exec_success:
            logger.warning(f"Sample execution failed: {exec_error}")
            logger.info(f"Query that failed:\n{sql}")
            iter_state.add_error(
                stage=ValidationStage.EXECUTION,
                error_type="execution_error",
                message=exec_error or "Query execution failed",
                details={"sql": sql[:500]}
            )
            return False, None, validation_details, empty_usage
        logger.info(f"✓ Sample execution passed ({len(sample_rows) if sample_rows else 0} rows returned)")
        
        # NOTE: Field descriptions are NOT generated here during validation iterations
        # They are generated ONLY ONCE after the query passes all validations
        # See _generate_field_descriptions_for_final_query() called after refinement loop
        
        # Stage 3: Alignment validation
        logger.info("Stage 3: Alignment validation (AI checking if results match insight)")
        iter_state.current_stage = ValidationStage.ALIGNMENT
        aligned, align_error, score, reasoning, align_usage = self.alignment_validator.validate(
            insight=insight,
            sql=sql,
            sample_results=sample_rows or [],
            result_schema=schema or [],
            llm_mode=llm_mode
        )
        iter_state.alignment_passed = aligned
        iter_state.alignment_score = score
        validation_details["alignment_valid"] = aligned
        validation_details["alignment_score"] = score
        validation_details["alignment_reasoning"] = reasoning
        
        if not aligned:
            score_str = f"{score:.2f}" if score is not None else "N/A"
            logger.warning(f"Alignment validation failed: score={score_str}, reasoning={reasoning}")
            iter_state.add_error(
                stage=ValidationStage.ALIGNMENT,
                error_type="alignment_error",
                message=f"Query results don't align with insight (score: {score_str})",
                details={"reasoning": reasoning}
            )
            return False, score, validation_details, align_usage or empty_usage
        
        # All validations passed!
        score_str = f"{score:.2f}" if score is not None else "N/A"
        logger.info(f"✓ Alignment validation passed (score={score_str})")
        iter_state.current_stage = ValidationStage.COMPLETE
        return True, score, validation_details, align_usage or empty_usage
    
    def _generate_descriptive_query_name(
        self,
        description: str,
        sql: str,
        insight: str = "",
        query_index: int = 0,
        naming_strategy: str = "rule_based"
    ) -> str:
        """
        Generate a descriptive snake_case query name from description and SQL.
        
        Uses a hybrid approach that can leverage LLM for better names or use
        rule-based extraction for speed.
        
        Args:
            description: Query description from LLM
            sql: SQL query text
            insight: Original insight/question (for LLM naming)
            query_index: Query index for fallback naming
            naming_strategy: Strategy to use ("rule_based", "llm", or "hybrid")
            
        Returns:
            Descriptive query name in snake_case format
        """
        # Try LLM approach if strategy allows
        if naming_strategy in ("llm", "hybrid") and insight:
            llm_name = self._generate_name_with_llm(description, sql, insight, query_index)
            if llm_name:
                return llm_name
            elif naming_strategy == "llm":
                # LLM required but failed, use fallback
                logger.warning("LLM naming failed, using fallback naming")
                return f"query_{query_index}"
        
        # Use rule-based approach
        return self._generate_name_rule_based(description, sql, query_index)
    
    def _generate_name_with_llm(
        self,
        description: str,
        sql: str,
        insight: str,
        query_index: int
    ) -> Optional[str]:
        """
        Generate descriptive query name using LLM.
        
        Args:
            description: Query description
            sql: SQL query
            insight: Original insight
            query_index: Query index for fallback
            
        Returns:
            Descriptive query name or None if generation fails
        """
        try:
            # Truncate SQL for prompt (first 500 chars)
            sql_preview = sql[:500] + ("..." if len(sql) > 500 else "")
            
            prompt = f"""Generate a short, descriptive identifier for this SQL query in snake_case format.

INSIGHT: {insight}

QUERY DESCRIPTION: {description}

SQL PREVIEW:
{sql_preview}

Requirements:
1. Maximum 5 words (joined by underscores)
2. Use snake_case format (e.g., "simple_inner_join_approach")
3. Capture the key distinguishing feature of this query
4. Be specific and meaningful (avoid generic terms like "query" or "data")
5. Focus on the approach or technique used (e.g., "cte_with_confidence_scoring", "direct_spread_calculation", "segmented_performance_analysis")

Return ONLY the snake_case identifier, nothing else."""

            logger.debug("Generating descriptive query name with LLM...")
            response = self.gemini_client.client.generate_content(prompt)
            name = response.text.strip().lower()
            
            # Sanitize: only allow alphanumeric and underscores
            name = re.sub(r'[^a-z0-9_]', '_', name)
            name = re.sub(r'_+', '_', name)  # Replace multiple underscores with single
            name = name.strip('_')
            
            # Validate length (max 60 chars)
            if name and len(name) <= 60 and len(name) >= 3:
                logger.info(f"Generated descriptive name with LLM: {name}")
                return name
            else:
                logger.warning(f"LLM generated invalid name (length={len(name)}): {name}")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to generate descriptive query name with LLM: {e}")
            return None
    
    def _generate_name_rule_based(
        self,
        description: str,
        sql: str,
        query_index: int
    ) -> str:
        """
        Generate descriptive query name using rule-based extraction.
        
        Extracts key technical terms and patterns from the description and SQL.
        
        Args:
            description: Query description
            sql: SQL query
            query_index: Query index for fallback
            
        Returns:
            Descriptive query name in snake_case
        """
        # Extract key technical terms from description and SQL
        key_terms: Set[str] = set()
        
        # Look for common SQL patterns in both description and SQL
        patterns = {
            'cte': r'\b(CTE|Common Table Expression|WITH\s+\w+\s+AS)\b',
            'window': r'\b(window function|OVER|PARTITION BY|ROW_NUMBER|RANK)\b',
            'aggregate': r'\b(SUM|AVG|COUNT|MAX|MIN|aggregat)\b',
            'subquery': r'\b(subquery|nested|derived table)\b',
            'union': r'\bUNION\b',
            'pivot': r'\b(PIVOT|UNPIVOT|pivot)\b',
            'lateral': r'\b(LATERAL|CROSS APPLY)\b',
        }
        
        description_lower = description.lower()
        sql_upper = sql.upper()
        
        # Check for patterns in description or SQL
        for term, pattern in patterns.items():
            if re.search(pattern, description, re.IGNORECASE) or re.search(pattern, sql_upper):
                key_terms.add(term)
        
        # Look for join types
        join_patterns = {
            'inner_join': r'\bINNER\s+JOIN\b',
            'left_join': r'\bLEFT\s+(OUTER\s+)?JOIN\b',
            'cross_join': r'\bCROSS\s+JOIN\b',
            'full_join': r'\bFULL\s+(OUTER\s+)?JOIN\b',
        }
        
        for join_term, join_pattern in join_patterns.items():
            if re.search(join_pattern, sql_upper):
                key_terms.add(join_term)
                break  # Only add one join type
        
        # Extract descriptive adjectives from description
        descriptive_words = [
            'simple', 'direct', 'modular', 'complex', 'optimized', 
            'efficient', 'comprehensive', 'detailed', 'basic', 'advanced',
            'straightforward', 'sophisticated', 'refined', 'enhanced'
        ]
        
        for word in descriptive_words:
            if re.search(rf'\b{word}\b', description_lower):
                key_terms.add(word)
                break  # Only add one adjective
        
        # Look for approach/strategy/method mentions
        approach_suffix = None
        if re.search(r'\bapproach\b', description_lower):
            approach_suffix = 'approach'
        elif re.search(r'\bstrategy\b', description_lower):
            approach_suffix = 'strategy'
        elif re.search(r'\bmethod\b', description_lower):
            approach_suffix = 'method'
        
        # Build name from key terms (limit to 4-5 terms max)
        if key_terms:
            name_parts = sorted(list(key_terms))[:4]
            if approach_suffix and len(name_parts) < 4:
                name_parts.append(approach_suffix)
            
            query_name = '_'.join(name_parts)
            
            # Ensure reasonable length
            if len(query_name) <= 60:
                logger.debug(f"Generated rule-based name: {query_name}")
                return query_name
        
        # Fallback: try to extract meaningful phrase from description
        # Remove common prefixes
        clean_desc = re.sub(r'^(Brief description:\s*\[?|Description:\s*)', '', description, flags=re.IGNORECASE)
        clean_desc = clean_desc.strip('[].')
        
        # Extract first few meaningful words
        words = re.findall(r'\b[a-zA-Z]+\b', clean_desc)
        if len(words) >= 3:
            # Take first 4 words, convert to snake_case
            name = '_'.join(words[:4]).lower()
            if len(name) <= 60:
                logger.debug(f"Generated fallback name from description: {name}")
                return name
        
        # Ultimate fallback
        logger.debug(f"Using index-based fallback name: query_{query_index}")
        return f"query_{query_index}"
    
    def _build_query_result(
        self,
        sql: str,
        description: str,
        history: QueryValidationHistory,
        total_time_ms: float,
        source_tables: List[str],
        target_table_name: Optional[str] = None,
        query_index: int = 0,
        query_refinement_tokens: int = 0,
        query_alignment_tokens: int = 0,
        query_llm_calls: int = 0
    ) -> QueryResult:
        """
        Build final QueryResult from validation history.
        
        Args:
            sql: Final SQL query
            description: Query description
            history: Validation history
            total_time_ms: Total time spent
            source_tables: Fully qualified table names used in query
            target_table_name: Name of target table for query naming
            query_index: Index of this query for unique naming
            
        Returns:
            QueryResult object
        """
        # Get final iteration state
        final_iter = history.get_current_iteration()
        
        # Build ValidationResult
        validation_result = ValidationResult(
            is_valid=history.final_valid,
            error_message=None if history.final_valid else self._get_final_error_message(history),
            error_type=None if history.final_valid else self._get_final_error_type(history),
            syntax_valid=final_iter.syntax_passed if final_iter else False,
            dryrun_valid=final_iter.dryrun_passed if final_iter else False,
            execution_valid=final_iter.execution_passed if final_iter else False,
            alignment_valid=final_iter.alignment_passed if final_iter else False,
            alignment_score=history.final_alignment_score,
            alignment_reasoning=None,
            execution_stats=final_iter.execution_stats if final_iter else None,
            sample_results=final_iter.sample_results if final_iter else None,
            result_schema=final_iter.result_schema if final_iter else None
        )
        
        # Generate descriptive query_name
        insight = history.insight or ""
        descriptive_name = self._generate_descriptive_query_name(
            description=description,
            sql=sql,
            insight=insight,
            query_index=query_index,
            naming_strategy=self.query_naming_strategy
        )
        
        # Add target table prefix if specified
        if target_table_name:
            query_name = f"{target_table_name}_{descriptive_name}"
        else:
            query_name = descriptive_name
        
        # Build QueryResult
        query_result = QueryResult(
            query_name=query_name,
            sql=sql,
            description=description,
            source_tables=source_tables,
            validation_status="valid" if history.final_valid else "failed",
            validation_details=validation_result,
            alignment_score=history.final_alignment_score or 0.0,
            iterations=history.total_iterations,
            generation_time_ms=total_time_ms,
            estimated_cost_usd=self._extract_cost(final_iter),
            estimated_bytes_processed=self._extract_bytes(final_iter),
            token_usage={
                "refinement_tokens": query_refinement_tokens,
                "alignment_tokens": query_alignment_tokens,
                "llm_calls": query_llm_calls
            } if (query_refinement_tokens + query_alignment_tokens) > 0 else None
        )
        
        return query_result
    
    def _get_final_error_message(self, history: QueryValidationHistory) -> str:
        """Get final error message from history."""
        final_iter = history.get_current_iteration()
        if final_iter and final_iter.errors:
            return final_iter.errors[-1].message
        return "Validation failed"
    
    def _get_final_error_type(self, history: QueryValidationHistory) -> str:
        """Get final error type from history."""
        final_iter = history.get_current_iteration()
        if final_iter and final_iter.errors:
            return final_iter.errors[-1].error_type
        return "unknown"
    
    def _extract_cost(self, iter_state: Optional[IterationState]) -> Optional[float]:
        """Extract estimated cost from iteration state."""
        if iter_state and iter_state.execution_stats:
            return iter_state.execution_stats.get("estimated_cost_usd")
        return None
    
    def _extract_bytes(self, iter_state: Optional[IterationState]) -> Optional[int]:
        """Extract estimated bytes from iteration state."""
        if iter_state and iter_state.execution_stats:
            return iter_state.execution_stats.get("total_bytes_processed")
        return None
    
    def _dataset_to_dict(self, dataset: DatasetMetadata) -> Dict[str, Any]:
        """Convert DatasetMetadata to dictionary."""
        return {
            "project_id": dataset.project_id,
            "dataset_id": dataset.dataset_id,
            "table_id": dataset.table_id,
            "asset_type": dataset.asset_type,
            "row_count": dataset.row_count,
            "size_bytes": dataset.size_bytes,
            "column_count": dataset.column_count,
            "schema_fields": dataset.schema_fields,
            "full_markdown": dataset.full_markdown,
            "has_pii": dataset.has_pii,
            "has_phi": dataset.has_phi,
            "environment": dataset.environment,
            "tags": dataset.tags
        }
    
    def _extract_source_tables(self, sql: str, datasets: List[DatasetMetadata]) -> List[str]:
        """
        Extract fully qualified table names used in SQL.
        
        Args:
            sql: The SQL query
            datasets: Available datasets
            
        Returns:
            List of fully qualified table IDs that appear in the SQL
        """
        source_tables = []
        for dataset in datasets:
            table_id = dataset.get_full_table_id()
            if table_id in sql:
                source_tables.append(table_id)
        return source_tables
    
    def _prepare_datasets_for_descriptions(self, datasets: List[DatasetMetadata]) -> List[Dict[str, Any]]:
        """
        Prepare dataset metadata for description generation.
        
        Extracts source table schemas including field descriptions from data-discovery-agent.
        The schema field contains: [{"name": "...", "type": "...", "description": "..."}]
        
        Args:
            datasets: List of DatasetMetadata objects
            
        Returns:
            List of dicts with table_id, description, and schema (including field descriptions)
        """
        return [
            {
                "table_id": ds.get_full_table_id(),
                "description": ds.description,
                "schema": ds.schema  # CRITICAL: Contains field descriptions from data-discovery-agent
            }
            for ds in datasets
        ]
    
    def _generate_field_descriptions_for_final_query(
        self,
        sql: str,
        schema: List[Dict[str, str]],
        insight: str,
        datasets: List[DatasetMetadata],
        llm_mode: str = "fast_llm"
    ) -> List[Dict[str, str]]:
        """
        Generate field descriptions for the final validated query.
        
        This is called ONLY ONCE after a query passes all validations,
        not during each refinement iteration.
        
        Args:
            sql: Final validated SQL query
            schema: Result schema from query execution
            insight: Original insight/question
            datasets: Available source datasets
            llm_mode: LLM model mode ('fast_llm' or 'detailed_llm')
            
        Returns:
            Schema with enriched field descriptions
        """
        if not schema:
            logger.warning("No schema provided for field description generation")
            return schema
        
        try:
            logger.info("Generating field descriptions for final validated query using Gemini Flash...")
            source_datasets_dicts = self._prepare_datasets_for_descriptions(datasets)
            field_descriptions = self.gemini_client.generate_field_descriptions(
                sql=sql,
                schema=schema,
                insight=insight,
                source_datasets=source_datasets_dicts,
                llm_mode=llm_mode
            )
            
            # Enrich schema with descriptions
            if field_descriptions:
                for field in schema:
                    field_name = field.get("name")
                    if field_name in field_descriptions:
                        field["description"] = field_descriptions[field_name]
                        logger.debug(f"Added description for field: {field_name}")
                
                logger.info(f"✓ Generated descriptions for {len(field_descriptions)} fields")
            else:
                logger.warning("No field descriptions generated")
            
            return schema
            
        except Exception as e:
            logger.warning(f"Failed to generate field descriptions: {e}")
            # Return original schema without descriptions - don't fail the entire query
            return schema

