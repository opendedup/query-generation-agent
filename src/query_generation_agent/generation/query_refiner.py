"""
Query Refiner

Iteratively refines and validates SQL queries until they pass all checks.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from ..clients.gemini_client import GeminiClient
from ..models.request_models import DatasetMetadata
from ..models.response_models import QueryResult, ValidationResult
from ..models.validation_models import (
    IterationState,
    QueryValidationHistory,
    ValidationError,
    ValidationStage,
)
from ..validation.alignment_validator import AlignmentValidator
from ..validation.dryrun_validator import DryRunValidator
from ..validation.syntax_validator import SyntaxValidator

logger = logging.getLogger(__name__)


class QueryRefiner:
    """
    Refines and validates queries through iterative feedback loop.
    
    Pipeline:
    1. Syntax validation
    2. Dry-run validation (BigQuery)
    3. Sample execution
    4. Alignment validation (LLM)
    5. If any step fails, refine and retry
    """
    
    def __init__(
        self,
        gemini_client: GeminiClient,
        syntax_validator: SyntaxValidator,
        dryrun_validator: DryRunValidator,
        alignment_validator: AlignmentValidator,
        max_iterations: int = 10
    ):
        """
        Initialize query refiner.
        
        Args:
            gemini_client: Gemini client for refinement
            syntax_validator: Syntax validator
            dryrun_validator: Dry-run validator
            alignment_validator: Alignment validator
            max_iterations: Maximum refinement iterations per query
        """
        self.gemini_client = gemini_client
        self.syntax_validator = syntax_validator
        self.dryrun_validator = dryrun_validator
        self.alignment_validator = alignment_validator
        self.max_iterations = max_iterations
    
    def refine_and_validate(
        self,
        candidate_id: str,
        initial_sql: str,
        description: str,
        insight: str,
        datasets: List[DatasetMetadata]
    ) -> QueryResult:
        """
        Refine and validate a query candidate through iterative feedback.
        
        Args:
            candidate_id: Unique identifier for this candidate
            initial_sql: Initial SQL query
            description: Query description
            insight: Original insight
            datasets: Available datasets
            
        Returns:
            QueryResult with final validation status
        """
        start_time = time.time()
        
        logger.info(f"Starting refinement for candidate {candidate_id}")
        logger.info(f"Initial SQL:\n{initial_sql}")
        logger.info("=" * 80)
        
        # Initialize validation history
        history = QueryValidationHistory(
            candidate_id=candidate_id,
            initial_sql=initial_sql
        )
        
        # Convert datasets to dict format
        dataset_dicts = [self._dataset_to_dict(d) for d in datasets]
        
        current_sql = initial_sql
        iteration = 0
        
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
            is_valid, final_score, validation_details = self._run_validation_pipeline(
                sql=current_sql,
                insight=insight,
                iter_state=iter_state
            )
            
            # Update iteration timing
            iter_state.iteration_time_ms = (time.time() - iteration_start) * 1000
            
            # Add iteration to history
            history.add_iteration(iter_state)
            
            if is_valid:
                # Query passed all validations!
                logger.info(f"Query validated successfully after {iteration + 1} iteration(s)")
                history.final_valid = True
                history.final_sql = current_sql
                history.final_alignment_score = final_score
                
                break
            
            # Check if we should continue
            if iteration + 1 >= self.max_iterations:
                logger.warning(f"Reached maximum iterations ({self.max_iterations}) without success")
                break
            
            # Refine query based on feedback
            feedback = history.get_feedback_for_refinement()
            logger.info(f"Refining query with feedback: {feedback[:200]}...")
            
            success, error_msg, refined_sql = self.gemini_client.refine_query(
                original_sql=current_sql,
                feedback=feedback,
                insight=insight,
                datasets=dataset_dicts
            )
            
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
            total_time_ms=total_time_ms
        )
    
    def _run_validation_pipeline(
        self,
        sql: str,
        insight: str,
        iter_state: IterationState
    ) -> tuple[bool, Optional[float], Optional[Dict[str, Any]]]:
        """
        Run the full validation pipeline on a query.
        
        Args:
            sql: SQL query to validate
            insight: Original insight
            iter_state: Iteration state to update
            
        Returns:
            Tuple of (is_valid, alignment_score, validation_details)
        """
        validation_details = {}
        
        # Stage 1: Syntax validation
        logger.info("Stage 1: Syntax validation")
        iter_state.current_stage = ValidationStage.SYNTAX
        syntax_valid, syntax_error = self.syntax_validator.validate(sql)
        iter_state.syntax_passed = syntax_valid
        validation_details["syntax_valid"] = syntax_valid
        
        if not syntax_valid:
            logger.warning(f"Syntax validation failed: {syntax_error}")
            iter_state.add_error(
                stage=ValidationStage.SYNTAX,
                error_type="syntax_error",
                message=syntax_error or "SQL syntax is invalid"
            )
            return False, None, validation_details
        logger.info("✓ Syntax validation passed")
        
        # Stage 2: Dry-run validation
        logger.info("Stage 2: Dry-run validation (BigQuery)")
        iter_state.current_stage = ValidationStage.DRYRUN
        dryrun_valid, dryrun_error, stats = self.dryrun_validator.validate_dryrun(sql)
        iter_state.dryrun_passed = dryrun_valid
        validation_details["dryrun_valid"] = dryrun_valid
        validation_details["execution_stats"] = stats
        
        if not dryrun_valid:
            logger.warning(f"Dry-run validation failed: {dryrun_error}")
            iter_state.add_error(
                stage=ValidationStage.DRYRUN,
                error_type="bigquery_error",
                message=dryrun_error or "Query failed dry-run validation"
            )
            return False, None, validation_details
        logger.info(f"✓ Dry-run validation passed (estimated {stats.get('total_bytes_processed', 0) / (1024*1024):.2f} MB)")
        
        # Stage 3: Sample execution
        logger.info("Stage 3: Sample execution")
        iter_state.current_stage = ValidationStage.EXECUTION
        exec_success, exec_error, sample_rows, schema = self.dryrun_validator.execute_sample(sql)
        iter_state.execution_passed = exec_success
        validation_details["execution_valid"] = exec_success
        validation_details["sample_results"] = sample_rows
        validation_details["result_schema"] = schema
        
        if not exec_success:
            logger.warning(f"Sample execution failed: {exec_error}")
            iter_state.add_error(
                stage=ValidationStage.EXECUTION,
                error_type="execution_error",
                message=exec_error or "Query execution failed"
            )
            return False, None, validation_details
        logger.info(f"✓ Sample execution passed ({len(sample_rows) if sample_rows else 0} rows returned)")
        
        # Stage 4: Alignment validation
        logger.info("Stage 4: Alignment validation (AI checking if results match insight)")
        iter_state.current_stage = ValidationStage.ALIGNMENT
        aligned, align_error, score, reasoning = self.alignment_validator.validate(
            insight=insight,
            sql=sql,
            sample_results=sample_rows or [],
            result_schema=schema or []
        )
        iter_state.alignment_passed = aligned
        iter_state.alignment_score = score
        validation_details["alignment_valid"] = aligned
        validation_details["alignment_score"] = score
        validation_details["alignment_reasoning"] = reasoning
        
        if not aligned:
            logger.warning(f"Alignment validation failed: score={score:.2f}, reasoning={reasoning}")
            iter_state.add_error(
                stage=ValidationStage.ALIGNMENT,
                error_type="alignment_error",
                message=f"Query results don't align with insight (score: {score:.2f})",
                details={"reasoning": reasoning}
            )
            return False, score, validation_details
        
        # All validations passed!
        logger.info(f"✓ Alignment validation passed (score={score:.2f})")
        iter_state.current_stage = ValidationStage.COMPLETE
        return True, score, validation_details
    
    def _build_query_result(
        self,
        sql: str,
        description: str,
        history: QueryValidationHistory,
        total_time_ms: float
    ) -> QueryResult:
        """
        Build final QueryResult from validation history.
        
        Args:
            sql: Final SQL query
            description: Query description
            history: Validation history
            total_time_ms: Total time spent
            
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
            result_schema=None
        )
        
        # Build QueryResult
        query_result = QueryResult(
            sql=sql,
            description=description,
            validation_status="valid" if history.final_valid else "failed",
            validation_details=validation_result,
            alignment_score=history.final_alignment_score or 0.0,
            iterations=history.total_iterations,
            generation_time_ms=total_time_ms,
            estimated_cost_usd=self._extract_cost(final_iter),
            estimated_bytes_processed=self._extract_bytes(final_iter)
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

