"""
Alignment Validator

Validates if query results align with the original insight using LLM.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from ..clients.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class AlignmentValidator:
    """
    Validates semantic alignment between query results and insight intent.
    
    Uses LLM to evaluate if the query correctly answers the data science question.
    """
    
    def __init__(
        self,
        gemini_client: GeminiClient,
        alignment_threshold: float = 0.85
    ):
        """
        Initialize alignment validator.
        
        Args:
            gemini_client: Gemini client for LLM validation
            alignment_threshold: Minimum score to consider aligned (0-1)
        """
        self.gemini_client = gemini_client
        self.alignment_threshold = alignment_threshold
    
    def validate(
        self,
        insight: str,
        sql: str,
        sample_results: List[Dict[str, Any]],
        result_schema: List[Dict[str, str]]
    ) -> Tuple[bool, Optional[str], Optional[float], Optional[str], Optional[Dict[str, Any]]]:
        """
        Validate alignment between query results and insight.
        
        Args:
            insight: Original data science insight/question
            sql: SQL query that was executed
            sample_results: Sample rows from query execution
            result_schema: Schema of query results
            
        Returns:
            Tuple of (is_aligned, error_message, alignment_score, reasoning, usage)
        """
        # Validate inputs
        if not sample_results:
            logger.warning("No sample results provided for alignment validation")
            return False, "No sample results to validate", 0.0, "Query returned no results", None
        
        if not result_schema:
            logger.warning("No result schema provided for alignment validation")
            # Can still proceed with validation
            result_schema = []
        
        # Check for NULL-heavy results (potential gaming of validation)
        null_ratio = self._calculate_null_ratio(sample_results)
        if null_ratio > 0.5:
            logger.warning(f"Query results contain {null_ratio*100:.1f}% NULL values")
            return False, "Results contain too many NULL values", 0.0, \
                   "Query returns rows but most fields are NULL. This suggests incorrect JOINs or data issues.", None
        
        logger.debug(f"Validating alignment for insight: {insight[:100]}...")
        
        # Call Gemini for alignment validation
        success, error_msg, score, reasoning, usage = self.gemini_client.validate_alignment(
            insight=insight,
            sql=sql,
            sample_results=sample_results,
            schema=result_schema
        )
        
        if not success:
            logger.error(f"Alignment validation failed: {error_msg}")
            return False, error_msg, None, None, None
        
        # Check if score meets threshold
        is_aligned = score >= self.alignment_threshold
        
        if is_aligned:
            logger.info(f"Alignment validation passed. Score: {score:.2f} (threshold: {self.alignment_threshold})")
        else:
            logger.info(
                f"Alignment validation failed. Score: {score:.2f} below threshold {self.alignment_threshold}"
            )
        
        return is_aligned, None, score, reasoning, usage
    
    def validate_with_feedback(
        self,
        insight: str,
        sql: str,
        sample_results: List[Dict[str, Any]],
        result_schema: List[Dict[str, str]]
    ) -> Tuple[bool, Optional[str], Optional[float], Optional[str], Optional[str]]:
        """
        Validate alignment and provide feedback for refinement.
        
        Args:
            insight: Original data science insight/question
            sql: SQL query that was executed
            sample_results: Sample rows from query execution
            result_schema: Schema of query results
            
        Returns:
            Tuple of (is_aligned, error_message, alignment_score, reasoning, feedback)
        """
        is_aligned, error_msg, score, reasoning, _ = self.validate(
            insight, sql, sample_results, result_schema
        )
        
        # Generate feedback for refinement if not aligned
        feedback = None
        if not is_aligned and score is not None:
            feedback = self._generate_refinement_feedback(
                insight, sql, score, reasoning, sample_results, result_schema
            )
        
        return is_aligned, error_msg, score, reasoning, feedback
    
    def _generate_refinement_feedback(
        self,
        insight: str,
        sql: str,
        score: float,
        reasoning: str,
        sample_results: List[Dict[str, Any]],
        result_schema: List[Dict[str, str]]
    ) -> str:
        """
        Generate detailed feedback for query refinement.
        
        Args:
            insight: Original insight
            sql: Current SQL
            score: Alignment score
            reasoning: LLM reasoning
            sample_results: Sample results
            result_schema: Result schema
            
        Returns:
            Feedback message for refinement
        """
        feedback_parts = [
            f"Alignment validation failed (score: {score:.2f}, threshold: {self.alignment_threshold})",
            "",
            "LLM Evaluation:",
            reasoning,
            "",
            "Current Query:",
            sql,
            "",
            "Result Schema:",
        ]
        
        for field in result_schema:
            feedback_parts.append(f"  - {field['name']} ({field['type']})")
        
        feedback_parts.extend([
            "",
            "Sample Results:",
            self._format_sample_results(sample_results[:3]),
            "",
            "To improve alignment:",
            "- Review if the query calculates the correct metrics",
            "- Check if filters and groupings match the insight requirements",
            "- Ensure result columns directly answer the question",
            "- Consider if additional columns or calculations are needed"
        ])
        
        return "\n".join(feedback_parts)
    
    def _format_sample_results(self, sample_results: List[Dict[str, Any]]) -> str:
        """
        Format sample results for display.
        
        Args:
            sample_results: Sample result rows
            
        Returns:
            Formatted string
        """
        if not sample_results:
            return "  (no results)"
        
        lines = []
        for i, row in enumerate(sample_results, 1):
            lines.append(f"  Row {i}: {row}")
        
        return "\n".join(lines)
    
    def quick_validate(
        self,
        insight: str,
        sql: str,
        sample_results: List[Dict[str, Any]]
    ) -> bool:
        """
        Quick alignment validation without detailed feedback.
        
        Args:
            insight: Original insight
            sql: SQL query
            sample_results: Sample results
            
        Returns:
            True if aligned (score >= threshold)
        """
        # Use empty schema for quick validation
        is_aligned, _, _, _, _ = self.validate(
            insight=insight,
            sql=sql,
            sample_results=sample_results,
            result_schema=[]
        )
        
        return is_aligned
    
    def _calculate_null_ratio(self, sample_results: List[Dict[str, Any]]) -> float:
        """
        Calculate ratio of NULL values in sample results.
        
        Used to detect queries that return rows but with mostly NULL values,
        which can indicate incorrect JOINs or attempts to game validation.
        
        Args:
            sample_results: Sample rows from query execution
            
        Returns:
            Ratio of NULL values (0.0 to 1.0)
        """
        if not sample_results:
            return 1.0
        
        total_values = 0
        null_values = 0
        
        for row in sample_results:
            for value in row.values():
                total_values += 1
                if value is None:
                    null_values += 1
        
        return null_values / total_values if total_values > 0 else 1.0

