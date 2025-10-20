"""
Validation State Models

Tracks the state of query validation through iterative refinement.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ValidationStage(str, Enum):
    """Validation stages in the pipeline."""
    
    SYNTAX = "syntax"
    DRYRUN = "dryrun"
    EXECUTION = "execution"
    ALIGNMENT = "alignment"
    COMPLETE = "complete"


class ValidationError(BaseModel):
    """Error encountered during validation."""
    
    stage: ValidationStage = Field(..., description="Stage where error occurred")
    error_type: str = Field(..., description="Type of error")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    
    def get_feedback_message(self) -> str:
        """
        Get feedback message for LLM refinement.
        
        Returns:
            Formatted feedback message
        """
        feedback = f"{self.stage.value.upper()} ERROR: {self.message}"
        
        if self.details:
            feedback += "\n\nDetails:"
            for key, value in self.details.items():
                feedback += f"\n  {key}: {value}"
        
        return feedback


class IterationState(BaseModel):
    """State of a single validation iteration."""
    
    iteration_number: int = Field(..., description="Iteration number (0-indexed)", ge=0)
    sql: str = Field(..., description="SQL query being validated")
    
    # Validation results per stage
    syntax_passed: bool = Field(default=False, description="Syntax validation passed")
    dryrun_passed: bool = Field(default=False, description="Dry-run validation passed")
    execution_passed: bool = Field(default=False, description="Execution validation passed")
    alignment_passed: bool = Field(default=False, description="Alignment validation passed")
    
    # Current stage
    current_stage: ValidationStage = Field(
        default=ValidationStage.SYNTAX,
        description="Current validation stage"
    )
    
    # Errors
    errors: List[ValidationError] = Field(
        default_factory=list,
        description="Errors encountered"
    )
    
    # Results
    alignment_score: Optional[float] = Field(None, description="Alignment score if computed")
    sample_results: Optional[List[Dict[str, Any]]] = Field(None, description="Sample query results")
    execution_stats: Optional[Dict[str, Any]] = Field(None, description="Execution statistics")
    
    # Timing
    iteration_time_ms: float = Field(default=0.0, description="Time for this iteration", ge=0)
    
    def is_valid(self) -> bool:
        """
        Check if query passed all validations.
        
        Returns:
            True if all stages passed
        """
        return (
            self.syntax_passed
            and self.dryrun_passed
            and self.execution_passed
            and self.alignment_passed
        )
    
    def get_last_error(self) -> Optional[ValidationError]:
        """
        Get the most recent error.
        
        Returns:
            Last ValidationError or None
        """
        return self.errors[-1] if self.errors else None
    
    def add_error(
        self,
        stage: ValidationStage,
        error_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add an error to this iteration.
        
        Args:
            stage: Stage where error occurred
            error_type: Type of error
            message: Error message
            details: Additional details
        """
        error = ValidationError(
            stage=stage,
            error_type=error_type,
            message=message,
            details=details
        )
        self.errors.append(error)


class QueryValidationHistory(BaseModel):
    """
    Complete validation history for a query candidate.
    
    Tracks all iterations and provides feedback for refinement.
    """
    
    candidate_id: str = Field(..., description="Unique identifier for this candidate")
    initial_sql: str = Field(..., description="Initial generated SQL")
    
    # Iterations
    iterations: List[IterationState] = Field(
        default_factory=list,
        description="History of validation iterations"
    )
    
    # Final state
    final_valid: bool = Field(default=False, description="Whether query is finally valid")
    final_sql: Optional[str] = Field(None, description="Final validated SQL")
    final_alignment_score: Optional[float] = Field(None, description="Final alignment score")
    
    # Metadata
    total_iterations: int = Field(default=0, description="Total number of iterations", ge=0)
    total_time_ms: float = Field(default=0.0, description="Total time spent", ge=0)
    
    def add_iteration(self, iteration: IterationState) -> None:
        """
        Add an iteration to history.
        
        Args:
            iteration: IterationState to add
        """
        self.iterations.append(iteration)
        self.total_iterations = len(self.iterations)
        self.total_time_ms += iteration.iteration_time_ms
    
    def get_current_iteration(self) -> Optional[IterationState]:
        """
        Get the most recent iteration.
        
        Returns:
            Latest IterationState or None
        """
        return self.iterations[-1] if self.iterations else None
    
    def get_all_errors(self) -> List[ValidationError]:
        """
        Get all errors from all iterations.
        
        Returns:
            List of all ValidationErrors
        """
        all_errors = []
        for iteration in self.iterations:
            all_errors.extend(iteration.errors)
        return all_errors
    
    def get_feedback_for_refinement(self) -> str:
        """
        Generate feedback message for LLM refinement.
        
        Returns:
            Formatted feedback including all errors and context
        """
        current = self.get_current_iteration()
        if not current:
            return "No validation attempts yet"
        
        feedback_parts = [
            f"Iteration {current.iteration_number + 1} of validation failed.",
            f"\nCurrent SQL:\n{current.sql}\n"
        ]
        
        # Add error feedback
        if current.errors:
            feedback_parts.append("\nErrors encountered:")
            for error in current.errors:
                feedback_parts.append(f"\n{error.get_feedback_message()}")
        
        # Add stage-specific guidance
        if not current.syntax_passed:
            feedback_parts.append("\nThe SQL syntax is invalid. Please fix syntax errors.")
        elif not current.dryrun_passed:
            feedback_parts.append("\nThe query failed BigQuery dry-run validation. Check table/column names.")
        elif not current.execution_passed:
            feedback_parts.append("\nThe query failed during execution. Check for runtime errors.")
        elif not current.alignment_passed:
            score = current.alignment_score or 0.0
            feedback_parts.append(
                f"\nThe query results don't fully align with the insight (score: {score:.2f}). "
                "Refine the query logic to better answer the question."
            )
        
        return "\n".join(feedback_parts)
    
    def should_continue(self, max_iterations: int) -> bool:
        """
        Check if refinement should continue.
        
        Args:
            max_iterations: Maximum allowed iterations
            
        Returns:
            True if should continue refining
        """
        if self.final_valid:
            return False  # Already valid
        
        if self.total_iterations >= max_iterations:
            return False  # Hit iteration limit
        
        return True
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "candidate_id": "query_1",
                "initial_sql": "SELECT * FROM transactions",
                "iterations": [
                    {
                        "iteration_number": 0,
                        "sql": "SELECT * FROM transactions",
                        "syntax_passed": True,
                        "dryrun_passed": False,
                        "errors": [
                            {
                                "stage": "dryrun",
                                "error_type": "table_not_found",
                                "message": "Table not found"
                            }
                        ],
                        "iteration_time_ms": 500.0
                    }
                ],
                "final_valid": False,
                "total_iterations": 1,
                "total_time_ms": 500.0
            }
        }

