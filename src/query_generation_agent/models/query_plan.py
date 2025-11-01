"""
Query Plan Models

Defines structured query execution plans created by the query planner.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class QueryPlan:
    """
    Structured execution plan for SQL query generation.
    
    Created by QueryPlanner to guide SQL generation with validated
    join strategies, column selections, and feasibility assessment.
    """
    
    # Strategy
    strategy: str  # "single_table", "join", "subquery", "union"
    reasoning: str  # Why this plan makes sense
    
    # Tables and relationships
    tables_required: List[str]  # Table IDs needed for query
    join_strategy: Optional[Dict[str, Any]] = None  # Join details if applicable
    
    # Query structure
    columns_needed: List[Dict[str, str]] = field(default_factory=list)
    filters: List[Dict[str, str]] = field(default_factory=list)
    aggregations: List[Dict[str, str]] = field(default_factory=list)
    order_by: Optional[str] = None
    limit: int = 30
    
    # Quality metrics
    feasibility_score: float = 0.0  # 0.0-1.0 confidence
    sample_value_evidence: Dict[str, Any] = field(default_factory=dict)
    
    # Issues and alternatives
    potential_issues: List[str] = field(default_factory=list)
    alternative_approaches: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert QueryPlan to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation of the plan
        """
        return {
            "strategy": self.strategy,
            "reasoning": self.reasoning,
            "tables_required": self.tables_required,
            "join_strategy": self.join_strategy,
            "columns_needed": self.columns_needed,
            "filters": self.filters,
            "aggregations": self.aggregations,
            "order_by": self.order_by,
            "limit": self.limit,
            "feasibility_score": self.feasibility_score,
            "sample_value_evidence": self.sample_value_evidence,
            "potential_issues": self.potential_issues,
            "alternative_approaches": self.alternative_approaches
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryPlan":
        """
        Create QueryPlan from dictionary.
        
        Args:
            data: Dictionary with plan data
            
        Returns:
            QueryPlan instance
        """
        return cls(
            strategy=data.get("strategy", "unknown"),
            reasoning=data.get("reasoning", ""),
            tables_required=data.get("tables_required", []),
            join_strategy=data.get("join_strategy"),
            columns_needed=data.get("columns_needed", []),
            filters=data.get("filters", []),
            aggregations=data.get("aggregations", []),
            order_by=data.get("order_by"),
            limit=data.get("limit", 30),
            feasibility_score=data.get("feasibility_score", 0.0),
            sample_value_evidence=data.get("sample_value_evidence", {}),
            potential_issues=data.get("potential_issues", []),
            alternative_approaches=data.get("alternative_approaches", [])
        )

