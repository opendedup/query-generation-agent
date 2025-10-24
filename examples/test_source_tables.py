"""
Simple test to demonstrate the source_tables field addition.

This script creates a QueryResult and shows that source_tables is now included.
"""

from query_generation_agent.models.response_models import QueryResult, ValidationResult

# Create a validation result
validation = ValidationResult(
    is_valid=True,
    syntax_valid=True,
    dryrun_valid=True,
    execution_valid=True,
    alignment_valid=True,
    alignment_score=0.95
)

# Create a query result with source_tables
query = QueryResult(
    sql="SELECT week, AVG(edge) as avg_edge FROM `lennyisagoodboy.lfndata.regression_predictions` GROUP BY week",
    description="Calculate average edge by week",
    source_tables=["lennyisagoodboy.lfndata.regression_predictions"],
    validation_status="valid",
    validation_details=validation,
    alignment_score=0.95,
    iterations=1,
    generation_time_ms=1500.0,
    estimated_cost_usd=0.005
)

# Print as JSON to show the new field
import json
print("QueryResult with source_tables field:")
print("=" * 80)
print(json.dumps(query.model_dump(mode="json"), indent=2))
print("=" * 80)
print()
print("✓ source_tables field is now included in the output!")
print(f"✓ Source tables: {query.source_tables}")

