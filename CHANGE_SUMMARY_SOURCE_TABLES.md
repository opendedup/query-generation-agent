# Change Summary: Added `source_tables` Field to Query Generation Output

## Overview

Added a new `source_tables` field to the `QueryResult` model in the query-generation-agent output. This field contains the fully qualified table names (in format `project.dataset.table`) that are used in each generated SQL query.

## Motivation

This change enables seamless integration with the data-graphql-agent, which requires `source_tables` as part of its input format. By including this field in the query-generation-agent output:

1. **Eliminates need for SQL parsing**: No need to extract table names from SQL using regex (which is error-prone)
2. **Single source of truth**: The query generator already knows which tables it used
3. **Better traceability**: Documents exactly which datasets were used for each query
4. **Enables lineage tracking**: Makes it easier to track data lineage from source → query → GraphQL API

## Files Changed

### 1. Response Models (`src/query_generation_agent/models/response_models.py`)

#### Changes:
- Added `source_tables: List[str]` field to `QueryResult` class (line 88-91)
- Updated example documentation in `QueryResult.Config` (line 157)
- Updated example documentation in `GenerateQueriesResponse.Config` (line 263)

**Example Output:**
```json
{
  "sql": "SELECT week, AVG(edge) FROM table GROUP BY week",
  "description": "Calculate average edge by week",
  "source_tables": ["lennyisagoodboy.lfndata.regression_predictions"],
  "validation_status": "valid",
  "validation_details": {...},
  "alignment_score": 0.95,
  "iterations": 1,
  "generation_time_ms": 1500.0
}
```

### 2. Query Refiner (`src/query_generation_agent/generation/query_refiner.py`)

#### Changes:
- Extract source tables from datasets in `refine_and_validate()` method (line 85-86)
- Pass `source_tables` to `_build_query_result()` method (line 170)
- Updated `_build_query_result()` signature to accept `source_tables` parameter (line 274)
- Include `source_tables` when creating `QueryResult` object (line 312)

**Key Implementation:**
```python
# Extract source tables from datasets
source_tables = [dataset.get_full_table_id() for dataset in datasets]

# Pass to result builder
return self._build_query_result(
    sql=history.final_sql or current_sql,
    description=description,
    history=history,
    total_time_ms=total_time_ms,
    source_tables=source_tables
)
```

### 3. Unit Tests (`tests/unit/test_models.py`)

#### Changes:
- Updated `test_query_result()` to include `source_tables` field (line 120)
- Updated `test_generate_queries_response()` to include `source_tables` for both queries (lines 153, 164)
- Added assertion to verify `source_tables` value (line 131)

## Backward Compatibility

⚠️ **Breaking Change**: This is a **breaking change** because `source_tables` is now a **required field** in the `QueryResult` model.

Any code that creates `QueryResult` objects directly must now provide the `source_tables` parameter.

### Migration Guide

**Before:**
```python
query = QueryResult(
    sql="SELECT * FROM table",
    description="Query description",
    validation_status="valid",
    validation_details=validation,
    alignment_score=0.92,
    iterations=2,
    generation_time_ms=1500.0
)
```

**After:**
```python
query = QueryResult(
    sql="SELECT * FROM table",
    description="Query description",
    source_tables=["project.dataset.table"],  # ← ADD THIS
    validation_status="valid",
    validation_details=validation,
    alignment_score=0.92,
    iterations=2,
    generation_time_ms=1500.0
)
```

## Testing

All unit tests pass successfully:
```bash
$ poetry run pytest tests/unit/test_models.py -v
============================= test session starts ==============================
collected 5 items

tests/unit/test_models.py::test_dataset_metadata_creation PASSED         [ 20%]
tests/unit/test_models.py::test_generate_queries_request_validation PASSED [ 40%]
tests/unit/test_models.py::test_validation_result PASSED                 [ 60%]
tests/unit/test_models.py::test_query_result PASSED                      [ 80%]
tests/unit/test_models.py::test_generate_queries_response PASSED         [100%]

=============================== 5 passed, 5 warnings in 0.17s ==============================
```

## Integration with data-graphql-agent

The output from query-generation-agent now includes all fields required by data-graphql-agent:

### Query Generation Output → GraphQL Input Transformation

**Query Generation Agent Output:**
```json
{
  "queries": [
    {
      "sql": "SELECT...",
      "description": "Calculate average...",
      "source_tables": ["project.dataset.table"],
      "validation_status": "valid",
      "alignment_score": 0.95
    }
  ]
}
```

**Transform to GraphQL Agent Input:**
```json
{
  "queries": [
    {
      "queryName": "calculateAverage",
      "sql": "SELECT...",
      "source_tables": ["project.dataset.table"]
    }
  ],
  "project_name": "analytics-api"
}
```

**Simple Transformation Function:**
```python
def transform_to_graphql_input(query_gen_output, project_name):
    return {
        "queries": [
            {
                "queryName": generate_camel_case(q["description"]),
                "sql": q["sql"],
                "source_tables": q["source_tables"]  # ← Already included!
            }
            for q in query_gen_output["queries"]
            if q["validation_status"] == "valid"
        ],
        "project_name": project_name
    }
```

## Benefits

1. ✅ **No SQL Parsing Required**: Eliminates error-prone regex-based table extraction
2. ✅ **Accurate**: Uses the exact datasets that were provided to the query generator
3. ✅ **Traceable**: Full lineage from dataset discovery → query generation → GraphQL API
4. ✅ **Simple Integration**: Direct field mapping between agents
5. ✅ **Handles Complex Queries**: Works with JOINs, CTEs, subqueries without parsing issues

## Example

See `examples/test_source_tables.py` for a working demonstration:

```bash
$ poetry run python examples/test_source_tables.py
```

## Next Steps

1. Update any downstream consumers of the query-generation-agent API to include `source_tables`
2. Update the integrated workflow example to leverage `source_tables` for GraphQL generation
3. Consider adding `source_tables` validation (e.g., verify tables exist in BigQuery)

## Author

AI Assistant (Claude Sonnet 4.5)

## Date

October 23, 2025

