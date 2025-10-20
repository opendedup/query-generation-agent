"""
Prompt Templates for Query Generation

Centralized prompt templates for Gemini interactions.
"""

from typing import Any, Dict, List


def format_schema_fields(schema_fields: List[Dict[str, Any]]) -> str:
    """
    Format schema fields for prompts.
    
    Args:
        schema_fields: List of field dictionaries
        
    Returns:
        Formatted schema string
    """
    if not schema_fields:
        return "  (schema not available)"
    
    lines = []
    for field in schema_fields:
        name = field.get("name", "unknown")
        field_type = field.get("type", "unknown")
        description = field.get("description", "")
        
        line = f"  - {name} ({field_type})"
        if description:
            line += f": {description}"
        lines.append(line)
    
    return "\n".join(lines)


def format_dataset_info(datasets: List[Dict[str, Any]], include_full_docs: bool = True) -> str:
    """
    Format dataset information for prompts.
    
    Args:
        datasets: List of dataset metadata
        include_full_docs: Whether to include full markdown documentation
        
    Returns:
        Formatted dataset information
    """
    dataset_info = []
    for ds in datasets:
        table_id = f"{ds['project_id']}.{ds['dataset_id']}.{ds['table_id']}"
        schema_text = format_schema_fields(ds.get("schema_fields", []))
        
        info = f"""
Table: `{table_id}`
Rows: {ds.get('row_count', 'unknown'):,}
Schema:
{schema_text}
"""
        
        if include_full_docs and ds.get('full_markdown'):
            # Include excerpt of documentation (first 500 chars)
            doc_excerpt = ds['full_markdown'][:500]
            info += f"""
Documentation excerpt:
{doc_excerpt}...
"""
        
        dataset_info.append(info)
    
    return "\n---\n".join(dataset_info)


# Query Generation Prompts

QUERY_GENERATION_SYSTEM = """You are an expert SQL query generator for BigQuery. You excel at:
- Writing efficient, correct BigQuery SQL
- Understanding data science requirements
- Generating multiple approaches to solve problems
- Following BigQuery best practices and syntax
"""

QUERY_GENERATION_TEMPLATE = """Your task is to generate {num_queries} different SQL queries that answer the following data science insight.

INSIGHT:
{insight}

AVAILABLE DATASETS:
{datasets}

REQUIREMENTS:
1. Generate exactly {num_queries} distinct queries that answer the insight from different angles or with different approaches
2. Use only the tables and columns shown above
3. Write valid BigQuery SQL (use backticks for table/column names with special chars)
4. Include appropriate aggregations, filters, and GROUP BY clauses as needed
5. Optimize for correctness and clarity over performance
6. Each query should be self-contained and executable
7. Include a brief description of what each query does
8. Consider edge cases (NULL values, date ranges, data quality)

OUTPUT FORMAT (JSON):
{{
    "queries": [
        {{
            "sql": "SELECT ...",
            "description": "Brief description of what this query calculates"
        }},
        ...
    ]
}}

Generate the queries now:"""


# Query Refinement Prompts

QUERY_REFINEMENT_SYSTEM = """You are an expert SQL query debugger and optimizer for BigQuery. You excel at:
- Analyzing and fixing SQL errors
- Understanding validation feedback
- Correcting syntax, schema, and logic issues
- Maintaining query intent while fixing problems
"""

QUERY_REFINEMENT_TEMPLATE = """A query failed validation and needs to be fixed.

ORIGINAL INSIGHT:
{insight}

AVAILABLE DATASETS:
{datasets}

CURRENT SQL (FAILED):
{original_sql}

VALIDATION FEEDBACK:
{feedback}

TASK:
Fix the SQL query to address the validation errors while still answering the original insight.

REQUIREMENTS:
1. Analyze the validation feedback carefully
2. Fix all identified errors (syntax, schema, logic)
3. Ensure the query still answers the original insight
4. Use valid BigQuery SQL syntax
5. Use only tables and columns that exist in the schemas above
6. If the feedback mentions specific columns or tables that don't exist, find alternatives
7. Maintain the analytical intent of the original query

OUTPUT FORMAT (JSON):
{{
    "sql": "SELECT ... (corrected query)",
    "reasoning": "Brief explanation of what was fixed and why"
}}

Generate the corrected query now:"""


# Alignment Validation Prompts

ALIGNMENT_VALIDATION_SYSTEM = """You are an expert data analyst who evaluates whether SQL query results align with data science requirements. You excel at:
- Understanding business requirements and data science insights
- Evaluating query correctness and completeness
- Identifying gaps between intent and implementation
- Providing constructive, actionable feedback
"""

ALIGNMENT_VALIDATION_TEMPLATE = """Evaluate whether a SQL query's results align with the intended data science insight.

ORIGINAL INSIGHT:
{insight}

SQL QUERY:
{sql}

RESULT SCHEMA:
{schema}

SAMPLE RESULTS (first 5 rows):
{sample_results}

TASK:
Evaluate whether the query results answer the original insight correctly and completely.

EVALUATION CRITERIA:
1. Does the query calculate the right metrics?
2. Does it apply the right filters and groupings?
3. Do the result columns match what's needed to answer the insight?
4. Are the sample results reasonable and consistent with the insight?
5. Would a data scientist find this query useful for the stated insight?
6. Does the query handle edge cases appropriately?

SCORING GUIDELINES:
- 1.0: Perfect alignment - query fully answers the insight
- 0.8-0.9: Good alignment - query answers most of the insight with minor gaps
- 0.6-0.7: Partial alignment - query is related but misses key aspects
- 0.4-0.5: Weak alignment - query is loosely related
- 0.0-0.3: Poor alignment - query doesn't answer the insight

OUTPUT FORMAT (JSON):
{{
    "alignment_score": 0.0 to 1.0,
    "aligned": true or false (true if score >= 0.85),
    "reasoning": "Detailed explanation of the score, covering what works and what doesn't",
    "suggestions": "Optional suggestions for improvement if score < 1.0"
}}

Provide your evaluation now:"""


def build_generation_prompt(insight: str, datasets: List[Dict[str, Any]], num_queries: int) -> str:
    """
    Build prompt for query generation.
    
    Args:
        insight: Data science insight
        datasets: Dataset metadata
        num_queries: Number of queries to generate
        
    Returns:
        Complete prompt string
    """
    datasets_text = format_dataset_info(datasets)
    
    return QUERY_GENERATION_TEMPLATE.format(
        insight=insight,
        datasets=datasets_text,
        num_queries=num_queries
    )


def build_refinement_prompt(
    original_sql: str,
    feedback: str,
    insight: str,
    datasets: List[Dict[str, Any]]
) -> str:
    """
    Build prompt for query refinement.
    
    Args:
        original_sql: SQL that failed validation
        feedback: Validation feedback
        insight: Original insight
        datasets: Dataset metadata
        
    Returns:
        Complete prompt string
    """
    datasets_text = format_dataset_info(datasets, include_full_docs=False)
    
    return QUERY_REFINEMENT_TEMPLATE.format(
        insight=insight,
        datasets=datasets_text,
        original_sql=original_sql,
        feedback=feedback
    )


def build_alignment_prompt(
    insight: str,
    sql: str,
    sample_results: List[Dict[str, Any]],
    schema: List[Dict[str, str]]
) -> str:
    """
    Build prompt for alignment validation.
    
    Args:
        insight: Original insight
        sql: SQL query
        sample_results: Sample query results
        schema: Result schema
        
    Returns:
        Complete prompt string
    """
    import json
    
    # Format schema
    schema_text = "\n".join([f"- {field['name']} ({field['type']})" for field in schema])
    
    # Format sample results (limit to first 5 rows)
    sample_text = json.dumps(sample_results[:5], indent=2)
    
    return ALIGNMENT_VALIDATION_TEMPLATE.format(
        insight=insight,
        sql=sql,
        schema=schema_text,
        sample_results=sample_text
    )

