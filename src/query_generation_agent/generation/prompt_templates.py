"""
Prompt Templates for Query Generation

Centralized prompt templates for Gemini interactions.
"""

from typing import Any, Dict, List, Optional


def format_schema_fields(schema_fields: List[Dict[str, Any]]) -> str:
    """
    Format schema fields for prompts (legacy function for backwards compatibility).
    
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


def format_enhanced_schema(schema_fields: List[Dict[str, Any]]) -> str:
    """
    Format schema with sample values for richer prompts.
    
    Args:
        schema_fields: List of field dictionaries with sample_values
        
    Returns:
        Formatted schema string with samples
    """
    if not schema_fields:
        return "  (schema not available)"
    
    lines = []
    for field in schema_fields:
        name = field.get("name", "unknown")
        ftype = field.get("type", "unknown")
        desc = field.get("description", "")
        samples = field.get("sample_values", [])
        
        line = f"  - {name} ({ftype})"
        if desc:
            line += f": {desc}"
        if samples:
            # Show first 3 sample values
            sample_str = ", ".join(f'"{s}"' if isinstance(s, str) else str(s) for s in samples[:3])
            line += f" [samples: {sample_str}]"
        
        lines.append(line)
    
    return "\n".join(lines)


def format_column_statistics(profiles: List[Dict[str, Any]]) -> str:
    """
    Format column statistics for prompts.
    
    Args:
        profiles: List of column profile dictionaries
        
    Returns:
        Formatted column statistics string
    """
    if not profiles:
        return "  (no statistics available)"
    
    lines = []
    for profile in profiles[:10]:  # Limit to first 10 columns
        col = profile.get("column_name")
        ptype = profile.get("profile_type")
        
        if ptype == "numeric":
            line = f"  - {col}: min={profile.get('min_value')}, max={profile.get('max_value')}, avg={profile.get('avg_value')}, distinct={profile.get('distinct_count')}"
        elif ptype == "string":
            line = f"  - {col}: distinct values={profile.get('distinct_count')}, length={profile.get('min_value')}-{profile.get('max_value')}"
        else:
            line = f"  - {col}: {profile.get('distinct_count')} distinct values"
        
        lines.append(line)
    
    return "\n".join(lines)


def format_dataset_info(datasets: List[Dict[str, Any]], include_full_docs: bool = True) -> str:
    """
    Format dataset information for prompts with rich metadata.
    
    Args:
        datasets: List of dataset metadata
        include_full_docs: Whether to include full markdown documentation
        
    Returns:
        Formatted dataset information
    """
    dataset_info = []
    for ds in datasets:
        table_id = f"{ds['project_id']}.{ds['dataset_id']}.{ds['table_id']}"
        
        # Use enhanced schema formatting if available, otherwise fall back to legacy
        schema_fields = ds.get("schema", ds.get("schema_fields", []))
        schema_text = format_enhanced_schema(schema_fields)
        
        # Build dataset info block
        info = f"""
Table: `{table_id}`
Description: {ds.get('description', 'No description')}
Rows: {ds.get('row_count', 'unknown'):,}
Type: {ds.get('asset_type', 'TABLE')}

Schema:
{schema_text}
"""
        
        # Add column statistics summary
        if ds.get('column_profiles'):
            stats = format_column_statistics(ds['column_profiles'])
            info += f"""
Column Statistics:
{stats}
"""
        
        # Add analytical insights examples
        if ds.get('analytical_insights'):
            insights = "\n".join(f"  - {insight}" for insight in ds['analytical_insights'][:3])
            info += f"""
Example Analytics Questions:
{insights}
"""
        
        # Add lineage info if available
        if ds.get('lineage'):
            sources = [l['source'] for l in ds['lineage'] if l.get('source')]
            if sources:
                info += f"""
Data Sources: {', '.join(sources[:3])}
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

QUERY GENERATION GUIDELINES:
1. USE THE SAMPLE VALUES: The schema includes sample values for each column - use these to understand data patterns and write appropriate filters
2. LEVERAGE COLUMN STATISTICS: Min/max values and distinct counts help you write efficient WHERE clauses and understand data distributions
3. LEARN FROM ANALYTICAL INSIGHTS: The example analytics questions show common use patterns for these tables
4. CONSIDER DATA LINEAGE: Upstream sources indicate data freshness and quality
5. USE APPROPRIATE AGGREGATIONS: For numeric columns with wide ranges, consider aggregations like AVG, SUM, COUNT
6. FILTER WISELY: Use sample values to create realistic filter conditions
7. GROUP STRATEGICALLY: Columns with low distinct counts (<100) are good candidates for GROUP BY

REQUIREMENTS:
1. Generate exactly {num_queries} distinct queries that answer the insight from different angles
2. Use only the tables and columns shown above
3. Write valid BigQuery SQL (use backticks for table/column names with special chars)
4. Include appropriate aggregations, filters, and GROUP BY clauses
5. Leverage sample values to create realistic examples
6. Consider column statistics when choosing aggregation strategies
7. Each query should be self-contained and executable
8. Include a brief description of what each query does

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


# Context-Aware Prompt Templates


def build_example_queries_section(example_queries: List[str]) -> str:
    """
    Build section showing example SQL queries.
    
    Args:
        example_queries: List of SQL query examples
        
    Returns:
        Formatted section string or empty string if no examples
    """
    if not example_queries:
        return ""
    
    examples_text = "\n\n".join(
        f"Example {i}:\n```sql\n{query}\n```"
        for i, query in enumerate(example_queries[:3], 1)  # Limit to 3 examples
    )
    
    return f"""
EXAMPLE SQL PATTERNS (from user):
The user provided these example queries as reference patterns. Use them as templates or inspiration 
for the new queries. Adapt their structure, logic, and patterns to work with the available datasets.

{examples_text}

IMPORTANT: Analyze these examples to understand:
- The query structure and JOIN patterns
- The type of analysis being performed
- The aggregation and grouping strategies used
- The filtering and WHERE clause patterns
Then adapt these patterns to the available tables and schema.
"""


def build_intent_section(
    inferred_intent: Optional[str],
    pattern_keywords: Optional[List[str]]
) -> str:
    """
    Build section explaining detected intent and patterns.
    
    Args:
        inferred_intent: Primary query intent
        pattern_keywords: Detected pattern keywords
        
    Returns:
        Formatted section string or empty string if no intent/patterns
    """
    if not inferred_intent and not pattern_keywords:
        return ""
    
    parts = []
    
    if inferred_intent:
        parts.append(f"**PRIMARY QUERY INTENT**: {inferred_intent.upper().replace('_', ' ')}")
    
    if pattern_keywords:
        patterns_str = ", ".join(pattern_keywords)
        parts.append(f"**DETECTED PATTERNS**: {patterns_str}")
    
    section = "\n".join(parts)
    
    return f"""
DETECTED ANALYSIS TYPE:
{section}

Focus your query generation on this type of analysis. Use SQL patterns and techniques 
that are appropriate for {inferred_intent or 'this analysis'}.
"""


def build_generation_prompt_with_context(
    insight: str,
    datasets: List[Dict[str, Any]],
    num_queries: int,
    example_queries: Optional[List[str]] = None,
    pattern_keywords: Optional[List[str]] = None,
    inferred_intent: Optional[str] = None
) -> str:
    """
    Build context-aware prompt for query generation.
    
    Args:
        insight: Data science insight
        datasets: Dataset metadata
        num_queries: Number of queries to generate
        example_queries: Optional example SQL queries
        pattern_keywords: Optional pattern keywords
        inferred_intent: Optional inferred intent
        
    Returns:
        Complete prompt string with context
    """
    datasets_text = format_dataset_info(datasets)
    
    # Build optional sections
    examples_section = build_example_queries_section(example_queries or [])
    intent_section = build_intent_section(inferred_intent, pattern_keywords)
    
    # Build intent-specific guidance
    intent_guidance = ""
    if inferred_intent:
        guidance_map = {
            "cohort_analysis": "Focus on grouping by time periods and tracking metrics across cohorts. Use DATE_TRUNC or similar functions.",
            "time_series_analysis": "Include temporal grouping and sorting. Consider window functions for trends.",
            "aggregation": "Use appropriate aggregate functions (SUM, AVG, COUNT, MAX, MIN). Consider GROUP BY for segmentation.",
            "filtering": "Write precise WHERE clauses. Use sample values to understand valid filter conditions.",
            "joining": "Carefully examine JOIN keys in sample values. Ensure compatible types and matching values.",
            "pivot_analysis": "Consider using CASE WHEN for pivoting or BigQuery's PIVOT syntax.",
            "window_functions": "Use OVER clauses with PARTITION BY and ORDER BY for window functions.",
            "ranking": "Use RANK(), DENSE_RANK(), or ROW_NUMBER() with appropriate PARTITION BY.",
        }
        intent_guidance = guidance_map.get(inferred_intent, "")
        if intent_guidance:
            intent_guidance = f"\n**INTENT-SPECIFIC GUIDANCE**: {intent_guidance}\n"
    
    prompt = f"""You are an expert SQL query generator for BigQuery. Your task is to generate {num_queries} different SQL queries that answer the following data science insight.

INSIGHT:
{insight}
{examples_section}
{intent_section}

AVAILABLE DATASETS:
{datasets_text}

QUERY GENERATION GUIDELINES:
1. USE THE SAMPLE VALUES: The schema includes sample values for each column - use these to understand data patterns and write appropriate filters
2. LEVERAGE COLUMN STATISTICS: Min/max values and distinct counts help you write efficient WHERE clauses and understand data distributions
3. LEARN FROM ANALYTICAL INSIGHTS: The example analytics questions show common use patterns for these tables
4. CONSIDER DATA LINEAGE: Upstream sources indicate data freshness and quality
5. USE APPROPRIATE AGGREGATIONS: For numeric columns with wide ranges, consider aggregations like AVG, SUM, COUNT
6. FILTER WISELY: Use sample values to create realistic filter conditions
7. GROUP STRATEGICALLY: Columns with low distinct counts (<100) are good candidates for GROUP BY
{intent_guidance}

REQUIREMENTS:
1. Generate exactly {num_queries} distinct queries that answer the insight from different angles
2. Use only the tables and columns shown above
3. Write valid BigQuery SQL (use backticks for table/column names with special chars)
4. Include appropriate aggregations, filters, and GROUP BY clauses
5. Leverage sample values to create realistic examples
6. Consider column statistics when choosing aggregation strategies
7. Each query should be self-contained and executable
8. Include a brief description of what each query does

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
    
    return prompt

