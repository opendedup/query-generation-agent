"""
Query Planner

Uses LLM to analyze table metadata and create structured query execution plans.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from ..clients.gemini_client import GeminiClient
from ..models.query_plan import QueryPlan
from ..models.request_models import DatasetMetadata

logger = logging.getLogger(__name__)


class QueryPlanner:
    """
    Creates structured query execution plans using LLM analysis.
    
    Analyzes table metadata from Vertex AI Search (including sample values)
    to detect join opportunities, assess feasibility, and propose strategies.
    """
    
    def __init__(self, gemini_client: GeminiClient):
        """
        Initialize query planner.
        
        Args:
            gemini_client: Gemini client for LLM analysis
        """
        self.gemini_client = gemini_client
    
    def plan_query(
        self,
        insight: str,
        datasets: List[DatasetMetadata],
        conversation_context: List[str],
        llm_mode: str = "fast_llm"
    ) -> Tuple[bool, Optional[str], Optional[QueryPlan], Dict[str, int]]:
        """
        Create a structured query execution plan.
        
        Args:
            insight: User's data question or goal
            datasets: Available datasets from Vertex AI Search
            conversation_context: Recent conversation for context
            llm_mode: LLM model mode ('fast_llm' or 'detailed_llm')
            
        Returns:
            Tuple of (success, error_message, query_plan, usage_metadata)
        """
        empty_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        logger.info("=" * 80)
        logger.info("STEP 1: Building planning prompt")
        logger.info(f"Analyzing {len(datasets)} table(s) for join opportunities")
        for ds in datasets:
            logger.info(f"  - {ds.get_full_table_id()}: {ds.row_count} rows, {ds.column_count} columns")
        
        try:
            # Build comprehensive prompt with sample values
            prompt = self._build_planning_prompt(insight, datasets, conversation_context)
            
            logger.info("STEP 3: Calling LLM for query plan generation")
            logger.info(f"Using model: {llm_mode}")
            
            # Call LLM in JSON mode
            response, usage = self.gemini_client._call_with_retry(
                prompt,
                json_mode=True,
                llm_mode=llm_mode
            )
            
            if not response:
                return False, "Failed to get response from LLM", None, usage
            
            logger.info("STEP 4: Parsing LLM response into QueryPlan")
            
            # Parse JSON response
            try:
                plan_data = json.loads(response)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                return False, f"Invalid JSON response: {str(e)}", None, usage
            
            # Create QueryPlan from response
            query_plan = QueryPlan.from_dict(plan_data)
            
            logger.info("STEP 5: Calculating feasibility score")
            logger.info(f"Initial feasibility from LLM: {query_plan.feasibility_score:.2f}")
            
            # Validate plan and adjust feasibility
            logger.info("STEP 6: Validating plan against schemas")
            validation_result = self._validate_plan(query_plan, datasets)
            
            # Adjust feasibility based on validation
            query_plan.feasibility_score = validation_result["adjusted_score"]
            query_plan.potential_issues.extend(validation_result["issues"])
            
            logger.info("STEP 7: Final plan validation complete")
            logger.info(f"Final feasibility score: {query_plan.feasibility_score:.2f}")
            logger.info(f"Strategy: {query_plan.strategy}")
            if query_plan.join_strategy:
                logger.info(f"Join type: {query_plan.join_strategy.get('type')}")
                logger.info(f"Join keys: {query_plan.join_strategy.get('join_keys')}")
            if query_plan.potential_issues:
                logger.warning(f"Potential issues detected: {query_plan.potential_issues}")
            logger.info("=" * 80)
            
            return True, None, query_plan, usage
            
        except Exception as e:
            logger.error(f"Error in query planning: {e}", exc_info=True)
            return False, f"Planning error: {str(e)}", None, empty_usage
    
    def _build_planning_prompt(
        self,
        insight: str,
        datasets: List[DatasetMetadata],
        conversation_context: List[str]
    ) -> str:
        """
        Build comprehensive planning prompt with sample values.
        
        Args:
            insight: User's question
            datasets: Available tables
            conversation_context: Recent messages
            
        Returns:
            Formatted prompt string
        """
        logger.info("STEP 2: Extracting sample values from column_profiles")
        
        # Format conversation context
        context_str = "\n".join(conversation_context) if conversation_context else "(No prior conversation)"
        
        # Format each table with full details
        table_sections = []
        for i, ds in enumerate(datasets, 1):
            table_section = self._format_table_with_samples(ds, i)
            table_sections.append(table_section)
            logger.info(f"  - Extracted samples for {ds.table_id}")
        
        tables_str = "\n\n".join(table_sections)
        
        # Build comprehensive prompt
        prompt = f"""You are an expert SQL query planner. Analyze the available tables and create a structured execution plan.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USER'S GOAL:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{insight}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONVERSATION CONTEXT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE TABLES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{tables_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK: Create a structured query execution plan
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. IDENTIFY JOIN OPPORTUNITIES
   - Compare sample values across tables to detect overlapping data
   - Look for columns with same/similar names
   - Check if sample values from different tables match or relate
   - Consider both direct matches and derived joins (concat, extract, etc.)

2. DETERMINE STRATEGY
   - "single_table": Query uses only one table
   - "join": Query combines multiple tables
   - "subquery": Query uses nested SELECT
   - "union": Query combines results from multiple tables

3. PROPOSE JOIN STRATEGY (if applicable)
   - Join type: INNER, LEFT, RIGHT, FULL OUTER
   - Join keys: Which columns to join on
   - Evidence: Show sample value overlaps
   - Transformations: Any needed (CONCAT, UPPER, etc.)

4. SELECT REQUIRED COLUMNS
   - List columns needed to answer the question
   - Include purpose for each column

5. DEFINE FILTERS
   - Time-based filters (if temporal query)
   - Value filters (if specific criteria mentioned)
   - Quality filters (NULL handling, etc.)

6. SPECIFY AGGREGATIONS (if needed)
   - GROUP BY columns
   - Aggregate functions (COUNT, SUM, AVG, etc.)

7. ASSESS FEASIBILITY (0.0-1.0)
   - 1.0: All requirements met, high confidence
   - 0.8-0.9: Good plan with minor concerns
   - 0.6-0.7: Workable but has issues
   - 0.4-0.5: Low confidence, may fail
   - <0.4: Critical issues, likely to fail

OUTPUT (JSON):
{{
  "strategy": "single_table" | "join" | "subquery" | "union",
  "reasoning": "Explain why this approach makes sense based on the data",
  "tables_required": ["table1", "table2"],
  "join_strategy": {{
    "type": "INNER" | "LEFT" | "RIGHT" | "FULL",
    "join_keys": [
      {{"left": "table1.column", "right": "table2.column", "transformation": null | "CONCAT(...)"}}
    ],
    "evidence": "Sample values show overlap: ..."
  }},
  "columns_needed": [
    {{"table": "table1", "column": "col1", "purpose": "filter criteria"}},
    {{"table": "table2", "column": "col2", "purpose": "result display"}}
  ],
  "filters": [
    {{"type": "time", "column": "timestamp", "condition": "last 7 days"}},
    {{"type": "quality", "column": "id", "condition": "IS NOT NULL"}}
  ],
  "aggregations": [
    {{"function": "AVG", "column": "value", "group_by": "category"}}
  ],
  "order_by": "timestamp DESC",
  "limit": 30,
  "feasibility_score": 0.95,
  "sample_value_evidence": {{
    "join_overlap": "80% of run_id values from table1 exist in table2",
    "data_quality": "No NULL values in key join columns"
  }},
  "potential_issues": ["List any concerns or risks"],
  "alternative_approaches": ["Describe backup strategies if this fails"]
}}

IMPORTANT:
- Use sample values as PRIMARY evidence for join feasibility
- Be specific about WHY you believe joins will work
- If no good join exists, say so clearly
- Consider data types when proposing joins
- Flag potential performance issues
"""
        
        return prompt
    
    def _format_table_with_samples(self, dataset: DatasetMetadata, table_num: int) -> str:
        """
        Format a single table with schema and sample values.
        
        Args:
            dataset: Dataset metadata
            table_num: Table number for display
            
        Returns:
            Formatted table string
        """
        full_id = dataset.get_full_table_id()
        row_str = f"{dataset.row_count:,}" if dataset.row_count else "unknown"
        
        # Format description
        desc = dataset.description or "No description available"
        
        # Format schema with sample values from both schema and column_profiles
        schema_lines = []
        for field in dataset.schema:
            col_name = field.get("name", "unknown")
            col_type = field.get("type", "unknown")
            col_desc = field.get("description", "")
            
            # Get samples from schema field
            schema_samples = field.get("sample_values", [])
            
            # Also check column_profiles for additional stats
            profile = dataset.get_column_profile(col_name) if hasattr(dataset, 'get_column_profile') else None
            
            # Format sample values
            if schema_samples:
                samples_str = ", ".join(f"'{s}'" for s in schema_samples[:5])
            else:
                samples_str = "No samples"
            
            # Format profile stats if available
            stats_str = ""
            if profile:
                distinct = profile.get("distinct_count")
                null_pct = profile.get("null_percentage", 0)
                stats_parts = []
                if distinct is not None:
                    stats_parts.append(f"distinct={distinct}")
                if null_pct is not None:
                    stats_parts.append(f"nulls={null_pct:.1f}%")
                if stats_parts:
                    stats_str = f" [{', '.join(stats_parts)}]"
            
            line = f"    • {col_name} ({col_type}){stats_str}"
            if col_desc:
                line += f"\n      Description: {col_desc}"
            line += f"\n      Samples: {samples_str}"
            
            schema_lines.append(line)
        
        schema_str = "\n".join(schema_lines)
        
        # Format lineage if available
        lineage_str = ""
        if dataset.lineage:
            sources = [l.get("source", "unknown") for l in dataset.lineage if l.get("source")]
            if sources:
                lineage_str = f"\n  Lineage Sources: {', '.join(sources[:3])}"
                if len(sources) > 3:
                    lineage_str += f" (+{len(sources)-3} more)"
        
        # Format AI insights
        insights_str = ""
        if dataset.analytical_insights:
            insights = "\n    • ".join(dataset.analytical_insights[:3])
            insights_str = f"\n  AI-Generated Insights:\n    • {insights}"
            if len(dataset.analytical_insights) > 3:
                insights_str += f"\n    ... and {len(dataset.analytical_insights)-3} more insights"
        
        return f"""TABLE {table_num}: {full_id}
  Description: {desc}
  Row Count: {row_str}
  Column Count: {dataset.column_count}{lineage_str}{insights_str}
  
  Schema with Sample Values:
{schema_str}"""
    
    def _calculate_feasibility_score(self, plan: QueryPlan, datasets: List[DatasetMetadata]) -> float:
        """
        Calculate feasibility score for the plan.
        
        Args:
            plan: Query plan to validate
            datasets: Available datasets
            
        Returns:
            Adjusted feasibility score (0.0-1.0)
        """
        score = plan.feasibility_score  # Start with LLM's assessment
        
        # Verify tables exist
        available_table_ids = {ds.table_id for ds in datasets}
        for required_table in plan.tables_required:
            if required_table not in available_table_ids:
                score *= 0.5  # Big penalty for missing table
                plan.potential_issues.append(f"Table {required_table} not found in available datasets")
        
        # Verify columns exist
        table_columns = {}
        for ds in datasets:
            table_columns[ds.table_id] = {field.get("name") for field in ds.schema}
        
        for col_spec in plan.columns_needed:
            table = col_spec.get("table")
            column = col_spec.get("column")
            if table in table_columns and column not in table_columns[table]:
                score *= 0.7  # Moderate penalty for missing column
                plan.potential_issues.append(f"Column {column} not found in {table}")
        
        return max(0.0, min(1.0, score))
    
    def _validate_plan(self, plan: QueryPlan, datasets: List[DatasetMetadata]) -> Dict[str, Any]:
        """
        Validate plan against available schemas.
        
        Args:
            plan: Query plan to validate
            datasets: Available datasets
            
        Returns:
            Validation result with adjusted score and issues
        """
        logger.info(f"Checking {len(plan.tables_required)} required tables")
        logger.info(f"Checking {len(plan.columns_needed)} required columns")
        
        issues = []
        adjusted_score = plan.feasibility_score
        
        # Build lookup structures - index by both short name and full qualified name
        available_tables = {}
        for ds in datasets:
            # Add by short table_id
            available_tables[ds.table_id] = ds
            # Also add by full qualified name
            full_id = ds.get_full_table_id()
            available_tables[full_id] = ds
            # Also add project.dataset.table format variants
            if hasattr(ds, 'project_id') and hasattr(ds, 'dataset_id'):
                full_id_alt = f"{ds.project_id}.{ds.dataset_id}.{ds.table_id}"
                available_tables[full_id_alt] = ds
        
        # Validate tables
        for table_id in plan.tables_required:
            if table_id not in available_tables:
                issues.append(f"Required table '{table_id}' not found in available datasets")
                adjusted_score *= 0.5
                logger.warning(f"Table '{table_id}' not found")
            else:
                logger.info(f"Table '{table_id}' found in available datasets")
        
        # Validate columns
        for col_spec in plan.columns_needed:
            table = col_spec.get("table")
            column = col_spec.get("column")
            
            if table in available_tables:
                ds = available_tables[table]
                column_names = {field.get("name") for field in ds.schema}
                if column not in column_names:
                    issues.append(f"Column '{column}' not found in table '{table}'")
                    adjusted_score *= 0.7
                    logger.warning(f"Column '{column}' not found in '{table}'")
            else:
                logger.warning(f"Table '{table}' not in available tables for column validation")
        
        return {
            "adjusted_score": max(0.0, min(1.0, adjusted_score)),
            "issues": issues
        }

