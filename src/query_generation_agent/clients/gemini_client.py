"""
Gemini Client for Query Generation and Validation

Uses Gemini API for SQL generation and alignment validation.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

logger = logging.getLogger(__name__)


class GeminiClient:
    """
    Client for Gemini API interactions.
    
    Handles:
    - SQL query generation from insights and schemas
    - Query refinement based on validation feedback
    - Alignment validation (checking if results match intent)
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-pro-latest",
        temperature: float = 0.2,
        max_retries: int = 3
    ):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Gemini API key
            model_name: Model to use for generation
            temperature: Temperature for generation (0-1, lower = more deterministic)
            max_retries: Maximum number of retries for failed requests
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        
        # Configure Gemini API
        genai.configure(api_key=api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel(model_name)
        
        logger.info(f"Gemini client initialized with model: {model_name}")
    
    def generate_queries(
        self,
        insight: str,
        datasets: List[Dict[str, Any]],
        num_queries: int = 3
    ) -> Tuple[bool, Optional[str], Optional[List[Dict[str, str]]], Dict[str, int]]:
        """
        Generate multiple SQL query candidates for an insight.
        
        Args:
            insight: Data science question to answer
            datasets: List of dataset metadata with schemas
            num_queries: Number of queries to generate
            
        Returns:
            Tuple of (success, error_message, list of query dicts, usage_metadata)
        """
        empty_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        try:
            prompt = self._build_generation_prompt(insight, datasets, num_queries)
            
            logger.debug(f"Generating {num_queries} queries for insight")
            logger.debug("=" * 80)
            logger.debug("LLM CONTEXT - QUERY GENERATION:")
            logger.debug("-" * 80)
            logger.debug(prompt)
            logger.debug("=" * 80)
            
            response, usage = self._call_with_retry(prompt, json_mode=True)
            
            if not response:
                return False, "Failed to get response from Gemini", None, usage
            
            # Parse JSON response
            try:
                queries_data = json.loads(response)
                queries = queries_data.get("queries", [])
                
                if not queries:
                    return False, "No queries returned from Gemini", None, usage
                
                logger.info(f"Successfully generated {len(queries)} queries")
                return True, None, queries, usage
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                return False, f"Invalid JSON response: {str(e)}", None, usage
                
        except Exception as e:
            error_msg = f"Error generating queries: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg, None, empty_usage
    
    def refine_query(
        self,
        original_sql: str,
        feedback: str,
        insight: str,
        datasets: List[Dict[str, Any]]
    ) -> Tuple[bool, Optional[str], Optional[str], Dict[str, int]]:
        """
        Refine a query based on validation feedback.
        
        Args:
            original_sql: Current SQL query that failed
            feedback: Validation feedback (errors, suggestions)
            insight: Original insight
            datasets: Dataset metadata
            
        Returns:
            Tuple of (success, error_message, refined_sql, usage_metadata)
        """
        empty_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        try:
            prompt = self._build_refinement_prompt(original_sql, feedback, insight, datasets)
            
            logger.debug("Refining query based on feedback")
            logger.debug("=" * 80)
            logger.debug("LLM CONTEXT - QUERY REFINEMENT:")
            logger.debug("-" * 80)
            logger.debug(prompt)
            logger.debug("=" * 80)
            
            response, usage = self._call_with_retry(prompt, json_mode=True)
            
            if not response:
                return False, "Failed to get response from Gemini", None, usage
            
            # Parse JSON response
            try:
                refinement_data = json.loads(response)
                refined_sql = refinement_data.get("sql", "")
                reasoning = refinement_data.get("reasoning", "")
                
                if not refined_sql:
                    return False, "No refined SQL returned", None, usage
                
                logger.info(f"Query refined successfully. Reasoning: {reasoning[:100]}")
                return True, None, refined_sql, usage
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                return False, f"Invalid JSON response: {str(e)}", None, usage
                
        except Exception as e:
            error_msg = f"Error refining query: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg, None, empty_usage
    
    def validate_alignment(
        self,
        insight: str,
        sql: str,
        sample_results: List[Dict[str, Any]],
        schema: List[Dict[str, str]]
    ) -> Tuple[bool, Optional[str], Optional[float], Optional[str], Dict[str, int]]:
        """
        Validate if query results align with the insight intent.
        
        Args:
            insight: Original data science question
            sql: SQL query that was executed
            sample_results: Sample rows from query execution
            schema: Result schema
            
        Returns:
            Tuple of (success, error_message, alignment_score, reasoning, usage_metadata)
        """
        empty_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        try:
            prompt = self._build_alignment_prompt(insight, sql, sample_results, schema)
            
            logger.debug("Validating query alignment with insight")
            logger.debug("=" * 80)
            logger.debug("LLM CONTEXT - ALIGNMENT VALIDATION:")
            logger.debug("-" * 80)
            logger.debug(prompt)
            logger.debug("=" * 80)
            
            response, usage = self._call_with_retry(prompt, json_mode=True)
            
            if not response:
                return False, "Failed to get response from Gemini", None, None, usage
            
            # Parse JSON response
            try:
                alignment_data = json.loads(response)
                score = alignment_data.get("alignment_score", 0.0)
                reasoning = alignment_data.get("reasoning", "")
                aligned = alignment_data.get("aligned", False)
                
                logger.info(f"Alignment validation: score={score:.2f}, aligned={aligned}")
                return True, None, score, reasoning, usage
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                return False, f"Invalid JSON response: {str(e)}", None, None, usage
                
        except Exception as e:
            error_msg = f"Error validating alignment: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg, None, None, empty_usage
    
    def _call_with_retry(self, prompt: str, json_mode: bool = False) -> Tuple[Optional[str], Dict[str, int]]:
        """
        Call Gemini API with retry logic.
        
        Args:
            prompt: Prompt to send
            json_mode: Whether to request JSON output
            
        Returns:
            Tuple of (response_text, usage_metadata)
            usage_metadata contains: prompt_tokens, completion_tokens, total_tokens
        """
        generation_config = GenerationConfig(
            temperature=self.temperature,
            response_mime_type="application/json" if json_mode else "text/plain"
        )
        
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                # Extract usage metadata
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count
                }
                
                logger.debug(f"Token usage: {usage['total_tokens']} total "
                           f"({usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion)")
                
                # Log the response for debugging
                logger.debug("=" * 80)
                logger.debug("LLM RESPONSE:")
                logger.debug("-" * 80)
                logger.debug(response.text)
                logger.debug("=" * 80)
                
                return response.text, usage
                
            except Exception as e:
                logger.warning(f"Gemini API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    sleep_time = 2 ** attempt
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error("All retry attempts failed")
                    return None, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        return None, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    def _build_generation_prompt(
        self,
        insight: str,
        datasets: List[Dict[str, Any]],
        num_queries: int
    ) -> str:
        """Build prompt for initial query generation."""
        
        # Format dataset schemas
        dataset_info = []
        for ds in datasets:
            table_id = f"{ds['project_id']}.{ds['dataset_id']}.{ds['table_id']}"
            schema_text = self._format_schema(ds.get("schema_fields", []))
            
            # Handle None values for optional fields
            row_count = ds.get('row_count')
            row_count_str = f"{row_count:,}" if row_count is not None else "unknown"
            
            full_markdown = ds.get('full_markdown') or '(no documentation)'
            doc_excerpt = full_markdown[:500] if len(full_markdown) > 500 else full_markdown
            
            dataset_info.append(f"""
Table: `{table_id}`
Rows: {row_count_str}
Schema:
{schema_text}

Documentation excerpt:
{doc_excerpt}...
""")
        
        datasets_text = "\n---\n".join(dataset_info)
        
        prompt = f"""You are an expert SQL query generator for BigQuery. Your task is to generate {num_queries} different SQL queries that answer the following data science insight.

INSIGHT:
{insight}

AVAILABLE DATASETS:
{datasets_text}

STRATEGY FOR GENERATING {num_queries} DIVERSE QUERIES:
Generate queries with FUNDAMENTALLY DIFFERENT approaches, not just variations. Each query should:
- Use DIFFERENT JOIN strategies (e.g., Query 1: simple JOIN, Query 2: use CTEs with aggregations, Query 3: different table combinations)
- Make DIFFERENT assumptions about data relationships (e.g., JOIN on different key combinations)
- Have DIFFERENT levels of complexity (e.g., Query 1: simplest possible, Query 2: moderate, Query 3: comprehensive)

This diversity ensures that if one approach fails (e.g., JOIN keys don't exist), other approaches might still succeed.

CRITICAL REQUIREMENTS:
1. Generate exactly {num_queries} FUNDAMENTALLY distinct queries using different strategies
2. Use ONLY tables and columns shown above - verify column names exist in schemas
3. EXAMINE SAMPLE VALUES to understand:
   - What JOIN keys might actually work (look for similar value patterns)
   - What values exist in columns (don't filter on values that don't appear in samples)
   - What data types are used (ensure JOIN keys have compatible types)
4. Write valid BigQuery SQL (use backticks for table/column names with special chars)
5. STRING LITERALS - BigQuery Syntax Rules:
   - Use DOUBLE QUOTES for strings containing apostrophes
   - Examples:
     ✅ CORRECT: WHERE company_name = "O'Reilly Media"
     ✅ CORRECT: WHERE status = "Customer's Choice"
     ✅ CORRECT: WHERE category = "Director's Cut"
     ❌ WRONG: WHERE company_name = 'O''Reilly Media'
   - For simple strings without apostrophes, single quotes are fine:
     ✅ CORRECT: WHERE status = 'active'
     ✅ CORRECT: WHERE region = 'US-WEST'
6. ENSURE queries return actual data, not NULL values:
   - Verify JOINs will match actual rows based on sample values
   - Filter for non-NULL key fields (e.g., WHERE order_id IS NOT NULL)
   - Don't explicitly set columns to NULL
7. Each query should be self-contained and executable
8. Optimize for CORRECTNESS first - a simple query that returns real data beats a complex query with NULLs

QUALITY CHECKS FOR EACH QUERY:
Before finalizing each query, verify:
- Does this query use DIFFERENT tables or JOIN logic than the others?
- Based on sample values, will this JOIN actually match rows?
- Are the key fields in my WHERE/JOIN conditions actually present in the schemas?
- Will this return meaningful data (not just NULLs)?

OUTPUT FORMAT (JSON):
{{
    "queries": [
        {{
            "sql": "SELECT ...",
            "description": "Brief description: [approach used, what makes this different from other queries]"
        }},
        ...
    ]
}}

Generate the queries now:"""
        
        return prompt
    
    def _build_refinement_prompt(
        self,
        original_sql: str,
        feedback: str,
        insight: str,
        datasets: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for query refinement."""
        
        # Format dataset schemas
        dataset_info = []
        for ds in datasets:
            table_id = f"{ds['project_id']}.{ds['dataset_id']}.{ds['table_id']}"
            schema_text = self._format_schema(ds.get("schema_fields", []))
            
            dataset_info.append(f"""
Table: `{table_id}`
Schema:
{schema_text}
""")
        
        datasets_text = "\n---\n".join(dataset_info)
        
        prompt = f"""You are an expert SQL query debugger and optimizer for BigQuery. A query failed validation and needs to be fixed.

ORIGINAL INSIGHT:
{insight}

AVAILABLE DATASETS:
{datasets_text}

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
5. STRING LITERALS: Use double quotes for strings containing apostrophes
   - Example: "O'Reilly Media" not 'O''Reilly Media'
   - For simple strings: 'active' or 'completed' is fine
6. Use only tables and columns that exist in the schemas above
7. Follow BigQuery best practices for readability and performance

OUTPUT FORMAT (JSON):
{{
    "sql": "SELECT ... (corrected query)",
    "reasoning": "Brief explanation of what was fixed and why"
}}

Generate the corrected query now:"""
        
        return prompt
    
    def _build_alignment_prompt(
        self,
        insight: str,
        sql: str,
        sample_results: List[Dict[str, Any]],
        schema: List[Dict[str, str]]
    ) -> str:
        """Build prompt for alignment validation."""
        
        # Format schema
        schema_text = "\n".join([f"- {field['name']} ({field['type']})" for field in schema])
        
        # Format sample results (limit to first 5 rows for brevity)
        # Convert datetime objects to strings for JSON serialization
        sample_text = json.dumps(sample_results[:5], indent=2, default=str)
        
        prompt = f"""You are an expert data analyst. Evaluate whether a SQL query's results align with the intended data science insight.

ORIGINAL INSIGHT:
{insight}

SQL QUERY:
{sql}

RESULT SCHEMA:
{schema_text}

SAMPLE RESULTS (first 5 rows):
{sample_text}

TASK:
Evaluate whether the query results answer the original insight correctly and completely.

EVALUATION CRITERIA:
1. Does the query calculate the right metrics?
2. Does it apply the right filters and groupings?
3. Do the result columns match what's needed to answer the insight?
4. Are the sample results reasonable and consistent with the insight?
5. Are key fields populated with real data (not NULL values)?
6. Would a data scientist find this query useful for the stated insight?

SCORING:
- 1.0: Perfect alignment - query fully answers the insight with real, non-NULL data
- 0.8-0.9: Good alignment - query answers most of the insight with minor gaps
- 0.6-0.7: Partial alignment - query is related but misses key aspects
- 0.4-0.5: Weak alignment - query is loosely related
- 0.0-0.3: Poor alignment - query doesn't answer the insight OR returns mostly NULL values

OUTPUT FORMAT (JSON):
{{
    "alignment_score": 0.0 to 1.0,
    "aligned": true or false (true if score >= 0.85),
    "reasoning": "Detailed explanation of the score, covering what works and what doesn't"
}}

Provide your evaluation now:"""
        
        return prompt
    
    def _format_schema(self, schema_fields: List[Dict[str, Any]]) -> str:
        """Format schema fields for prompt."""
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
    
    def generate_view_ddl(self, prompt: str) -> Tuple[bool, Optional[str], str]:
        """
        Generate CREATE VIEW DDL statement from prompt.
        
        Args:
            prompt: Formatted prompt with target schema and source tables
            
        Returns:
            Tuple of (success, error_message, ddl_statement)
        """
        try:
            logger.debug("Generating VIEW DDL")
            logger.debug("=" * 80)
            logger.debug("LLM CONTEXT - VIEW DDL GENERATION:")
            logger.debug("-" * 80)
            logger.debug(prompt)
            logger.debug("=" * 80)
            
            response = self._call_with_retry(prompt, json_mode=False)
            
            if not response:
                return False, "Failed to get response from Gemini", ""
            
            # Response should be DDL directly (not JSON)
            ddl = response.strip()
            
            if not ddl:
                return False, "Empty DDL returned from Gemini", ""
            
            logger.info(f"Successfully generated VIEW DDL ({len(ddl)} chars)")
            return True, None, ddl
            
        except Exception as e:
            error_msg = f"Error generating VIEW DDL: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg, ""
    
    def generate_field_descriptions(
        self,
        sql: str,
        schema: List[Dict[str, str]],
        insight: str,
        source_datasets: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Generate semantic descriptions for query output fields using Gemini Flash.
        
        Leverages existing field descriptions from source table schemas (provided by
        data-discovery-agent) and the SQL query logic to generate clear, concise 
        descriptions for each output field.
        
        Args:
            sql: The SQL query
            schema: Basic field schema (name, type, mode) from BigQuery execution
            insight: Original user insight/question
            source_datasets: Source datasets with schema descriptions from data-discovery-agent
                            Each dataset has: {"table_id": "...", "description": "...", 
                            "schema": [{"name": "...", "type": "...", "description": "..."}]}
            
        Returns:
            Dictionary mapping field names to descriptions
        """
        try:
            # Build prompt for field description generation
            prompt = self._build_field_description_prompt(sql, schema, insight, source_datasets)
            
            logger.debug("Generating field descriptions using Gemini Flash")
            logger.debug("=" * 80)
            logger.debug("LLM CONTEXT - FIELD DESCRIPTION GENERATION:")
            logger.debug("-" * 80)
            logger.debug(prompt)
            logger.debug("=" * 80)
            
            # Use Flash model for speed and cost efficiency
            flash_model = genai.GenerativeModel("gemini-flash-latest")
            
            response = flash_model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=0.2,
                    response_mime_type="application/json"
                )
            )
            
            if not response or not response.text:
                logger.warning("No response from Gemini for field descriptions")
                return {}
            
            # Parse JSON response
            try:
                descriptions = json.loads(response.text)
                
                if not isinstance(descriptions, dict):
                    logger.warning(f"Expected dict from Gemini, got {type(descriptions)}")
                    return {}
                
                logger.info(f"Successfully generated {len(descriptions)} field descriptions")
                return descriptions
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse field descriptions JSON: {e}")
                logger.debug(f"Response text: {response.text[:500]}")
                return {}
                
        except Exception as e:
            logger.warning(f"Error generating field descriptions: {e}")
            return {}
    
    def _build_field_description_prompt(
        self,
        sql: str,
        schema: List[Dict[str, str]],
        insight: str,
        source_datasets: List[Dict[str, Any]]
    ) -> str:
        """
        Build prompt for field description generation.
        
        Args:
            sql: The SQL query
            schema: Basic field schema
            insight: Original user insight
            source_datasets: Source datasets with schema descriptions
            
        Returns:
            Formatted prompt string
        """
        # Extract field names from schema
        field_names = [field.get("name") for field in schema if field.get("name")]
        
        # Build source table schema section
        source_schema_text = ""
        for ds in source_datasets:
            table_id = ds.get("table_id", "unknown")
            source_schema_text += f"\nTable: {table_id}\n"
            
            if ds.get("description"):
                source_schema_text += f"Description: {ds['description']}\n"
            
            source_schema_text += "Fields:\n"
            for field in ds.get("schema", []):
                field_name = field.get("name", "unknown")
                field_type = field.get("type", "unknown")
                field_desc = field.get("description", "")
                
                if field_desc:
                    source_schema_text += f"  - {field_name} ({field_type}): {field_desc}\n"
                else:
                    source_schema_text += f"  - {field_name} ({field_type})\n"
        
        # Build output field list
        output_fields_text = "\n".join([f"- {name}" for name in field_names])
        
        prompt = f"""Generate concise field descriptions for a SQL query's output fields.

USER QUESTION:
{insight}

SQL QUERY:
{sql}

SOURCE TABLE SCHEMAS:
{source_schema_text}

OUTPUT FIELDS NEEDING DESCRIPTIONS:
{output_fields_text}

INSTRUCTIONS:
- For direct column references from source tables, use or adapt the source field description
- For calculated fields (CASE WHEN, arithmetic operations), explain the calculation based on the SQL expression
- For aggregated fields (SUM, COUNT, AVG, etc.), describe the aggregation and what it represents
- For derived fields, explain the calculation using source field descriptions as context
- Keep each description concise (1-2 sentences maximum)
- Focus on what the field represents in the context of answering the user's question

Return ONLY valid JSON with field names as keys and descriptions as values:
{{"field_name": "Clear, concise description here", ...}}
"""
        
        return prompt

