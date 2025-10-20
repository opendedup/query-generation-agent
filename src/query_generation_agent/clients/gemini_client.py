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
    ) -> Tuple[bool, Optional[str], Optional[List[Dict[str, str]]]]:
        """
        Generate multiple SQL query candidates for an insight.
        
        Args:
            insight: Data science question to answer
            datasets: List of dataset metadata with schemas
            num_queries: Number of queries to generate
            
        Returns:
            Tuple of (success, error_message, list of query dicts)
        """
        try:
            prompt = self._build_generation_prompt(insight, datasets, num_queries)
            
            logger.debug(f"Generating {num_queries} queries for insight")
            
            response = self._call_with_retry(prompt, json_mode=True)
            
            if not response:
                return False, "Failed to get response from Gemini", None
            
            # Parse JSON response
            try:
                queries_data = json.loads(response)
                queries = queries_data.get("queries", [])
                
                if not queries:
                    return False, "No queries returned from Gemini", None
                
                logger.info(f"Successfully generated {len(queries)} queries")
                return True, None, queries
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                return False, f"Invalid JSON response: {str(e)}", None
                
        except Exception as e:
            error_msg = f"Error generating queries: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg, None
    
    def refine_query(
        self,
        original_sql: str,
        feedback: str,
        insight: str,
        datasets: List[Dict[str, Any]]
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Refine a query based on validation feedback.
        
        Args:
            original_sql: Current SQL query that failed
            feedback: Validation feedback (errors, suggestions)
            insight: Original insight
            datasets: Dataset metadata
            
        Returns:
            Tuple of (success, error_message, refined_sql)
        """
        try:
            prompt = self._build_refinement_prompt(original_sql, feedback, insight, datasets)
            
            logger.debug("Refining query based on feedback")
            
            response = self._call_with_retry(prompt, json_mode=True)
            
            if not response:
                return False, "Failed to get response from Gemini", None
            
            # Parse JSON response
            try:
                refinement_data = json.loads(response)
                refined_sql = refinement_data.get("sql", "")
                reasoning = refinement_data.get("reasoning", "")
                
                if not refined_sql:
                    return False, "No refined SQL returned", None
                
                logger.info(f"Query refined successfully. Reasoning: {reasoning[:100]}")
                return True, None, refined_sql
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                return False, f"Invalid JSON response: {str(e)}", None
                
        except Exception as e:
            error_msg = f"Error refining query: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg, None
    
    def validate_alignment(
        self,
        insight: str,
        sql: str,
        sample_results: List[Dict[str, Any]],
        schema: List[Dict[str, str]]
    ) -> Tuple[bool, Optional[str], Optional[float], Optional[str]]:
        """
        Validate if query results align with the insight intent.
        
        Args:
            insight: Original data science question
            sql: SQL query that was executed
            sample_results: Sample rows from query execution
            schema: Result schema
            
        Returns:
            Tuple of (success, error_message, alignment_score, reasoning)
        """
        try:
            prompt = self._build_alignment_prompt(insight, sql, sample_results, schema)
            
            logger.debug("Validating query alignment with insight")
            
            response = self._call_with_retry(prompt, json_mode=True)
            
            if not response:
                return False, "Failed to get response from Gemini", None, None
            
            # Parse JSON response
            try:
                alignment_data = json.loads(response)
                score = alignment_data.get("alignment_score", 0.0)
                reasoning = alignment_data.get("reasoning", "")
                aligned = alignment_data.get("aligned", False)
                
                logger.info(f"Alignment validation: score={score:.2f}, aligned={aligned}")
                return True, None, score, reasoning
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                return False, f"Invalid JSON response: {str(e)}", None, None
                
        except Exception as e:
            error_msg = f"Error validating alignment: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg, None, None
    
    def _call_with_retry(self, prompt: str, json_mode: bool = False) -> Optional[str]:
        """
        Call Gemini API with retry logic.
        
        Args:
            prompt: Prompt to send
            json_mode: Whether to request JSON output
            
        Returns:
            Response text or None if all retries failed
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
                
                return response.text
                
            except Exception as e:
                logger.warning(f"Gemini API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    sleep_time = 2 ** attempt
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error("All retry attempts failed")
                    return None
        
        return None
    
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
            
            dataset_info.append(f"""
Table: `{table_id}`
Rows: {ds.get('row_count', 'unknown'):,}
Schema:
{schema_text}

Documentation excerpt:
{ds.get('full_markdown', '')[:500]}...
""")
        
        datasets_text = "\n---\n".join(dataset_info)
        
        prompt = f"""You are an expert SQL query generator for BigQuery. Your task is to generate {num_queries} different SQL queries that answer the following data science insight.

INSIGHT:
{insight}

AVAILABLE DATASETS:
{datasets_text}

REQUIREMENTS:
1. Generate exactly {num_queries} distinct queries that answer the insight from different angles or with different approaches
2. Use only the tables and columns shown above
3. Write valid BigQuery SQL (use backticks for table/column names with special chars)
4. Include appropriate aggregations, filters, and GROUP BY clauses as needed
5. Optimize for correctness and clarity over performance
6. Each query should be self-contained and executable
7. Include a brief description of what each query does

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
5. Use only tables and columns that exist in the schemas above

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
        sample_text = json.dumps(sample_results[:5], indent=2)
        
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
5. Would a data scientist find this query useful for the stated insight?

SCORING:
- 1.0: Perfect alignment - query fully answers the insight
- 0.8-0.9: Good alignment - query answers most of the insight with minor gaps
- 0.6-0.7: Partial alignment - query is related but misses key aspects
- 0.4-0.5: Weak alignment - query is loosely related
- 0.0-0.3: Poor alignment - query doesn't answer the insight

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

