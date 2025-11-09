"""
Insight Parser

Extracts structured information from insight text using LLM with JSON response schema.
"""

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..clients.gemini_client import GeminiClient

from ..models.request_models import InsightContext

logger = logging.getLogger(__name__)


class ExtractionResponseSchema(BaseModel):
    """
    Pydantic schema for LLM extraction response.
    
    Defines the structured output format for insight context extraction.
    """
    
    cleaned_insight: str = Field(
        description="Clarified version of the insight without SQL blocks"
    )
    example_queries: List[str] = Field(
        default_factory=list,
        description="SQL queries from insight text (REFERENCE ONLY - may be pseudocode or have syntax errors)"
    )
    pattern_keywords: List[str] = Field(
        default_factory=list,
        description="Query pattern keywords detected (cohort, aggregation, etc.)"
    )
    primary_intent: Optional[str] = Field(
        default=None,
        description="Primary query intent classification"
    )
    reasoning: str = Field(
        description="Explanation of extraction decisions"
    )
    confidence: float = Field(
        default=0.5,
        description="Confidence score 0-1 for extraction quality",
        ge=0.0,
        le=1.0
    )


class InsightParser:
    """
    Parses insight text using LLM to extract structured context.
    
    Uses Gemini with JSON response schema to intelligently extract:
    - SQL code blocks (for use as templates/examples)
    - Dataset/table references (explicit and implicit)
    - Query pattern keywords
    - User intent classification
    """
    
    # Available intent types
    INTENT_TYPES = [
        "cohort_analysis",
        "time_series_analysis",
        "aggregation",
        "filtering",
        "joining",
        "pivot_analysis",
        "window_functions",
        "ranking",
        "deduplication",
        "data_quality_check",
        "exploratory_analysis",
        "other"
    ]
    
    # Available pattern keywords
    PATTERN_KEYWORDS = [
        "cohort", "retention", "funnel",
        "time_series", "temporal", "trend",
        "aggregation", "sum", "average", "count",
        "pivot", "crosstab",
        "window_function", "running_total", "rank",
        "join", "merge", "combine",
        "filter", "subset",
        "group_by", "segment"
    ]
    
    def __init__(self, gemini_client: "GeminiClient"):
        """
        Initialize insight parser with Gemini client.
        
        Args:
            gemini_client: Gemini client for LLM-based extraction
        """
        self.gemini_client = gemini_client
    
    def parse(self, insight: str, llm_mode: str = "fast_llm") -> InsightContext:
        """
        Parse insight text into structured context using LLM.
        
        Args:
            insight: Raw insight text from user
            llm_mode: LLM model mode ('fast_llm' or 'detailed_llm')
            
        Returns:
            InsightContext with extracted metadata
        """
        logger.info("Parsing insight with LLM for structured context...")
        
        try:
            # Build extraction prompt
            prompt = self._build_extraction_prompt(insight)
            
            # Call Gemini for extraction with JSON schema
            success, error_msg, extracted_data, usage = self.gemini_client.extract_insight_context(
                insight=insight,
                prompt=prompt,
                response_schema=ExtractionResponseSchema,
                llm_mode=llm_mode
            )
            
            if not success or not extracted_data:
                logger.warning(f"LLM extraction failed: {error_msg}, returning minimal context")
                return self._create_fallback_context(insight)
            
            # Build InsightContext from LLM extraction
            context = InsightContext(
                original_text=insight,
                cleaned_text=extracted_data.get("cleaned_insight", insight),
                example_queries=extracted_data.get("example_queries", []),
                pattern_keywords=extracted_data.get("pattern_keywords", []),
                inferred_intent=extracted_data.get("primary_intent"),
                reasoning=extracted_data.get("reasoning"),
                confidence=extracted_data.get("confidence", 0.5),
                metadata={
                    "extraction_method": "llm",
                    "token_usage": usage
                }
            )
            
            # Log extraction results
            self._log_extraction_results(context)
            
            return context
            
        except Exception as e:
            logger.error(f"Error parsing insight with LLM: {e}", exc_info=True)
            return self._create_fallback_context(insight)
    
    def _build_extraction_prompt(self, insight: str) -> str:
        """
        Build prompt for LLM to extract structured context from insight.
        
        Args:
            insight: Raw insight text
            
        Returns:
            Prompt string for Gemini
        """
        prompt = f"""You are an expert at analyzing data science questions and extracting structured information.

INSIGHT TEXT:
{insight}

Your task is to analyze this insight and extract structured information that will help generate SQL queries.

EXTRACTION TASKS:

1. **Extract SQL Examples**: Find any SQL queries mentioned in the text (in code blocks, inline, or described)
   - Extract as-is - may be pseudocode, incomplete, or have syntax errors
   - These are REFERENCE ONLY for understanding:
     * Join patterns (how tables relate)
     * Calculation approaches (aggregations, window functions)
     * Query structure and analytical flow
   - DO NOT validate syntax - focus on extracting the intent
   - If SQL is described but not written, note the pattern described

2. **Detect Query Patterns**: Identify what type of SQL pattern is needed
   - Available patterns: {', '.join(self.PATTERN_KEYWORDS)}
   - Multiple patterns are OK
   - Choose patterns that match the insight's analytical intent

3. **Classify Primary Intent**: What is the main goal of the query?
   - Available intents: {', '.join(self.INTENT_TYPES)}
   - Choose the most specific intent
   - If multiple intents apply, choose the primary one

4. **Clean the Insight**: Provide a clarified version of the insight
   - Remove SQL code blocks (will be in example_queries)
   - Clarify ambiguous language
   - Preserve the core question/requirement
   - Make it clear and actionable

5. **Provide Reasoning**: Explain your extraction decisions
   - What patterns did you detect in any SQL examples? (joins, aggregations, etc.)
   - What made you select this intent?
   - What analytical approach is needed?
   - If SQL provided, what patterns/techniques does it demonstrate?

6. **Confidence Score**: Rate 0.0-1.0 how clear and unambiguous the insight is
   - 1.0: Crystal clear intent with specific requirements and data needed
   - 0.7-0.9: Clear analytical goal, may include SQL reference examples
   - 0.4-0.6: Vague intent or missing context
   - 0.0-0.3: Very unclear or ambiguous request

IMPORTANT GUIDELINES:
- SQL examples show INTENT and PATTERNS, not necessarily correct syntax
- Look for patterns: how are tables joined? what calculations are needed?
- Extract the analytical approach even if SQL syntax is wrong
- Multiple patterns are OK - don't limit to one
- If no SQL examples found, return empty array
- Even without SQL, try to understand what type of query is needed
- Extract the analytical intent, not just keywords
- Pseudocode is valuable - it shows user's mental model

Extract the structured information now."""
        
        return prompt
    
    def _log_extraction_results(self, context: InsightContext) -> None:
        """
        Log extraction results for debugging.
        
        Args:
            context: Extracted insight context
        """
        logger.info("=" * 80)
        logger.info("INSIGHT EXTRACTION RESULTS:")
        logger.info("-" * 80)
        
        if context.example_queries:
            logger.info(f"✓ Found {len(context.example_queries)} SQL examples")
            for i, sql in enumerate(context.example_queries, 1):
                preview = sql[:100] + "..." if len(sql) > 100 else sql
                logger.info(f"  Example {i}: {preview}")
        else:
            logger.info("- No SQL examples found")
        
        if context.pattern_keywords:
            logger.info(f"✓ Detected patterns: {', '.join(context.pattern_keywords)}")
        else:
            logger.info("- No specific patterns detected")
        
        if context.inferred_intent:
            logger.info(f"✓ Primary intent: {context.inferred_intent}")
        else:
            logger.info("- Intent unclear")
        
        if context.reasoning:
            logger.info(f"LLM reasoning: {context.reasoning[:200]}...")
        
        logger.info(f"Extraction confidence: {context.confidence:.2f}")
        logger.info("=" * 80)
    
    def _create_fallback_context(self, insight: str) -> InsightContext:
        """
        Create minimal context when LLM extraction fails.
        
        Args:
            insight: Original insight text
            
        Returns:
            Basic InsightContext with no extracted data
        """
        logger.warning("Using fallback context (LLM extraction unavailable)")
        return InsightContext(
            original_text=insight,
            cleaned_text=insight,
            example_queries=[],
            pattern_keywords=[],
            inferred_intent=None,
            reasoning="Fallback context - LLM extraction failed",
            confidence=0.0,
            metadata={"extraction_method": "fallback"}
        )

