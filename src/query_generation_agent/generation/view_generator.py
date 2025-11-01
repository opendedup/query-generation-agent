"""
View Generator

Generates CREATE VIEW DDL statements from target schemas and source tables.
Uses Gemini AI to intelligently map source data to target view schemas.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from ..clients.gemini_client import GeminiClient
from ..models.request_models import DatasetMetadata
from ..parsers.prp_parser import TargetViewSpec

logger = logging.getLogger(__name__)


class ViewGenerator:
    """
    Generates CREATE VIEW DDL statements using LLM.
    
    Takes target view specifications and source table metadata,
    generates SQL DDL that transforms source data into required format.
    """
    
    def __init__(self, gemini_client: GeminiClient):
        """
        Initialize view generator.
        
        Args:
            gemini_client: Gemini client for DDL generation
        """
        self.gemini_client = gemini_client
    
    def generate_view_ddl(
        self,
        target_view: TargetViewSpec,
        source_datasets: List[DatasetMetadata],
        target_location: str = "",
        llm_mode: str = "fast_llm"
    ) -> Tuple[bool, Optional[str], str]:
        """
        Generate CREATE VIEW DDL matching target schema.
        
        Args:
            target_view: Target view specification from PRP
            source_datasets: Available source tables
            target_location: Target location in format project.dataset
            llm_mode: LLM model mode ('fast_llm' or 'detailed_llm')
            
        Returns:
            Tuple of (success, error_message, ddl_statement)
        """
        logger.info(f"Generating VIEW DDL for: {target_view.view_name}")
        
        # Build prompt
        prompt = self._build_view_generation_prompt(
            target_view, source_datasets, target_location
        )
        
        # Generate DDL using Gemini
        success, error_msg, ddl = self.gemini_client.generate_view_ddl(prompt, llm_mode=llm_mode)
        
        if not success:
            logger.error(f"Failed to generate VIEW DDL: {error_msg}")
            return False, error_msg, ""
        
        # Clean up DDL
        ddl = self._clean_ddl(ddl)
        
        logger.info(f"Successfully generated VIEW DDL for {target_view.view_name} ({len(ddl)} chars)")
        return True, None, ddl
    
    def _build_view_generation_prompt(
        self,
        target_view: TargetViewSpec,
        source_datasets: List[DatasetMetadata],
        target_location: str
    ) -> str:
        """
        Build prompt for view DDL generation.
        
        Args:
            target_view: Target view specification
            source_datasets: Available source tables
            target_location: Target location for view
            
        Returns:
            Formatted prompt string
        """
        # Format target schema
        target_schema_text = self._format_target_schema(target_view)
        
        # Format source tables
        source_tables_text = self._format_source_tables(source_datasets)
        
        # Build full view name
        if target_location:
            full_view_name = f"`{target_location}.{target_view.view_name}`"
        else:
            full_view_name = f"`{target_view.view_name}`"
        
        prompt = f"""Generate a CREATE VIEW statement that produces the following target schema.

VIEW NAME: {full_view_name}
DESCRIPTION: {target_view.description}

TARGET SCHEMA (must match exactly):
{target_schema_text}

AVAILABLE SOURCE TABLES:
{source_tables_text}

INSTRUCTIONS:
1. Create a view that produces EXACTLY the target schema columns and types
2. Use appropriate transformations, JOINs, and CAST operations from source tables
3. Handle NULL values and data type conversions properly
4. Add SQL comments explaining complex logic and assumptions
5. Use BigQuery best practices (backticks for identifiers, explicit CAST, etc.)
6. If source data doesn't exist for a column:
   - Use CAST(NULL AS <type>) AS column_name
   - Add a TODO comment explaining what data is needed
7. If source tables don't exist yet:
   - Reference them anyway with proper syntax
   - Add TODO comments about required table creation
8. Include helpful comments for maintainability

IMPORTANT:
- The view MUST have the exact column names and types from TARGET SCHEMA
- Column order should match the target schema
- Use explicit CAST() for all type conversions
- Add inline comments for business logic

Generate the complete CREATE OR REPLACE VIEW statement now:"""
        
        return prompt
    
    def _format_target_schema(self, target_view: TargetViewSpec) -> str:
        """
        Format target schema for prompt.
        
        Args:
            target_view: Target view specification
            
        Returns:
            Formatted schema string
        """
        lines = []
        for col in target_view.columns:
            name = col.get("name", "unknown")
            col_type = col.get("type", "STRING")
            desc = col.get("description", "")
            
            line = f"- {name} ({col_type})"
            if desc:
                line += f": {desc}"
            lines.append(line)
        
        return "\n".join(lines)
    
    def _format_source_tables(self, source_datasets: List[DatasetMetadata]) -> str:
        """
        Format source tables for prompt.
        
        Args:
            source_datasets: Available source tables
            
        Returns:
            Formatted source tables string
        """
        if not source_datasets:
            return "No source tables provided - generate placeholder view with NULLs"
        
        tables = []
        for ds in source_datasets:
            table_id = ds.get_full_table_id()
            desc = ds.description or "No description"
            row_count = f"{ds.row_count:,}" if ds.row_count else "unknown"
            
            # Format schema
            schema_lines = []
            for field in ds.schema[:20]:  # Limit to first 20 fields
                field_name = field.get("name", "unknown")
                field_type = field.get("type", "unknown")
                field_desc = field.get("description", "")
                
                schema_line = f"  - {field_name} ({field_type})"
                if field_desc:
                    schema_line += f": {field_desc}"
                schema_lines.append(schema_line)
            
            if len(ds.schema) > 20:
                schema_lines.append(f"  ... and {len(ds.schema) - 20} more columns")
            
            schema_text = "\n".join(schema_lines) if schema_lines else "  (schema not available)"
            
            table_section = f"""
Table: `{table_id}`
Description: {desc}
Rows: {row_count}
Schema:
{schema_text}
"""
            tables.append(table_section.strip())
        
        return "\n\n".join(tables)
    
    def _clean_ddl(self, ddl: str) -> str:
        """
        Clean up generated DDL statement.
        
        Args:
            ddl: Raw DDL from LLM
            
        Returns:
            Cleaned DDL string
        """
        # Remove markdown code blocks if present
        if ddl.startswith("```"):
            lines = ddl.split("\n")
            # Remove first and last lines (markdown fences)
            ddl = "\n".join(lines[1:-1]) if len(lines) > 2 else ddl
        
        # Remove language identifier after opening fence
        ddl = ddl.replace("```sql\n", "").replace("```\n", "").replace("```", "")
        
        # Ensure it starts with CREATE
        ddl = ddl.strip()
        if not ddl.upper().startswith("CREATE"):
            logger.warning("Generated DDL doesn't start with CREATE, attempting to fix")
            # Try to find CREATE statement in the text
            create_idx = ddl.upper().find("CREATE")
            if create_idx != -1:
                ddl = ddl[create_idx:]
        
        return ddl.strip()

