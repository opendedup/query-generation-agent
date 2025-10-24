"""
PRP Parser

Parses Product Requirement Prompts (PRPs) to extract data requirements.
Specifically handles Section 9: Data Requirements with table schemas.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TargetViewSpec(BaseModel):
    """
    Target view specification extracted from PRP.
    
    Contains the expected schema and metadata for a view to be created.
    """
    
    view_name: str = Field(..., description="Name of the view/table")
    description: str = Field(..., description="Purpose/description of the view")
    columns: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of columns with name, type, and description"
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "view_name": self.view_name,
            "description": self.description,
            "columns": self.columns
        }


def parse_prp_section_9(prp_markdown: str) -> List[TargetViewSpec]:
    """
    Extract data requirements from PRP Section 9.
    
    Parses markdown to find table definitions with schemas and converts
    them into structured TargetViewSpec objects.
    
    Args:
        prp_markdown: Full PRP markdown or just Section 9 content
        
    Returns:
        List of TargetViewSpec objects representing target views
        
    Example markdown format:
        ## 9. Data Requirements
        ### Table: `ensemble_predictions`
        **Description:** Contains the model's weekly predictions...
        
        **Schema:**
        - `game_id` (STRING): A unique identifier...
        - `team` (STRING): The team the prediction is for.
    """
    logger.info("Parsing PRP markdown for Section 9 data requirements")
    
    # Extract Section 9 if full PRP provided
    section_9 = _extract_section_9(prp_markdown)
    if not section_9:
        logger.warning("Could not find Section 9 in PRP markdown, using full content")
        section_9 = prp_markdown
    
    # Find all table definitions
    target_views = []
    
    # Pattern to match table headers: ### Table: `table_name` or ### Table: table_name
    table_pattern = r'###\s+Table:\s+`?([a-zA-Z0-9_]+)`?'
    
    # Split by table headers
    table_matches = list(re.finditer(table_pattern, section_9))
    
    for i, match in enumerate(table_matches):
        table_name = match.group(1)
        
        # Get content between this table and the next (or end of section)
        start_pos = match.end()
        end_pos = table_matches[i + 1].start() if i + 1 < len(table_matches) else len(section_9)
        table_content = section_9[start_pos:end_pos]
        
        # Extract table info
        try:
            view_spec = extract_table_schema(table_name, table_content)
            target_views.append(view_spec)
            logger.info(f"Parsed table: {table_name} with {len(view_spec.columns)} columns")
        except Exception as e:
            logger.warning(f"Failed to parse table {table_name}: {e}")
            continue
    
    logger.info(f"Successfully parsed {len(target_views)} target view specifications")
    return target_views


def extract_table_schema(table_name: str, markdown_section: str) -> TargetViewSpec:
    """
    Parse markdown table schema into structured format.
    
    Extracts description and schema from markdown content for a single table.
    
    Args:
        table_name: Name of the table
        markdown_section: Markdown content for this table
        
    Returns:
        TargetViewSpec with parsed schema
        
    Raises:
        ValueError: If schema cannot be parsed
    """
    # Extract description
    description = _extract_description(markdown_section)
    
    # Extract schema fields
    columns = _extract_schema_fields(markdown_section)
    
    if not columns:
        raise ValueError(f"No schema fields found for table {table_name}")
    
    return TargetViewSpec(
        view_name=table_name,
        description=description,
        columns=columns
    )


def _extract_section_9(markdown: str) -> Optional[str]:
    """
    Extract Section 9: Data Requirements from PRP markdown.
    
    Args:
        markdown: Full PRP markdown
        
    Returns:
        Section 9 content or None if not found
    """
    # Try different section header patterns
    patterns = [
        r'##\s+9\.\s+Data Requirements(.*?)(?=##\s+\d+\.|$)',
        r'##\s+Data Requirements(.*?)(?=##\s+[A-Za-z]|$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, markdown, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(0)
    
    return None


def _extract_description(markdown_section: str) -> str:
    """
    Extract table description from markdown section.
    
    Args:
        markdown_section: Markdown content for a table
        
    Returns:
        Description string
    """
    # Look for **Description:** pattern
    desc_pattern = r'\*\*Description:\*\*\s*(.+?)(?=\n\n|\*\*|$)'
    match = re.search(desc_pattern, markdown_section, re.DOTALL)
    
    if match:
        description = match.group(1).strip()
        # Clean up newlines and extra whitespace
        description = re.sub(r'\s+', ' ', description)
        return description
    
    # Fallback: use first paragraph
    paragraphs = markdown_section.strip().split('\n\n')
    if paragraphs:
        first_para = paragraphs[0].strip()
        # Remove markdown formatting
        first_para = re.sub(r'\*\*([^*]+)\*\*', r'\1', first_para)
        return first_para[:200]  # Limit length
    
    return "No description provided"


def _extract_schema_fields(markdown_section: str) -> List[Dict[str, str]]:
    """
    Extract schema fields from markdown section.
    
    Parses field definitions in format:
    - `field_name` (TYPE): Description
    
    Args:
        markdown_section: Markdown content for a table
        
    Returns:
        List of field dictionaries with name, type, description
    """
    columns = []
    
    # Find the **Schema:** section
    schema_match = re.search(
        r'\*\*Schema:\*\*\s*(.*?)(?=\n\*\*|\n###|\Z)',
        markdown_section,
        re.DOTALL | re.IGNORECASE
    )
    
    if not schema_match:
        logger.warning("No **Schema:** section found")
        return columns
    
    schema_content = schema_match.group(1)
    
    # Pattern to match field definitions: - `field_name` (TYPE): Description
    field_pattern = r'-\s+`([a-zA-Z0-9_]+)`\s+\(([A-Z0-9]+)\):\s*(.+?)(?=\n-|\n\n|\Z)'
    
    for match in re.finditer(field_pattern, schema_content, re.DOTALL):
        field_name = match.group(1).strip()
        field_type = match.group(2).strip()
        field_desc = match.group(3).strip()
        
        # Clean up description (remove extra whitespace, newlines)
        field_desc = re.sub(r'\s+', ' ', field_desc)
        
        columns.append({
            "name": field_name,
            "type": field_type,
            "description": field_desc
        })
    
    return columns

