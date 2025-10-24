"""
Tests for PRP Parser

Tests parsing of PRP markdown Section 9 to extract target view specifications.
"""

import pytest

from query_generation_agent.parsers.prp_parser import (
    TargetViewSpec,
    extract_table_schema,
    parse_prp_section_9,
)


def test_parse_simple_prp_section() -> None:
    """Test parsing a simple PRP Section 9 with one table."""
    prp_markdown = """
## 9. Data Requirements
### Table: `ensemble_predictions`
**Description:** Contains the model's weekly predictions for NFL games.

**Schema:**
- `game_id` (STRING): A unique identifier for the game.
- `team` (STRING): The team the prediction is for.
- `week` (INTEGER): The week of the game.
- `spread` (FLOAT): The closing point spread.
"""
    
    target_views = parse_prp_section_9(prp_markdown)
    
    assert len(target_views) == 1
    assert target_views[0].view_name == "ensemble_predictions"
    assert "model's weekly predictions" in target_views[0].description
    assert len(target_views[0].columns) == 4
    
    # Check first column
    assert target_views[0].columns[0]["name"] == "game_id"
    assert target_views[0].columns[0]["type"] == "STRING"
    assert "unique identifier" in target_views[0].columns[0]["description"]


def test_parse_multiple_tables() -> None:
    """Test parsing PRP Section 9 with multiple tables."""
    prp_markdown = """
## 9. Data Requirements
### Table: `ensemble_predictions`
**Description:** Contains predictions.

**Schema:**
- `game_id` (STRING): Game identifier.
- `team` (STRING): Team name.

### Table: `game_results`
**Description:** Contains game outcomes.

**Schema:**
- `game_id` (STRING): Game identifier.
- `home_team_score` (INTEGER): Home team score.
- `away_team_score` (INTEGER): Away team score.
"""
    
    target_views = parse_prp_section_9(prp_markdown)
    
    assert len(target_views) == 2
    assert target_views[0].view_name == "ensemble_predictions"
    assert target_views[1].view_name == "game_results"
    assert len(target_views[0].columns) == 2
    assert len(target_views[1].columns) == 3


def test_extract_table_schema() -> None:
    """Test extracting schema from a table section."""
    table_content = """
**Description:** Contains the final scores.

**Schema:**
- `game_id` (STRING): Unique game identifier.
- `score` (INTEGER): Final score.
- `result` (BOOL): Win or loss.
"""
    
    view_spec = extract_table_schema("test_table", table_content)
    
    assert view_spec.view_name == "test_table"
    assert "final scores" in view_spec.description.lower()
    assert len(view_spec.columns) == 3
    assert view_spec.columns[0]["name"] == "game_id"
    assert view_spec.columns[1]["type"] == "INTEGER"
    assert view_spec.columns[2]["name"] == "result"


def test_parse_without_section_9_header() -> None:
    """Test parsing when Section 9 header is not present."""
    prp_markdown = """
### Table: `test_table`
**Description:** Test table.

**Schema:**
- `id` (STRING): Identifier.
"""
    
    # Should still find the table even without Section 9 header
    target_views = parse_prp_section_9(prp_markdown)
    
    assert len(target_views) == 1
    assert target_views[0].view_name == "test_table"


def test_parse_table_without_backticks() -> None:
    """Test parsing table name without backticks."""
    prp_markdown = """
### Table: test_table
**Description:** Test table.

**Schema:**
- `id` (STRING): Identifier.
"""
    
    target_views = parse_prp_section_9(prp_markdown)
    
    assert len(target_views) == 1
    assert target_views[0].view_name == "test_table"


def test_target_view_spec_to_dict() -> None:
    """Test TargetViewSpec.to_dict() method."""
    view_spec = TargetViewSpec(
        view_name="test_view",
        description="Test description",
        columns=[
            {"name": "id", "type": "STRING", "description": "ID field"},
            {"name": "value", "type": "INTEGER", "description": "Value field"}
        ]
    )
    
    result = view_spec.to_dict()
    
    assert result["view_name"] == "test_view"
    assert result["description"] == "Test description"
    assert len(result["columns"]) == 2
    assert result["columns"][0]["name"] == "id"


def test_parse_empty_markdown() -> None:
    """Test parsing empty markdown returns empty list."""
    target_views = parse_prp_section_9("")
    assert len(target_views) == 0


def test_parse_markdown_with_no_schema() -> None:
    """Test parsing table without schema section."""
    prp_markdown = """
### Table: `test_table`
**Description:** Test table without schema.
"""
    
    # Should not create a view spec without schema
    target_views = parse_prp_section_9(prp_markdown)
    
    # Will raise error during extraction, so should be empty
    assert len(target_views) == 0

