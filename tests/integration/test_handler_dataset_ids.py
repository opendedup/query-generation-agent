"""Integration tests for handler with dataset_ids."""

import json
from typing import TYPE_CHECKING, Any, Dict

import pytest

if TYPE_CHECKING:
    from pytest_mock.plugin import MockerFixture

from query_generation_agent.mcp.handlers import MCPHandlers
from query_generation_agent.mcp.config import QueryGenerationConfig
from query_generation_agent.clients.bigquery_client import BigQueryClient
from query_generation_agent.clients.gemini_client import GeminiClient


@pytest.fixture
def mock_config() -> QueryGenerationConfig:
    """Create a mock configuration for testing."""
    # Use a dictionary to create config with minimal required fields
    return QueryGenerationConfig(
        project_id="test-project",
        bq_execution_project="test-project",
        gemini_api_key="test-key",
        discovery_agent_url="http://localhost:8080"
    )


@pytest.fixture
def mock_bigquery_client(mocker: "MockerFixture") -> BigQueryClient:
    """Create a mock BigQuery client."""
    return mocker.MagicMock(spec=BigQueryClient)


@pytest.fixture
def mock_gemini_client(mocker: "MockerFixture") -> GeminiClient:
    """Create a mock Gemini client."""
    return mocker.MagicMock(spec=GeminiClient)


@pytest.fixture
def handlers(
    mock_config: QueryGenerationConfig,
    mock_bigquery_client: BigQueryClient,
    mock_gemini_client: GeminiClient
) -> MCPHandlers:
    """Create MCPHandlers instance for testing."""
    return MCPHandlers(
        config=mock_config,
        bigquery_client=mock_bigquery_client,
        gemini_client=mock_gemini_client
    )


@pytest.fixture
def mock_dataset() -> Dict[str, Any]:
    """Create a mock dataset response."""
    return {
        "table_id": "test_table",
        "project_id": "test_project",
        "dataset_id": "test_dataset",
        "description": "Test table",
        "asset_type": "table",
        "schema": [
            {
                "name": "id",
                "type": "STRING",
                "mode": "NULLABLE",
                "description": "ID"
            }
        ],
        "row_count": 1000,
        "size_bytes": 5000,
        "column_count": 1,
        "has_pii": False,
        "has_phi": False,
        "environment": "PROD"
    }


@pytest.mark.asyncio
async def test_handle_generate_queries_with_dataset_ids(
    handlers: MCPHandlers,
    mock_dataset: Dict[str, Any],
    mocker: "MockerFixture"
) -> None:
    """Test handle_generate_queries with dataset_ids parameter."""
    # Prepare arguments with dataset_ids
    arguments = {
        "insight": "Show me the top 10 records by value",
        "dataset_ids": [
            {
                "project_id": "test_project",
                "dataset_id": "test_dataset",
                "table_id": "test_table"
            }
        ],
        "max_queries": 1,
        "max_iterations": 1
    }
    
    # Mock DiscoveryClient
    from query_generation_agent.clients.discovery_client import DiscoveryClient
    
    mock_discovery_client = mocker.MagicMock(spec=DiscoveryClient)
    mock_discovery_client.get_multiple_datasets_by_ids = mocker.AsyncMock(
        return_value=[mock_dataset]
    )
    mock_discovery_client.close = mocker.AsyncMock()
    
    # Patch DiscoveryClient constructor
    mocker.patch(
        "query_generation_agent.mcp.handlers.DiscoveryClient",
        return_value=mock_discovery_client
    )
    
    # Mock _parse_request to avoid validation errors
    mock_request = mocker.MagicMock()
    mock_request.insight = "Show me the top 10 records by value"
    mock_request.datasets = []  # Empty for this test
    mock_request.max_queries = 1
    mock_request.max_iterations = 1
    mock_request.llm_mode = "fast_llm"
    mock_request.stop_on_first_valid = True
    mock_request.target_table_name = None
    
    mocker.patch.object(handlers, "_parse_request", return_value=mock_request)
    
    # Mock query generation components to return error (to keep test simple)
    mock_planner = mocker.MagicMock()
    mock_planner.plan_query = mocker.MagicMock(return_value=(False, "Test error", None, {}))
    
    mocker.patch(
        "query_generation_agent.mcp.handlers.QueryPlanner",
        return_value=mock_planner
    )
    
    # Call the handler
    result = await handlers.handle_generate_queries(arguments)
    
    # Verify that DiscoveryClient was called
    mock_discovery_client.get_multiple_datasets_by_ids.assert_called_once()
    call_args = mock_discovery_client.get_multiple_datasets_by_ids.call_args
    assert call_args[1]["dataset_ids"] == arguments["dataset_ids"]
    
    # Verify close was called
    mock_discovery_client.close.assert_called_once()
    
    # Verify result is a list with TextContent
    assert len(result) == 1
    assert hasattr(result[0], "text")


@pytest.mark.asyncio
async def test_handle_generate_queries_dataset_fetch_error(
    handlers: MCPHandlers,
    mocker: "MockerFixture"
) -> None:
    """Test handle_generate_queries when dataset fetch fails."""
    arguments = {
        "insight": "Show me the data",
        "dataset_ids": [
            {
                "project_id": "test_project",
                "dataset_id": "test_dataset",
                "table_id": "nonexistent_table"
            }
        ]
    }
    
    # Mock DiscoveryClient to raise error
    from query_generation_agent.clients.discovery_client import DiscoveryClient
    
    mock_discovery_client = mocker.MagicMock(spec=DiscoveryClient)
    mock_discovery_client.get_multiple_datasets_by_ids = mocker.AsyncMock(
        side_effect=ValueError("Dataset not found")
    )
    mock_discovery_client.close = mocker.AsyncMock()
    
    mocker.patch(
        "query_generation_agent.mcp.handlers.DiscoveryClient",
        return_value=mock_discovery_client
    )
    
    # Call the handler
    result = await handlers.handle_generate_queries(arguments)
    
    # Verify error response
    assert len(result) == 1
    response_text = result[0].text
    response_data = json.loads(response_text)
    
    assert "error" in response_data
    assert "Failed to fetch datasets" in response_data["error"]
    assert response_data["total_attempted"] == 0
    assert response_data["total_validated"] == 0
    
    # Verify close was called
    mock_discovery_client.close.assert_called_once()

