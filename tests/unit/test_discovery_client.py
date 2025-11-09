"""Tests for DiscoveryClient."""

import json
from typing import TYPE_CHECKING

import pytest
import httpx

if TYPE_CHECKING:
    from pytest_mock.plugin import MockerFixture

from query_generation_agent.clients.discovery_client import DiscoveryClient


@pytest.fixture
def discovery_client() -> DiscoveryClient:
    """Create a DiscoveryClient instance for testing."""
    return DiscoveryClient(base_url="http://localhost:8080", timeout=10.0)


@pytest.fixture
def mock_dataset() -> dict:
    """Create a mock dataset response."""
    return {
        "table_id": "test_table",
        "project_id": "test_project",
        "dataset_id": "test_dataset",
        "description": "Test table description",
        "asset_type": "table",
        "schema": [
            {
                "name": "id",
                "type": "STRING",
                "mode": "NULLABLE",
                "description": "Unique identifier",
                "sample_values": ["id1", "id2", "id3"]
            },
            {
                "name": "value",
                "type": "INTEGER",
                "mode": "NULLABLE",
                "description": "Some value",
                "sample_values": ["100", "200", "300"]
            }
        ],
        "row_count": 1000,
        "size_bytes": 5000000,
        "column_count": 2,
        "has_pii": False,
        "has_phi": False,
        "environment": "PROD",
        "column_profiles": [],
        "lineage": [],
        "analytical_insights": []
    }


@pytest.mark.asyncio
async def test_get_datasets_for_query_generation(
    discovery_client: DiscoveryClient,
    mock_dataset: dict,
    mocker: "MockerFixture"
) -> None:
    """Test get_datasets_for_query_generation method."""
    # Mock the _call_tool method
    mock_response = {
        "result": [{
            "text": json.dumps({
                "discovered_assets": [mock_dataset],
                "total_count": 1
            })
        }]
    }
    mocker.patch.object(discovery_client, "_call_tool", return_value=mock_response)
    
    # Call the method
    datasets = await discovery_client.get_datasets_for_query_generation(
        query="test table",
        project_id="test_project",
        dataset_id="test_dataset"
    )
    
    # Verify results
    assert len(datasets) == 1
    assert datasets[0]["table_id"] == "test_table"
    assert datasets[0]["project_id"] == "test_project"
    assert datasets[0]["dataset_id"] == "test_dataset"


@pytest.mark.asyncio
async def test_get_dataset_by_id(
    discovery_client: DiscoveryClient,
    mock_dataset: dict,
    mocker: "MockerFixture"
) -> None:
    """Test get_dataset_by_id method."""
    # Mock get_datasets_for_query_generation
    mocker.patch.object(
        discovery_client,
        "get_datasets_for_query_generation",
        return_value=[mock_dataset]
    )
    
    # Call the method
    dataset = await discovery_client.get_dataset_by_id(
        project_id="test_project",
        dataset_id="test_dataset",
        table_id="test_table"
    )
    
    # Verify results
    assert dataset is not None
    assert dataset["table_id"] == "test_table"
    assert dataset["project_id"] == "test_project"


@pytest.mark.asyncio
async def test_get_dataset_by_id_not_found(
    discovery_client: DiscoveryClient,
    mocker: "MockerFixture"
) -> None:
    """Test get_dataset_by_id when dataset is not found."""
    # Mock get_datasets_for_query_generation to return empty list
    mocker.patch.object(
        discovery_client,
        "get_datasets_for_query_generation",
        return_value=[]
    )
    
    # Call the method
    dataset = await discovery_client.get_dataset_by_id(
        project_id="test_project",
        dataset_id="test_dataset",
        table_id="nonexistent_table"
    )
    
    # Verify result is None
    assert dataset is None


@pytest.mark.asyncio
async def test_get_multiple_datasets_by_ids(
    discovery_client: DiscoveryClient,
    mock_dataset: dict,
    mocker: "MockerFixture"
) -> None:
    """Test get_multiple_datasets_by_ids method with successful fetch."""
    dataset_ids = [
        {"project_id": "p1", "dataset_id": "d1", "table_id": "t1"},
        {"project_id": "p2", "dataset_id": "d2", "table_id": "t2"}
    ]
    
    # Create mock datasets
    mock_dataset1 = {**mock_dataset, "table_id": "t1", "project_id": "p1", "dataset_id": "d1"}
    mock_dataset2 = {**mock_dataset, "table_id": "t2", "project_id": "p2", "dataset_id": "d2"}
    
    # Mock get_dataset_by_id to return different datasets
    call_count = [0]
    
    async def mock_get_dataset_by_id(project_id: str, dataset_id: str, table_id: str):
        result = mock_dataset1 if call_count[0] == 0 else mock_dataset2
        call_count[0] += 1
        return result
    
    mocker.patch.object(
        discovery_client,
        "get_dataset_by_id",
        side_effect=mock_get_dataset_by_id
    )
    
    # Call the method
    datasets = await discovery_client.get_multiple_datasets_by_ids(dataset_ids)
    
    # Verify results
    assert len(datasets) == 2
    assert datasets[0]["table_id"] == "t1"
    assert datasets[1]["table_id"] == "t2"


@pytest.mark.asyncio
async def test_get_multiple_datasets_by_ids_with_errors(
    discovery_client: DiscoveryClient,
    mocker: "MockerFixture"
) -> None:
    """Test get_multiple_datasets_by_ids with some datasets not found."""
    dataset_ids = [
        {"project_id": "p1", "dataset_id": "d1", "table_id": "t1"},
        {"project_id": "p2", "dataset_id": "d2", "table_id": "t2"}
    ]
    
    # Mock get_dataset_by_id: first returns dataset, second returns None
    call_count = [0]
    
    async def mock_get_dataset_by_id(project_id: str, dataset_id: str, table_id: str):
        if call_count[0] == 0:
            call_count[0] += 1
            return {"table_id": "t1", "project_id": "p1", "dataset_id": "d1"}
        else:
            call_count[0] += 1
            return None  # Not found
    
    mocker.patch.object(
        discovery_client,
        "get_dataset_by_id",
        side_effect=mock_get_dataset_by_id
    )
    
    # Call the method and expect ValueError
    with pytest.raises(ValueError) as exc_info:
        await discovery_client.get_multiple_datasets_by_ids(dataset_ids)
    
    # Verify error message
    assert "Failed to fetch 1 dataset(s)" in str(exc_info.value)
    assert "p2.d2.t2" in str(exc_info.value)


@pytest.mark.asyncio
async def test_query_data_assets(
    discovery_client: DiscoveryClient,
    mocker: "MockerFixture"
) -> None:
    """Test query_data_assets method."""
    # Mock the _call_tool method
    mock_response = {
        "result": [{
            "text": "# Search Results\n\nFound 1 table"
        }]
    }
    mocker.patch.object(discovery_client, "_call_tool", return_value=mock_response)
    
    # Call the method
    result = await discovery_client.query_data_assets(
        query="test tables",
        filters={"has_pii": True}
    )
    
    # Verify result
    assert "Search Results" in result
    assert "Found 1 table" in result


@pytest.mark.asyncio
async def test_call_tool_http_error(
    discovery_client: DiscoveryClient,
    mocker: "MockerFixture"
) -> None:
    """Test _call_tool with HTTP error."""
    # Mock httpx to raise an error
    mock_post = mocker.patch.object(discovery_client.client, "post")
    mock_post.side_effect = httpx.HTTPError("Connection failed")
    
    # Call should raise HTTPError
    with pytest.raises(httpx.HTTPError):
        await discovery_client._call_tool("test_tool", {})


@pytest.mark.asyncio
async def test_close(discovery_client: DiscoveryClient, mocker: "MockerFixture") -> None:
    """Test close method."""
    mock_aclose = mocker.patch.object(discovery_client.client, "aclose")
    
    await discovery_client.close()
    
    mock_aclose.assert_called_once()

