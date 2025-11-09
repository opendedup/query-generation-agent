"""HTTP client for the Data Discovery Agent MCP service."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class DiscoveryClient:
    """Client for interacting with the Data Discovery Agent via HTTP."""

    def __init__(self, base_url: str, timeout: float = 300.0):
        """Initialize the discovery client.
        
        Args:
            base_url: Base URL for the discovery agent (e.g., http://localhost:8080)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    async def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool via HTTP.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool response as dictionary
            
        Raises:
            ValueError: If the request fails with detailed error information
        """
        url = f"{self.base_url}/mcp/call-tool"
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            },
            "id": 1
        }
        
        logger.info(f"Calling discovery tool: {tool_name}")
        logger.info(f"Request URL: {url}")
        logger.info(f"Request payload: {json.dumps(payload, indent=2)}")
        
        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            logger.debug(f"Response: {result}")
            return result
            
        except httpx.ConnectError as e:
            error_msg = (
                f"Connection error calling discovery tool '{tool_name}': "
                f"Unable to connect to {self.base_url}. "
                f"Please verify the data-discovery-agent is running and accessible. "
                f"Details: {str(e)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg) from e
            
        except httpx.TimeoutException as e:
            error_msg = (
                f"Timeout error calling discovery tool '{tool_name}': "
                f"Request to {url} timed out after {self.timeout}s. "
                f"The discovery agent may be overloaded or unresponsive. "
                f"Details: {str(e)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg) from e
            
        except httpx.HTTPStatusError as e:
            # Try to extract detailed error from response
            error_detail = "Unknown error"
            try:
                error_body = e.response.json()
                if isinstance(error_body, dict):
                    # Try different error formats
                    if "detail" in error_body:
                        error_detail = error_body["detail"]
                    elif "error" in error_body:
                        if isinstance(error_body["error"], dict):
                            error_detail = error_body["error"].get("message", str(error_body["error"]))
                        else:
                            error_detail = str(error_body["error"])
                    elif "message" in error_body:
                        error_detail = error_body["message"]
                    else:
                        error_detail = json.dumps(error_body, indent=2)
            except Exception:
                # If JSON parsing fails, use raw text
                error_detail = e.response.text[:500]  # Limit to 500 chars
            
            if e.response.status_code == 400:
                error_msg = (
                    f"Invalid request to discovery tool '{tool_name}' (HTTP 400): {error_detail}\n\n"
                    f"This usually means the arguments are invalid or missing required fields.\n"
                    f"Arguments sent: {json.dumps(arguments, indent=2)}"
                )
            elif e.response.status_code == 404:
                error_msg = (
                    f"Discovery tool '{tool_name}' not found (HTTP 404): {error_detail}\n\n"
                    f"The tool may not be registered or the endpoint URL is incorrect.\n"
                    f"URL: {url}"
                )
            elif e.response.status_code == 500:
                error_msg = (
                    f"Internal server error in discovery tool '{tool_name}' (HTTP 500): {error_detail}\n\n"
                    f"The discovery agent encountered an error while processing the request.\n"
                    f"Check the data-discovery-agent logs for more details."
                )
            elif e.response.status_code == 503:
                error_msg = (
                    f"Discovery service unavailable (HTTP 503): {error_detail}\n\n"
                    f"The data-discovery-agent may be starting up or experiencing issues."
                )
            else:
                error_msg = (
                    f"HTTP {e.response.status_code} error calling discovery tool '{tool_name}': {error_detail}\n\n"
                    f"URL: {url}"
                )
            
            logger.error(error_msg)
            logger.error(f"Full response: {e.response.text}")
            raise ValueError(error_msg) from e
            
        except json.JSONDecodeError as e:
            error_msg = (
                f"Invalid JSON response from discovery tool '{tool_name}': "
                f"The response could not be parsed as JSON. "
                f"This may indicate a server error or misconfiguration. "
                f"Details: {str(e)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg) from e
            
        except Exception as e:
            error_msg = (
                f"Unexpected error calling discovery tool '{tool_name}': {str(e)}\n\n"
                f"URL: {url}\n"
                f"Error type: {type(e).__name__}"
            )
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg) from e

    async def get_datasets_for_query_generation(
        self,
        query: str,
        project_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        page_size: int = 10
    ) -> List[Dict[str, Any]]:
        """Get dataset metadata in DiscoveredAssetDict format for query generation.
        
        Args:
            query: Natural language search query
            project_id: Optional filter by GCP project ID
            dataset_id: Optional filter by BigQuery dataset ID
            page_size: Maximum number of results (default: 10)
            
        Returns:
            List of discovered datasets with full metadata including:
            - table_id, project_id, dataset_id, table_name
            - description, asset_type
            - schema (with sample_values)
            - row_count, size_bytes, column_count
            - column_profiles, lineage, analytical_insights
            - has_pii, has_phi, environment
            
        Raises:
            ValueError: If the request fails with detailed error information
            
        Example:
            >>> datasets = await client.get_datasets_for_query_generation(
            ...     query="customer analytics tables",
            ...     project_id="my-project"
            ... )
            >>> for ds in datasets:
            ...     print(f"Table: {ds['table_id']}")
            ...     print(f"Schema: {len(ds['schema'])} columns")
        """
        logger.info(f"Fetching datasets for query generation: {query}")
        
        arguments = {
            "query": query,
            "output_format": "json",
            "page_size": page_size
        }
        
        if project_id:
            arguments["project_id"] = project_id
        if dataset_id:
            arguments["dataset_id"] = dataset_id
        
        try:
            result = await self._call_tool("get_datasets_for_query_generation", arguments)
        except ValueError as e:
            # Re-raise with additional context
            raise ValueError(
                f"Failed to fetch datasets for query generation.\n"
                f"Query: {query}\n"
                f"Filters: project_id={project_id}, dataset_id={dataset_id}\n\n"
                f"Error: {str(e)}"
            ) from e
        
        # Parse JSON response from TextContent
        try:
            if "result" in result:
                # Extract text from MCP TextContent response
                if isinstance(result["result"], list) and len(result["result"]) > 0:
                    text_content = result["result"][0].get("text", "{}")
                    data = json.loads(text_content)
                    discovered_assets = data.get("discovered_assets", [])
                    
                    logger.info(f"Fetched {len(discovered_assets)} dataset(s)")
                    return discovered_assets
            
            logger.warning("No datasets found in response")
            return []
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            error_msg = (
                f"Failed to parse response from discovery agent.\n"
                f"Query: {query}\n"
                f"Response structure may be invalid.\n"
                f"Error: {str(e)}\n"
                f"Response: {json.dumps(result, indent=2)[:500]}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg) from e
    
    async def get_dataset_by_id(
        self,
        project_id: str,
        dataset_id: str,
        table_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get a specific dataset by its fully qualified ID.
        
        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            
        Returns:
            Dataset metadata dictionary in DiscoveredAssetDict format,
            or None if not found
            
        Example:
            >>> dataset = await client.get_dataset_by_id(
            ...     project_id="my-project",
            ...     dataset_id="analytics",
            ...     table_id="customers"
            ... )
            >>> if dataset:
            ...     print(f"Found: {dataset['table_id']}")
        """
        logger.info(f"Fetching dataset: {project_id}.{dataset_id}.{table_id}")
        
        # Use table_id as query with exact project/dataset filters
        datasets = await self.get_datasets_for_query_generation(
            query=table_id,
            project_id=project_id,
            dataset_id=dataset_id,
            page_size=1
        )
        
        if datasets:
            # Verify exact match
            dataset = datasets[0]
            if (dataset.get("project_id") == project_id and
                dataset.get("dataset_id") == dataset_id and
                dataset.get("table_id") == table_id):
                logger.info(f"Found exact match: {project_id}.{dataset_id}.{table_id}")
                return dataset
            else:
                logger.warning(
                    f"Found dataset but IDs don't match exactly: "
                    f"{dataset.get('project_id')}.{dataset.get('dataset_id')}.{dataset.get('table_id')}"
                )
        
        logger.warning(f"Dataset not found: {project_id}.{dataset_id}.{table_id}")
        return None
    
    async def get_multiple_datasets_by_ids(
        self,
        dataset_ids: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Get multiple datasets by their IDs (parallel fetch).
        
        Args:
            dataset_ids: List of dicts with 'project_id', 'dataset_id', 'table_id' keys
            
        Returns:
            List of dataset metadata dictionaries (in same order as input)
            
        Raises:
            ValueError: If any dataset is not found
            
        Example:
            >>> dataset_ids = [
            ...     {"project_id": "p1", "dataset_id": "d1", "table_id": "t1"},
            ...     {"project_id": "p2", "dataset_id": "d2", "table_id": "t2"}
            ... ]
            >>> datasets = await client.get_multiple_datasets_by_ids(dataset_ids)
        """
        logger.info(f"Fetching {len(dataset_ids)} datasets in parallel")
        
        # Fetch all datasets in parallel
        tasks = [
            self.get_dataset_by_id(
                project_id=ds["project_id"],
                dataset_id=ds["dataset_id"],
                table_id=ds["table_id"]
            )
            for ds in dataset_ids
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check for errors and None values
        datasets = []
        errors = []
        
        for i, (ds_id, result) in enumerate(zip(dataset_ids, results)):
            dataset_fqn = f"{ds_id['project_id']}.{ds_id['dataset_id']}.{ds_id['table_id']}"
            
            if isinstance(result, Exception):
                # Extract the error message, keeping the detailed information
                error_str = str(result)
                errors.append(f"Error fetching {dataset_fqn}:\n{error_str}")
                logger.error(f"Failed to fetch {dataset_fqn}: {error_str}")
            elif result is None:
                error_msg = f"Dataset not found: {dataset_fqn}"
                errors.append(error_msg)
                logger.warning(error_msg)
            else:
                datasets.append(result)
        
        if errors:
            # Format errors with numbering for clarity
            formatted_errors = "\n\n".join([f"{i+1}. {err}" for i, err in enumerate(errors)])
            error_msg = (
                f"Failed to fetch {len(errors)} of {len(dataset_ids)} dataset(s):\n\n"
                f"{formatted_errors}\n\n"
                f"Successfully fetched: {len(datasets)} dataset(s)"
            )
            logger.error(f"Failed to fetch some datasets: {error_msg}")
            raise ValueError(error_msg)
        
        logger.info(f"Successfully fetched all {len(datasets)} datasets")
        return datasets
    
    async def query_data_assets(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        output_format: str = "markdown"
    ) -> str:
        """Query data assets using natural language (human-readable output).
        
        Args:
            query: Natural language search query
            filters: Optional filters (project_id, dataset_id, has_pii, etc.)
            output_format: "markdown" or "json" (default: "markdown")
            
        Returns:
            Markdown or JSON formatted search results
            
        Example:
            >>> results = await client.query_data_assets(
            ...     query="tables with customer data",
            ...     filters={"has_pii": True}
            ... )
            >>> print(results)  # Markdown formatted results
        """
        arguments = {
            "query": query,
            "output_format": output_format
        }
        
        if filters:
            arguments.update(filters)
        
        result = await self._call_tool("query_data_assets", arguments)
        
        # Extract text from MCP TextContent response
        if "result" in result:
            if isinstance(result["result"], list) and len(result["result"]) > 0:
                return result["result"][0].get("text", "No results found")
        
        return "No results found"

