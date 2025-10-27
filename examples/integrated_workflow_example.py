"""
Integrated Workflow Example: Planning → Discovery → Query Generation → GraphQL API

This example demonstrates the complete end-to-end workflow:
1. Gather requirements using data-planning-agent with interactive Q&A
2. Generate a structured Data Product Requirement Prompt (PRP)
3. Extract Section 9 data requirements and discover source tables
4. Generate SELECT queries for each target table specification
5. Generate Apollo GraphQL Server from validated queries
6. Save all results as files in output directory

Prerequisites:
- data-planning-agent running on http://localhost:8082 (not needed if using --prp-file)
- data-discovery-agent running on http://localhost:8080
- query-generation-agent running on http://localhost:8081
- data-graphql-agent running on http://localhost:8083

Usage:
    # Use default initial intent (interactive Q&A)
    python integrated_workflow_example.py
    
    # Specify custom initial intent
    python integrated_workflow_example.py \
        --initial-intent "Analyze customer transaction patterns"
    
    # Load existing PRP from file (skip planning)
    python integrated_workflow_example.py \
        --prp-file output/prp_20250101_120000.md
    
    # Load query results and generate GraphQL only (skip all previous steps)
    python integrated_workflow_example.py \
        --query-results-file output/backtest_evaluation_view_20251024_190934.json \
        --project-name backtest-analysis
    
    # Control query generation parameters
    python integrated_workflow_example.py \
        --max-queries 5 \
        --max-iterations 15
    
    # Skip GraphQL generation
    python integrated_workflow_example.py \
        --skip-graphql
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import httpx


class IntegratedMCPClient:
    """Client for integrated planning, data discovery, query generation, and GraphQL API workflow."""
    
    def __init__(
        self,
        planning_url: str = "http://localhost:8082",
        discovery_url: str = "http://localhost:8080",
        query_gen_url: str = "http://localhost:8081",
        graphql_url: str = "http://localhost:8083",
        timeout: float = 600.0,
        max_query_wait_seconds: float = 900.0,
        output_dir: str = "output"
    ):
        """
        Initialize integrated MCP client.
        
        Args:
            planning_url: Data planning agent base URL
            discovery_url: Data discovery agent base URL
            query_gen_url: Query generation agent base URL
            graphql_url: Data GraphQL agent base URL
            timeout: Request timeout in seconds
            max_query_wait_seconds: Maximum time to wait for query generation (default: 900s = 15 minutes)
            output_dir: Directory to save output files (default: "output")
        """
        self.planning_url = planning_url
        self.discovery_url = discovery_url
        self.query_gen_url = query_gen_url
        self.graphql_url = graphql_url
        self.timeout = timeout
        self.max_query_wait_seconds = max_query_wait_seconds
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)
    
    def load_prp_from_file(self, prp_file: str) -> str:
        """
        Load PRP content from a file.
        
        Args:
            prp_file: Path to PRP markdown file
            
        Returns:
            PRP content as string
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is empty
        """
        prp_path = Path(prp_file)
        if not prp_path.exists():
            raise FileNotFoundError(f"PRP file not found: {prp_file}")
        
        prp_text = prp_path.read_text()
        if not prp_text.strip():
            raise ValueError(f"PRP file is empty: {prp_file}")
        
        return prp_text
    
    def load_query_results_from_file(self, query_results_file: str) -> Dict[str, Any]:
        """
        Load query generation results from a JSON file.
        
        Args:
            query_results_file: Path to query results JSON file
            
        Returns:
            Query results dictionary with queries and metadata
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is empty or invalid JSON
        """
        results_path = Path(query_results_file)
        if not results_path.exists():
            raise FileNotFoundError(f"Query results file not found: {query_results_file}")
        
        results_text = results_path.read_text()
        if not results_text.strip():
            raise ValueError(f"Query results file is empty: {query_results_file}")
        
        try:
            results = json.loads(results_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in query results file: {e}")
        
        # Validate the structure
        if not isinstance(results, dict):
            raise ValueError("Query results must be a JSON object")
        
        if "queries" not in results:
            raise ValueError("Query results must contain a 'queries' field")
        
        if not isinstance(results["queries"], list):
            raise ValueError("Query results 'queries' field must be an array")
        
        return results
    
    async def start_planning_session(self, initial_intent: str) -> tuple[str, str]:
        """
        Start a new planning session with the data planning agent.
        
        Args:
            initial_intent: Initial business intent/requirement
            
        Returns:
            Tuple of (session_id, initial_questions)
        """
        request = {
            "name": "start_planning_session",
            "arguments": {
                "initial_intent": initial_intent
            }
        }
        
        timeout_config = httpx.Timeout(timeout=self.timeout, read=self.timeout)
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            response = await client.post(
                f"{self.planning_url}/mcp/call-tool",
                json=request
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Handle MCP content array format: {"result": [{"type": "text", "text": "..."}]}
            if "result" in result and isinstance(result["result"], list):
                content_list = result["result"]
                if len(content_list) > 0 and "text" in content_list[0]:
                    result_text = content_list[0]["text"]
                    
                    # Extract session ID from markdown text
                    session_id_match = re.search(r'\*\*Session ID:\*\*\s*`([^`]+)`', result_text)
                    if session_id_match:
                        session_id = session_id_match.group(1)
                        # Return the full text as questions (it's already formatted for display)
                        return session_id, result_text
            
            # Handle JSON-RPC 2.0 format
            elif "jsonrpc" in result and "result" in result:
                rpc_result = result["result"]
                
                # Check if it's MCP content format
                if isinstance(rpc_result, dict) and "content" in rpc_result:
                    content_list = rpc_result["content"]
                    if len(content_list) > 0 and "text" in content_list[0]:
                        result_text = content_list[0]["text"]
                        parsed = json.loads(result_text)
                        return parsed["session_id"], parsed["questions"]
                
                # Check if it's direct format inside JSON-RPC
                elif isinstance(rpc_result, dict) and "session_id" in rpc_result:
                    return rpc_result["session_id"], rpc_result["questions"]
            
            # Handle MCP protocol wrapped response with "content" key
            elif "result" in result and isinstance(result["result"], dict) and "content" in result["result"]:
                content_list = result["result"]["content"]
                if len(content_list) > 0 and "text" in content_list[0]:
                    result_text = content_list[0]["text"]
                    parsed = json.loads(result_text)
                    return parsed["session_id"], parsed["questions"]
            
            # Direct format
            elif "session_id" in result and "questions" in result:
                return result["session_id"], result["questions"]
            
            print(f"DEBUG: Full response: {json.dumps(result, indent=2)[:1000]}")
            raise ValueError("Unexpected response format from planning agent")
    
    async def modify_existing_prp(self, existing_prp: str, requested_changes: str) -> tuple[str, str]:
        """
        Start a PRP modification session with the data planning agent.
        
        Args:
            existing_prp: Existing PRP markdown content
            requested_changes: Description of changes to make to the PRP
            
        Returns:
            Tuple of (session_id, initial_questions)
        """
        request = {
            "name": "modify_existing_prp",
            "arguments": {
                "existing_prp": existing_prp,
                "requested_changes": requested_changes
            }
        }
        
        timeout_config = httpx.Timeout(timeout=self.timeout, read=self.timeout)
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            response = await client.post(
                f"{self.planning_url}/mcp/call-tool",
                json=request
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Handle MCP content array format: {"result": [{"type": "text", "text": "..."}]}
            if "result" in result and isinstance(result["result"], list):
                content_list = result["result"]
                if len(content_list) > 0 and "text" in content_list[0]:
                    result_text = content_list[0]["text"]
                    
                    # Extract session ID from markdown text
                    session_id_match = re.search(r'\*\*Session ID:\*\*\s*`([^`]+)`', result_text)
                    if session_id_match:
                        session_id = session_id_match.group(1)
                        # Return the full text as questions (it's already formatted for display)
                        return session_id, result_text
            
            # Handle JSON-RPC 2.0 format
            elif "jsonrpc" in result and "result" in result:
                rpc_result = result["result"]
                
                # Check if it's MCP content format
                if isinstance(rpc_result, dict) and "content" in rpc_result:
                    content_list = rpc_result["content"]
                    if len(content_list) > 0 and "text" in content_list[0]:
                        result_text = content_list[0]["text"]
                        parsed = json.loads(result_text)
                        return parsed["session_id"], parsed["questions"]
                
                # Check if it's direct format inside JSON-RPC
                elif isinstance(rpc_result, dict) and "session_id" in rpc_result:
                    return rpc_result["session_id"], rpc_result["questions"]
            
            # Handle MCP protocol wrapped response with "content" key
            elif "result" in result and isinstance(result["result"], dict) and "content" in result["result"]:
                content_list = result["result"]["content"]
                if len(content_list) > 0 and "text" in content_list[0]:
                    result_text = content_list[0]["text"]
                    parsed = json.loads(result_text)
                    return parsed["session_id"], parsed["questions"]
            
            # Direct format
            elif "session_id" in result and "questions" in result:
                return result["session_id"], result["questions"]
            
            print(f"DEBUG: Full response: {json.dumps(result, indent=2)[:1000]}")
            raise ValueError("Unexpected response format from planning agent")
    
    async def continue_planning_conversation(
        self, 
        session_id: str, 
        user_response: str
    ) -> tuple[str, bool]:
        """
        Continue the planning conversation with user responses.
        
        Args:
            session_id: Planning session ID
            user_response: User's response to questions
            
        Returns:
            Tuple of (next_questions, is_complete)
        """
        request = {
            "name": "continue_conversation",
            "arguments": {
                "session_id": session_id,
                "user_response": user_response
            }
        }
        
        timeout_config = httpx.Timeout(timeout=self.timeout, read=self.timeout)
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            try:
                response = await client.post(
                    f"{self.planning_url}/mcp/call-tool",
                    json=request
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                print(f"✗ HTTP Error {e.response.status_code} from planning agent")
                print(f"Response: {e.response.text[:2000]}")
                raise
            
            result = response.json()
            
            # Handle MCP content array format: {"result": [{"type": "text", "text": "..."}]}
            if "result" in result and isinstance(result["result"], list):
                content_list = result["result"]
                if len(content_list) > 0 and "text" in content_list[0]:
                    result_text = content_list[0]["text"]
                    
                    # Check if conversation is complete (text contains "Requirements Complete" or similar)
                    is_complete = (
                        "✅ Requirements Complete" in result_text or 
                        "✅ Requirements Gathering Complete" in result_text or
                        "Requirements Gathering Complete" in result_text or
                        "requirements gathering is complete" in result_text.lower()
                    )
                    
                    # Return the full text as questions (it's already formatted for display)
                    return result_text, is_complete
            
            # Handle JSON-RPC 2.0 format
            elif "jsonrpc" in result and "result" in result:
                rpc_result = result["result"]
                
                # Check if it's MCP content format
                if isinstance(rpc_result, dict) and "content" in rpc_result:
                    content_list = rpc_result["content"]
                    if len(content_list) > 0 and "text" in content_list[0]:
                        result_text = content_list[0]["text"]
                        parsed = json.loads(result_text)
                        return parsed["questions"], parsed.get("is_complete", False)
                
                # Check if it's direct format inside JSON-RPC
                elif isinstance(rpc_result, dict) and "questions" in rpc_result:
                    return rpc_result["questions"], rpc_result.get("is_complete", False)
            
            # Handle MCP protocol wrapped response with "content" key
            elif "result" in result and isinstance(result["result"], dict) and "content" in result["result"]:
                content_list = result["result"]["content"]
                if len(content_list) > 0 and "text" in content_list[0]:
                    result_text = content_list[0]["text"]
                    parsed = json.loads(result_text)
                    return parsed["questions"], parsed.get("is_complete", False)
            
            # Direct format
            elif "questions" in result:
                return result["questions"], result.get("is_complete", False)
            
            raise ValueError("Unexpected response format from planning agent")
    
    async def generate_prp(self, session_id: str) -> str:
        """
        Generate the final PRP document from a planning session.
        
        Args:
            session_id: Planning session ID
            
        Returns:
            Generated PRP markdown text
        """
        request = {
            "name": "generate_data_prp",
            "arguments": {
                "session_id": session_id
            }
        }
        
        timeout_config = httpx.Timeout(timeout=self.timeout, read=self.timeout)
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            response = await client.post(
                f"{self.planning_url}/mcp/call-tool",
                json=request
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Handle MCP content array format: {"result": [{"type": "text", "text": "..."}]}
            if "result" in result and isinstance(result["result"], list):
                content_list = result["result"]
                if len(content_list) > 0 and "text" in content_list[0]:
                    # The text itself is the PRP markdown
                    return content_list[0]["text"]
            
            # Handle JSON-RPC 2.0 format
            elif "jsonrpc" in result and "result" in result:
                rpc_result = result["result"]
                
                # Check if it's MCP content format
                if isinstance(rpc_result, dict) and "content" in rpc_result:
                    content_list = rpc_result["content"]
                    if len(content_list) > 0 and "text" in content_list[0]:
                        result_text = content_list[0]["text"]
                        parsed = json.loads(result_text)
                        return parsed["prp_text"]
                
                # Check if it's direct format inside JSON-RPC
                elif isinstance(rpc_result, dict) and "prp_text" in rpc_result:
                    return rpc_result["prp_text"]
            
            # Handle MCP protocol wrapped response with "content" key
            elif "result" in result and isinstance(result["result"], dict) and "content" in result["result"]:
                content_list = result["result"]["content"]
                if len(content_list) > 0 and "text" in content_list[0]:
                    result_text = content_list[0]["text"]
                    parsed = json.loads(result_text)
                    return parsed["prp_text"]
            
            # Direct format
            elif "prp_text" in result:
                return result["prp_text"]
            
            raise ValueError("Unexpected response format from planning agent")

    def _extract_target_schema_from_prp(self, prp_text: str) -> List[Dict[str, Any]]:
        """
        Extract target view schemas from PRP Section 9.

        Args:
            prp_text: The full PRP markdown text.

        Returns:
            A list of target table specifications.
        """
        target_tables = []
        
        # Isolate Section 9 - handle both "9." and "## 9." formats
        section_9_match = re.search(r"##?\s*9\.\s+Data\s+Requirements(.*?)(?=##?\s*\d+\.|$)", prp_text, re.DOTALL | re.IGNORECASE)
        if not section_9_match:
            print("DEBUG: Could not find Section 9 in PRP")
            return []

        section_9_content = section_9_match.group(1)
        print(f"DEBUG: Found Section 9, length: {len(section_9_content)} characters")

        # Find all target view blocks - look for View Name pattern
        view_pattern = r'\*\*View\s+Name\*\*:\s*`([^`]+)`(.*?)(?=\*\*View\s+Name\*\*:|###|$)'
        view_blocks = re.findall(view_pattern, section_9_content, re.DOTALL)
        
        print(f"DEBUG: Found {len(view_blocks)} view blocks")
        
        for view_name, view_content in view_blocks:
            print(f"DEBUG: Processing view: {view_name}")
            
            # Extract purpose/description
            description_match = re.search(r'-\s+\*\*Purpose\*\*:\s*([^\n]+)', view_content)
            description = description_match.group(1).strip() if description_match else ""

            # Extract schema section
            schema_match = re.search(r'-\s+\*\*Schema\*\*:(.*?)(?=-\s+\*\*|$)', view_content, re.DOTALL)
            if not schema_match:
                print(f"DEBUG: No schema found for {view_name}")
                continue
                
            schema_content = schema_match.group(1)
            
            columns = []
            # Extract columns - format is "  - `column_name` (type): description"
            column_matches = re.findall(r'\s+-\s+`([^`]+)`\s+\(([^)]+)\):\s*([^\n]+)', schema_content)
            print(f"DEBUG: Found {len(column_matches)} columns for {view_name}")
            
            for col_name, col_type, col_desc in column_matches:
                columns.append({
                    "name": col_name.strip(),
                    "type": col_type.strip(),
                    "description": col_desc.strip()
                })
            
            if columns:
                target_tables.append({
                    "target_table_name": view_name,
                    "target_description": description,
                    "target_columns": columns
                })
                print(f"DEBUG: Added target table: {view_name} with {len(columns)} columns")
                
        return target_tables

    async def discover_source_tables_from_prp(
        self,
        prp_text: str,
        max_results_per_query: int = 5
    ) -> Dict[str, Any]:
        """
        Discover source tables using the new strategy-driven discovery agent.

        Args:
            prp_text: Generated PRP markdown text
            max_results_per_query: Maximum source tables per search query

        Returns:
            Discovery results structured for query generation.
        """
        print(f"📄 Extracting target schema from PRP Section 9...")
        
        target_schemas = self._extract_target_schema_from_prp(prp_text)
        if not target_schemas:
            print("✗ Error: Could not find any target view schemas in PRP Section 9.")
            return {"target_tables": []}
        
        # For this workflow, we'll focus on the first target schema found.
        target_schema = target_schemas[0]
        print(f"✓ Extracted schema for target view: `{target_schema['target_table_name']}`")
        print()
        
        print(f"🔍 Discovering source tables with new strategy-driven agent...")
        
        request = {
            "name": "discover_from_prp",
            "arguments": {
                "prp_markdown": prp_text,
                "target_schema": target_schema,
                "max_results_per_query": max_results_per_query,
            }
        }
        
        # Debug: Print what we're sending
        print(f"DEBUG: Sending request with tool: {request['name']}")
        print(f"DEBUG: Arguments keys: {list(request['arguments'].keys())}")
        print(f"DEBUG: target_schema present: {'target_schema' in request['arguments']}")
        print(f"DEBUG: target_schema type: {type(request['arguments'].get('target_schema'))}")
        if 'target_schema' in request['arguments']:
            print(f"DEBUG: target_schema keys: {list(request['arguments']['target_schema'].keys())}")
        print()
        
        try:
            timeout_config = httpx.Timeout(timeout=self.timeout, read=self.timeout)
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                response = await client.post(
                    f"{self.discovery_url}/mcp/call-tool",
                    json=request
                )
                response.raise_for_status()
                
                result = response.json()
                
                if "result" in result and len(result["result"]) > 0:
                    result_text = result["result"][0]["text"]
                    
                    if not result_text or not result_text.strip():
                        print("✗ Error: Empty response from discovery tool")
                        return {"target_tables": []}
                    
                    try:
                        # Check for actual MCP error format (starts with "# Error:")
                        if result_text.strip().startswith("# Error:"):
                             print("✗ Error received from discovery tool:")
                             print("--- SERVER RESPONSE ---")
                             print(result_text)
                             print("-----------------------")
                             return {"target_tables": []}
                        
                        data = json.loads(result_text)
                    except json.JSONDecodeError:
                        print(f"✗ Error: Invalid JSON response from discovery tool.")
                        print("--- SERVER RESPONSE ---")
                        print(result_text)
                        print("-----------------------")
                        return {"target_tables": []}

                    # Save the raw discovery results for debugging
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    discovery_file = self.output_dir / f"prp_discovery_{timestamp}.json"
                    discovery_file.write_text(json.dumps(data, indent=2))
                    print(f"💾 Saved raw discovery results to: {discovery_file}")
                    print()

                    # Adapt the new response format to the one expected by the rest of this script
                    all_discovered_tables = []
                    if "search_plan_results" in data:
                        for step in data["search_plan_results"]:
                            all_discovered_tables.extend(step.get("discovered_tables", []))
                    
                    # Deduplicate tables based on table_id
                    unique_tables = {tbl['table_id']: tbl for tbl in all_discovered_tables}.values()

                    # Reconstruct the "target_tables" structure for the query generator
                    adapted_result = {
                        "target_tables": [
                            {
                                **target_schema,
                                "discovered_tables": list(unique_tables)
                            }
                        ],
                        "total_targets": 1,
                        "total_discovered": len(unique_tables)
                    }

                    print(f"✓ Discovery complete across {data.get('total_steps', 0)} search step(s)")
                    print(f"✓ Found {len(unique_tables)} unique source table(s) for `{target_schema['target_table_name']}`")
                    print()
                    
                    return adapted_result
                else:
                    print("✗ No 'result' key in discovery response body.")
                    print(f"Full response: {result}")
                    return {"target_tables": []}

        except httpx.HTTPStatusError as e:
            print(f"✗ HTTP Error: Received status code {e.response.status_code} from discovery agent.")
            print("--- FULL SERVER TRACEBACK ---")
            print(e.response.text)
            print("-----------------------------")
            return {"target_tables": []}
        except Exception:
            print(f"✗ An unexpected error occurred in the client.")
            print("--- FULL CLIENT TRACEBACK ---")
            traceback.print_exc()
            print("-----------------------------")
            return {"target_tables": []}

    async def handle_user_confirmation(self, gap_details: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Prompt the user to resolve a data gap.
        
        Args:
            gap_details: The details of the data gap from the agent.
            
        Returns:
            A dictionary with the resolved gap, or None if the user skips.
        """
        print("=" * 80)
        print("ACTION REQUIRED: Please Resolve a Data Gap")
        print("=" * 80)
        print(f"\nThe agent could not automatically find a data source for the target view:")
        print(f"  Target View: {gap_details['target_view']}")
        print(f"  Data Gap: {gap_details['gap_description']}")
        print("\nPlease select one of the following candidate tables to resolve this gap:")
        
        candidates = gap_details["candidate_tables"]
        for i, table in enumerate(candidates):
            print(f"  {i + 1}. {table}")
        
        print(f"  {len(candidates) + 1}. Skip this table and continue")
        print("-" * 80)
        
        while True:
            try:
                selection = input("Enter your choice (number): ")
                choice = int(selection)
                if 1 <= choice <= len(candidates):
                    selected_table = candidates[choice - 1]
                    print(f"You selected: {selected_table}")
                    return {gap_details["gap_id"]: selected_table}
                elif choice == len(candidates) + 1:
                    print("You chose to skip. The agent will proceed without this target view.")
                    return None
                else:
                    print("Invalid choice. Please enter a number from the list.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    async def generate_queries_for_target(
        self,
        target_table_name: str,
        target_description: str,
        target_columns: List[Dict[str, Any]],
        source_datasets: List[Dict[str, Any]],
        max_queries: int = 3,
        max_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Generate SELECT queries for a target table specification using async endpoint with polling.
        
        Args:
            target_table_name: Name of target table
            target_description: Description of target table
            target_columns: Target column specifications
            source_datasets: Discovered source tables
            max_queries: Maximum number of queries to generate
            max_iterations: Maximum refinement iterations
            
        Returns:
            Query generation results
        """
        print(f"🤖 Generating queries for: {target_table_name}")
        print(f"   Using {len(source_datasets)} source table(s)")
        print()
        
        # Build insight from target specification
        column_list = ", ".join([col['name'] for col in target_columns[:10]])
        if len(target_columns) > 10:
            column_list += f", ... ({len(target_columns)} total columns)"
        
        insight = f"Generate a query that returns data for {target_table_name}: {target_description}. Required columns: {column_list}"
        
        request = {
            "name": "generate_queries",
            "arguments": {
                "insight": insight,
                "datasets": source_datasets,
                "max_queries": max_queries,
                "max_iterations": max_iterations,
                "target_table_name": target_table_name
            }
        }
        
        # Step 1: Start async task
        timeout_config = httpx.Timeout(timeout=30.0)  # 30s timeout for starting task
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            try:
                response = await client.post(
                    f"{self.query_gen_url}/mcp/call-tool-async",
                    json=request
                )
                response.raise_for_status()
                
                task_data = response.json()
                task_id = task_data["task_id"]
                
                print(f"   Task started: {task_id}")
                print()
            except httpx.HTTPStatusError as e:
                print(f"   HTTP Error: {e.response.status_code}")
                print(f"   Response body: {e.response.text[:500]}")
                print(f"✗ Failed to start async query generation task")
                sys.exit(1)
            except Exception as e:
                print(f"   Request failed: {type(e).__name__}: {e}")
                print(f"✗ Failed to start async query generation task")
                sys.exit(1)
        
        # Step 2: Poll for completion
        start_time = time.time()
        poll_timeout = httpx.Timeout(timeout=30.0)  # 30s timeout per poll
        
        async with httpx.AsyncClient(timeout=poll_timeout) as client:
            while True:
                elapsed = time.time() - start_time
                
                # Check if we've exceeded max wait time
                if elapsed > self.max_query_wait_seconds:
                    print(f"✗ Query generation timed out after {self.max_query_wait_seconds}s ({self.max_query_wait_seconds/60:.1f} minutes)")
                    print(f"   Task ID: {task_id}")
                    print(f"   Target: {target_table_name}")
                    sys.exit(1)
                
                # Check task status (silent polling)
                try:
                    status_response = await client.get(
                        f"{self.query_gen_url}/mcp/tasks/{task_id}"
                    )
                    status_response.raise_for_status()
                    status_data = status_response.json()
                    
                except Exception as e:
                    print(f"✗ Error checking task status: {e}")
                    print(f"   Task ID: {task_id}")
                    sys.exit(1)
                
                # Check if completed
                if status_data["status"] == "completed":
                    # Fetch result
                    try:
                        result_response = await client.get(
                            f"{self.query_gen_url}/mcp/tasks/{task_id}/result"
                        )
                        result_response.raise_for_status()
                        result_data = result_response.json()
                        
                        # Extract the actual result from the response
                        return result_data.get("result", result_data)
                        
                    except Exception as e:
                        print(f"✗ Error fetching task result: {e}")
                        print(f"   Task ID: {task_id}")
                        sys.exit(1)
                
                # Check if failed
                elif status_data["status"] == "failed":
                    error = status_data.get("error", "Unknown error")
                    print(f"✗ Query generation failed on server")
                    print(f"   Task ID: {task_id}")
                    print(f"   Error: {error}")
                    print(f"   Target: {target_table_name}")
                    sys.exit(1)
                
                # Still running or pending - wait and poll again
                await asyncio.sleep(5)  # Poll every 5 seconds
    
    def _sanitize_query_name(self, target_name: str) -> str:
        """
        Convert target table name to camelCase for GraphQL query name.
        
        Args:
            target_name: Target table name (e.g., "backtest_evaluation_view")
            
        Returns:
            CamelCase query name (e.g., "backtestEvaluationView")
        """
        # Remove special characters and split by underscore
        parts = target_name.replace("-", "_").split("_")
        
        # Convert to camelCase
        if not parts:
            return "query"
        
        # First word lowercase, rest capitalized
        camel_case = parts[0].lower()
        for part in parts[1:]:
            if part:
                camel_case += part.capitalize()
        
        return camel_case
    
    async def generate_graphql_api(
        self,
        target_name: str,
        validated_queries: List[Dict[str, Any]],
        project_name: str,
        output_path: Optional[str] = None,
        validation_level: str = "standard",
        api_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate Apollo GraphQL Server from validated SQL queries.
        
        Args:
            target_name: Name of target table (used for query naming)
            validated_queries: List of validated queries from query generation
            project_name: Name of the project for lineage tracking
            output_path: Optional output path (uses env var if not provided)
            validation_level: Validation level (quick, standard, full)
            
        Returns:
            GraphQL generation results
        """
        # Filter only valid queries
        valid_queries = [
            q for q in validated_queries
            if q.get("validation_status") == "valid"
        ]
        
        if not valid_queries:
            return {
                "success": False,
                "message": "No valid queries to convert to GraphQL",
                "error": "All queries failed validation"
            }
        
        print(f"🔮 Generating GraphQL API for: {target_name}")
        print(f"   Valid queries: {len(valid_queries)}")
        print()
        
        # Transform query format for GraphQL agent
        # Queries now have query_name from query-generation-agent, pass through directly
        # Include metadata for rich GraphQL schema documentation
        graphql_queries = []
        for query in valid_queries:
            graphql_query = {
                "query_name": query["query_name"],
                "sql": query["sql"],
                "source_tables": query["source_tables"]
            }
            
            # Add optional metadata fields if available
            if "description" in query:
                graphql_query["description"] = query["description"]
            if "alignment_score" in query:
                graphql_query["alignment_score"] = query["alignment_score"]
            if "iterations" in query:
                graphql_query["iterations"] = query["iterations"]
            if "generation_time_ms" in query:
                graphql_query["generation_time_ms"] = query["generation_time_ms"]
            
            # IMPORTANT: Pass validation_details with result_schema for field descriptions
            if "validation_details" in query:
                graphql_query["validation_details"] = query["validation_details"]
            
            graphql_queries.append(graphql_query)
        
        request = {
            "name": "generate_graphql_api",
            "arguments": {
                "queries": graphql_queries,
                "project_name": project_name,
                "validation_level": validation_level
            }
        }
        
        # Add API-level metadata if provided (for schema documentation)
        if api_metadata:
            if "insight" in api_metadata:
                request["arguments"]["insight"] = api_metadata["insight"]
            if "summary" in api_metadata:
                request["arguments"]["summary"] = api_metadata["summary"]
            if "dataset_count" in api_metadata:
                request["arguments"]["dataset_count"] = api_metadata["dataset_count"]
            if "execution_time_ms" in api_metadata:
                request["arguments"]["execution_time_ms"] = api_metadata["execution_time_ms"]
        
        # Add output_path if provided
        if output_path:
            request["arguments"]["output_path"] = output_path
        
        try:
            timeout_config = httpx.Timeout(timeout=self.timeout, read=self.timeout)
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                response = await client.post(
                    f"{self.graphql_url}/mcp/call-tool",
                    json=request
                )
                response.raise_for_status()
                
                result = response.json()
                
                # The actual result is inside the 'result' key for JSON-RPC responses
                if "jsonrpc" in result and "result" in result:
                    app_result = result["result"]
                else:
                    app_result = result
                
                # Handle MCP protocol wrapped response (text content)
                if isinstance(app_result, dict) and "content" in app_result:
                    content_list = app_result["content"]
                    if len(content_list) > 0 and "text" in content_list[0]:
                        result_text = content_list[0]["text"]
                        try:
                            return json.loads(result_text)
                        except json.JSONDecodeError as e:
                            print(f"✗ Error parsing GraphQL response: {e}")
                            print(f"Response text: {result_text[:500]}")
                            return {
                                "success": False,
                                "message": "Failed to parse GraphQL response",
                                "error": str(e)
                            }
                
                # Direct JSON format
                elif isinstance(app_result, dict) and "success" in app_result:
                    return app_result
                
                # Unknown format
                else:
                    print(f"✗ Unexpected response structure from GraphQL agent")
                    print(f"Original response keys: {list(result.keys())}")
                    print(f"Application-level result: {app_result}")
                    return {
                        "success": False,
                        "message": "Unexpected response format",
                        "error": "Unknown response structure"
                    }
        
        except httpx.HTTPStatusError as e:
            print(f"✗ HTTP Error: {e.response.status_code}")
            print(f"Response: {e.response.text[:500]}")
            return {
                "success": False,
                "message": f"HTTP error {e.response.status_code}",
                "error": e.response.text
            }
        except Exception as e:
            print(f"✗ Error generating GraphQL API: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "message": "Failed to generate GraphQL API",
                "error": str(e)
            }
    
    async def check_health(self) -> Dict[str, bool]:
        """
        Check health of all services.
        
        Returns:
            Dict with health status of each service
        """
        status = {
            "planning_agent": False,
            "discovery_agent": False,
            "query_generation_agent": False,
            "graphql_agent": False
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(f"{self.planning_url}/health")
                status["planning_agent"] = response.status_code == 200
            except Exception as e:
                print(f"✗ Data Planning Agent not available: {e}")
            
            try:
                response = await client.get(f"{self.discovery_url}/health")
                status["discovery_agent"] = response.status_code == 200
            except Exception as e:
                print(f"✗ Data Discovery Agent not available: {e}")
            
            try:
                response = await client.get(f"{self.query_gen_url}/health")
                status["query_generation_agent"] = response.status_code == 200
            except Exception as e:
                print(f"✗ Query Generation Agent not available: {e}")
            
            try:
                response = await client.get(f"{self.graphql_url}/health")
                status["graphql_agent"] = response.status_code == 200
            except Exception as e:
                print(f"✗ Data GraphQL Agent not available: {e}")
        
        return status


async def run_interactive_planning(
    client: IntegratedMCPClient,
    initial_intent: str,
    max_turns: int = 10
) -> str:
    """
    Run interactive planning session with Q&A.
    
    Args:
        client: Integrated MCP client
        initial_intent: Initial business intent
        max_turns: Maximum Q&A turns (client-side limit, separate from agent's limit)
    
    Returns:
        Generated PRP text
    """
    print(f"🎯 Starting planning session...")
    print(f"   Initial intent: '{initial_intent}'")
    print(f"   Max turns (client-side): {max_turns}")
    print()
    print("ℹ️  Note: The planning agent has its own MAX_CONVERSATION_TURNS setting")
    print("   that may trigger completion before the client-side limit.")
    print()
    
    # Start session
    session_id, questions = await client.start_planning_session(initial_intent)
    
    print(f"   Session ID: {session_id}")
    print()
    
    # Interactive Q&A loop
    turn = 0
    is_complete = False
    
    while not is_complete and turn < max_turns:
        print("=" * 80)
        print(f"PLANNING QUESTIONS (Turn {turn + 1}/{max_turns})")
        print("=" * 80)
        print()
        print(questions)
        print()
        
        # Check if already complete (no need for user input)
        if is_complete:
            print("=" * 80)
            print("✓ Requirements gathering complete!")
            print()
            break
        
        print("-" * 80)
        print("Your response:")
        print("Tip: For multiple choice, include the letter (a, b, c, d)")
        print("-" * 80)
        
        # Get user input
        user_response = input("> ").strip()
        
        if not user_response:
            print("⚠️  Empty response, please try again")
            continue
        
        print()
        print("⏳ Processing your response...")
        print()
        
        # Continue conversation
        next_questions, is_complete = await client.continue_planning_conversation(
            session_id, user_response
        )
        
        questions = next_questions
        turn += 1
        
        # If complete after this turn, show the completion message and break
        if is_complete:
            print("=" * 80)
            print(f"PLANNING QUESTIONS (Turn {turn}/{max_turns})")
            print("=" * 80)
            print()
            print(questions)
            print()
            print("=" * 80)
            print("✓ Requirements gathering complete!")
            print()
            break
    
    # Generate PRP
    if not is_complete:
        print("⚠️  Max client-side turns reached, proceeding with available information")
        print()
    
    print("📝 Generating Data PRP...")
    print()
    
    prp_text = await client.generate_prp(session_id)
    
    # Save PRP to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prp_file = client.output_dir / f"prp_{timestamp}.md"
    prp_file.write_text(prp_text)
    
    print("✓ Data PRP generated")
    print(f"💾 Saved PRP to: {prp_file}")
    print()
    
    return prp_text


async def run_update_planning(
    client: IntegratedMCPClient,
    prp_file: str,
    requested_changes: str,
    max_turns: int = 10
) -> str:
    """
    Run PRP update session with Q&A for modifications.
    
    Args:
        client: Integrated MCP client
        prp_file: Path to existing PRP file
        requested_changes: Description of changes to make
        max_turns: Maximum Q&A turns (client-side limit, separate from agent's limit)
    
    Returns:
        Generated updated PRP text
    """
    print(f"🔄 Starting PRP update session...")
    print(f"   Existing PRP: {prp_file}")
    print(f"   Requested changes: '{requested_changes}'")
    print(f"   Max turns (client-side): {max_turns}")
    print()
    print("ℹ️  Note: The planning agent has its own MAX_CONVERSATION_TURNS setting")
    print("   that may trigger completion before the client-side limit.")
    print()
    
    # Load existing PRP
    try:
        existing_prp = client.load_prp_from_file(prp_file)
        print("✓ Loaded existing PRP")
        print()
    except (FileNotFoundError, ValueError) as e:
        print(f"✗ Error loading PRP file: {e}")
        raise
    
    # Start modification session
    session_id, questions = await client.modify_existing_prp(existing_prp, requested_changes)
    
    print(f"   Session ID: {session_id}")
    print()
    
    # Interactive Q&A loop (same as planning)
    turn = 0
    is_complete = False
    
    while not is_complete and turn < max_turns:
        print("=" * 80)
        print(f"PRP MODIFICATION QUESTIONS (Turn {turn + 1}/{max_turns})")
        print("=" * 80)
        print()
        print(questions)
        print()
        
        # Check if already complete (no need for user input)
        if is_complete:
            print("=" * 80)
            print("✓ Modification gathering complete!")
            print()
            break
        
        print("-" * 80)
        print("Your response:")
        print("Tip: For multiple choice, include the letter (a, b, c, d)")
        print("-" * 80)
        
        # Get user input
        user_response = input("> ").strip()
        
        if not user_response:
            print("⚠️  Empty response, please try again")
            continue
        
        print()
        print("⏳ Processing your response...")
        print()
        
        # Continue conversation
        next_questions, is_complete = await client.continue_planning_conversation(
            session_id, user_response
        )
        
        questions = next_questions
        turn += 1
        
        # If complete after this turn, show the completion message and break
        if is_complete:
            print("=" * 80)
            print(f"PRP MODIFICATION QUESTIONS (Turn {turn}/{max_turns})")
            print("=" * 80)
            print()
            print(questions)
            print()
            print("=" * 80)
            print("✓ Modification gathering complete!")
            print()
            break
    
    # Generate updated PRP
    if not is_complete:
        print("⚠️  Max client-side turns reached, proceeding with available information")
        print()
    
    print("📝 Generating updated Data PRP...")
    print()
    
    prp_text = await client.generate_prp(session_id)
    
    # Save updated PRP to file with new timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    updated_prp_file = client.output_dir / f"prp_updated_{timestamp}.md"
    updated_prp_file.write_text(prp_text)
    
    print("✓ Updated Data PRP generated")
    print(f"💾 Saved updated PRP to: {updated_prp_file}")
    print()
    
    return prp_text


def print_query_results(results: Dict[str, Any], output_dir: Path, target_name: str = "") -> None:
    """
    Pretty print query generation results.
    
    Args:
        results: Query generation results
        output_dir: Directory to save output files
        target_name: Optional target table name for context
    """
    # Save query results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_target = target_name.replace(" ", "_") if target_name else "queries"
    query_file = output_dir / f"{safe_target}_{timestamp}.json"
    query_file.write_text(json.dumps(results, indent=2))
    print(f"💾 Saved query results to: {query_file}")
    print()
    
    print("=" * 100)
    if target_name:
        print(f"QUERY GENERATION: {target_name}")
    else:
        print("QUERY GENERATION RESULTS")
    print("=" * 100)
    print()
    
    print(f"📊 Summary:")
    print(f"   Total queries attempted: {results.get('total_attempted', 0)}")
    print(f"   Total queries validated: {results.get('total_validated', 0)}")
    print(f"   Execution time: {results.get('execution_time_ms', 0):.0f}ms ({results.get('execution_time_ms', 0)/1000:.1f}s)")
    print()
    
    if results.get("warnings"):
        print("⚠️  Warnings:")
        for warning in results["warnings"]:
            print(f"   - {warning}")
        print()
    
    queries = results.get("queries", [])
    
    if not queries:
        print("No queries generated.")
        return
    
    for i, query in enumerate(queries, 1):
        print("-" * 100)
        print(f"Query #{i}")
        print("-" * 100)
        
        status = query.get("validation_status", "unknown")
        status_emoji = "✓" if status == "valid" else "✗"
        
        print(f"{status_emoji} Status: {status.upper()}")
        print(f"   Alignment Score: {query.get('alignment_score', 0):.2f}")
        print(f"   Iterations: {query.get('iterations', 0)}")
        print(f"   Generation Time: {query.get('generation_time_ms', 0):.0f}ms")
        print()
        
        if status == "valid":
            print(f"📝 Description:")
            print(f"   {query.get('description', 'N/A')}")
            print()
            
            print(f"💾 SQL Query:")
            print()
            sql_lines = query.get("sql", "").split("\n")
            for line in sql_lines:
                print(f"   {line}")
            print()
            
            # Cost estimation
            if query.get("estimated_cost_usd"):
                print(f"💰 Estimated Cost: ${query['estimated_cost_usd']:.6f} USD")
            
            if query.get("estimated_bytes_processed"):
                bytes_processed = query["estimated_bytes_processed"]
                mb = bytes_processed / (1024 * 1024)
                gb = bytes_processed / (1024 * 1024 * 1024)
                if gb >= 1:
                    print(f"📦 Estimated Data: {gb:.2f} GB")
                else:
                    print(f"📦 Estimated Data: {mb:.2f} MB")
            print()
            
            # Validation details
            validation = query.get("validation_details", {})
            if validation.get("dry_run_success"):
                print("✓ Dry run validation: PASSED")
            
            if validation.get("syntax_valid"):
                print("✓ Syntax validation: PASSED")
            
            if validation.get("alignment_score"):
                print(f"✓ Alignment check: {validation['alignment_score']:.2f}")
            print()
            
        else:
            # Error case
            validation = query.get("validation_details", {})
            error = validation.get("error_message", "Unknown error")
            
            print(f"❌ Error: {error}")
            print()
            
            if query.get("sql"):
                print("Last attempted SQL:")
                print()
                for line in query["sql"].split("\n"):
                    print(f"   {line}")
                print()
    
    print("=" * 100)


async def main(
    initial_intent: Optional[str] = None,
    prp_file: Optional[str] = None,
    update_prp: Optional[str] = None,
    query_results_file: Optional[str] = None,
    max_results_per_query: int = 5,
    max_planning_turns: int = 10,
    max_queries: int = 3,
    max_iterations: int = 10,
    max_query_wait_seconds: float = 900.0,
    planning_url: str = "http://localhost:8082",
    discovery_url: str = "http://localhost:8080",
    query_gen_url: str = "http://localhost:8081",
    graphql_url: str = "http://localhost:8083",
    skip_graphql: bool = False,
    project_name: Optional[str] = None
) -> None:
    """
    Run the integrated workflow example.
    
    Generates SELECT queries for all target tables defined in PRP Section 9,
    and optionally creates a GraphQL API from the validated queries.
    
    Args:
        initial_intent: Initial business intent for planning or changes for update-prp
        prp_file: Path to existing PRP file (skips planning phase)
        update_prp: Path to existing PRP file to update (requires initial_intent for changes)
        query_results_file: Path to query results JSON (skips to GraphQL generation only)
        max_results_per_query: Maximum source tables to discover per target
        max_planning_turns: Maximum Q&A turns in planning phase
        max_queries: Maximum number of queries to generate per target
        max_iterations: Maximum refinement iterations per query
        max_query_wait_seconds: Maximum time to wait for query generation (default: 900s = 15 minutes)
        planning_url: Data planning agent URL
        discovery_url: Data discovery agent URL
        query_gen_url: Query generation agent URL
        graphql_url: Data GraphQL agent URL
        skip_graphql: Skip GraphQL API generation
        project_name: Project name for GraphQL lineage tracking
    """
    print("=" * 100)
    print("INTEGRATED WORKFLOW: PLANNING → DISCOVERY → QUERY GENERATION → GRAPHQL API")
    print("=" * 100)
    print()
    
    # Validate input parameters
    if update_prp:
        if not initial_intent:
            print("✗ Error: --update-prp requires -i/--initial-intent to describe the changes")
            return
        if prp_file or query_results_file:
            print("✗ Error: --update-prp cannot be used with --prp-file or --query-results-file")
            return
    
    # Check for mutual exclusivity
    if query_results_file:
        if prp_file or initial_intent:
            print("⚠️  --query-results-file provided. Skipping planning, discovery, and query generation.")
            print("   Going directly to GraphQL generation.")
            print()
    elif prp_file and initial_intent:
        print("⚠️  Both --prp-file and --initial-intent provided. Using --prp-file (skipping planning).")
        print()
    
    # Use default if not provided and no prp_file, update_prp, or query_results_file
    if initial_intent is None and prp_file is None and update_prp is None and query_results_file is None:
        initial_intent = "We want to analyze customer transaction patterns to identify high-value customers"
        print(f"ℹ️  Using default intent: '{initial_intent}'")
        print()
    
    # Initialize client with output directory
    output_dir = "output"
    client = IntegratedMCPClient(
        planning_url=planning_url,
        discovery_url=discovery_url,
        query_gen_url=query_gen_url,
        graphql_url=graphql_url,
        max_query_wait_seconds=max_query_wait_seconds,
        output_dir=output_dir
    )
    
    print(f"📁 Output directory: {output_dir}/")
    print()
    
    # Check health (all agents or skip as needed)
    print("🏥 Checking service health...")
    health = await client.check_health()
    
    # If using query_results_file, only check GraphQL agent
    if query_results_file:
        if not skip_graphql:
            if not health["graphql_agent"]:
                print("   ✗ Data GraphQL Agent: unavailable")
                print("   Please start: cd /home/user/git/data-graphql-agent && poetry run python -m data_graphql_agent.mcp")
                return
            print("   ✓ Data GraphQL Agent: healthy")
    else:
        # Check all relevant agents for full workflow
        # Only check planning agent if not using prp_file (but always check if using update_prp)
        if not prp_file or update_prp:
            if not health["planning_agent"]:
                print("   ✗ Data Planning Agent: unavailable")
                print("   Please start: cd /home/user/git/data-planning-agent && poetry run python -m data_planning_agent.mcp")
                return
            print("   ✓ Data Planning Agent: healthy")
        
        if not health["discovery_agent"]:
            print("   ✗ Data Discovery Agent: unavailable")
            print("   Please start: cd /home/user/git/data-discovery-agent && poetry run python -m data_discovery_agent.mcp")
            return
        print("   ✓ Data Discovery Agent: healthy")
        
        if not health["query_generation_agent"]:
            print("   ✗ Query Generation Agent: unavailable")
            print("   Please start: cd /home/user/git/query-generation-agent && poetry run python -m query_generation_agent.mcp")
            return
        print("   ✓ Query Generation Agent: healthy")
        
        # Only check GraphQL agent if not skipping GraphQL generation
        if not skip_graphql:
            if not health["graphql_agent"]:
                print("   ✗ Data GraphQL Agent: unavailable")
                print("   Please start: cd /home/user/git/data-graphql-agent && poetry run python -m data_graphql_agent.mcp")
                return
            print("   ✓ Data GraphQL Agent: healthy")
    
    print()
    
    # Fast path: Load query results and generate GraphQL only
    if query_results_file:
        print("=" * 100)
        print("FAST PATH: GRAPHQL GENERATION FROM EXISTING QUERIES")
        print("=" * 100)
        print()
        
        # Load query results from file
        print(f"📄 Loading query results from file: {query_results_file}")
        try:
            query_results = client.load_query_results_from_file(query_results_file)
            print("✓ Query results loaded successfully")
            print(f"   Total queries in file: {len(query_results.get('queries', []))}")
            
            valid_queries = [
                q for q in query_results.get("queries", [])
                if q.get("validation_status") == "valid"
            ]
            print(f"   Valid queries: {len(valid_queries)}")
            print()
        except (FileNotFoundError, ValueError) as e:
            print(f"✗ Error loading query results file: {e}")
            return
        
        if not valid_queries:
            print("✗ No valid queries found in file. Cannot generate GraphQL API.")
            return
        
        # Determine project name and target name
        # Use project_name from file if present, otherwise derive from filename or use provided
        if not project_name and "project_name" in query_results:
            project_name = query_results["project_name"]
        
        if not project_name:
            # Extract from filename (e.g., backtest_evaluation_view_20251024_190934.json)
            filename = Path(query_results_file).stem
            # Remove timestamp suffix if present
            target_name = re.sub(r'_\d{8}_\d{6}$', '', filename)
            project_name = target_name.replace("_", "-")
        else:
            # Derive target name from project name
            target_name = project_name.replace("-", "_")
        
        print(f"📦 Project name: {project_name}")
        print(f"🎯 Target name: {target_name}")
        print()
        
        # Generate GraphQL API
        if not skip_graphql:
            print("=" * 100)
            print("GENERATING GRAPHQL API")
            print("=" * 100)
            print()
            
            try:
                # Extract API-level metadata from query results
                api_metadata = {
                    "insight": query_results.get("insight"),
                    "summary": query_results.get("summary"),
                    "dataset_count": query_results.get("dataset_count"),
                    "execution_time_ms": query_results.get("execution_time_ms"),
                }
                
                graphql_result = await client.generate_graphql_api(
                    target_name=target_name,
                    validated_queries=query_results["queries"],
                    project_name=project_name,
                    validation_level="standard",
                    api_metadata=api_metadata
                )
                
                print(f"DEBUG: Raw GraphQL response: {json.dumps(graphql_result, indent=2)}")
                
                # Save GraphQL result
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                graphql_file = client.output_dir / f"graphql_results_{timestamp}.json"
                graphql_file.write_text(json.dumps([{
                    "target_table": target_name,
                    "result": graphql_result
                }], indent=2))
                print(f"💾 Saved GraphQL results to: {graphql_file}")
                print()
                
                # Print result
                if graphql_result.get("success"):
                    print("✓ GraphQL API generated successfully")
                    print(f"   Output: {graphql_result.get('output_path', 'N/A')}")
                    print(f"   Files generated: {len(graphql_result.get('files_generated', []))}")
                    if graphql_result.get("message"):
                        print(f"   {graphql_result['message']}")
                    print()
                    
                    # List generated files
                    files_generated = graphql_result.get("files_generated", [])
                    if files_generated:
                        print("📋 Generated files:")
                        for file_info in files_generated[:10]:  # Show first 10
                            print(f"   - {file_info.get('path', file_info.get('name', 'unknown'))}")
                        if len(files_generated) > 10:
                            print(f"   ... and {len(files_generated) - 10} more files")
                        print()
                    
                    print("=" * 100)
                    print("WORKFLOW COMPLETE")
                    print("=" * 100)
                    print()
                    print("Next steps:")
                    print("   1. Review generated GraphQL API code")
                    print("   2. Deploy GraphQL server (see Docker files in output)")
                    print("   3. Test GraphQL endpoints using generated test client")
                    return
                else:
                    print("✗ Failed to generate GraphQL API")
                    print(f"   Error: {graphql_result.get('error', 'Unknown error')}")
                    return
                    
            except Exception as e:
                print(f"✗ Error generating GraphQL API: {e}")
                traceback.print_exc()
                return
        else:
            print("⊘ GraphQL generation skipped (--skip-graphql flag)")
            return
    
    # Step 1: Get PRP (from file, update, or interactive planning)
    if update_prp:
        # Update existing PRP with modifications
        try:
            prp_text = await run_update_planning(
                client, update_prp, initial_intent, max_planning_turns
            )
        except Exception as e:
            print(f"✗ Error updating PRP: {e}")
            return
    elif prp_file:
        print(f"📄 Loading PRP from file: {prp_file}")
        print()
        try:
            prp_text = client.load_prp_from_file(prp_file)
            print("✓ PRP loaded successfully")
            print()
        except (FileNotFoundError, ValueError) as e:
            print(f"✗ Error loading PRP file: {e}")
            return
    else:
        # Interactive Planning (Q&A)
        prp_text = await run_interactive_planning(client, initial_intent, max_planning_turns)
    
    # Step 2: Discover source tables for PRP Section 9 targets
    discovery_results = await client.discover_source_tables_from_prp(
        prp_text=prp_text,
        max_results_per_query=max_results_per_query
    )
    
    target_tables = discovery_results.get("target_tables", [])
    
    if not target_tables:
        print("No target tables found in PRP Section 9. Exiting.")
        return
    
    print(f"📋 Target Tables from Section 9:")
    for i, target in enumerate(target_tables, 1):
        print(f"   {i}. {target['target_table_name']}")
        print(f"      Sources found: {len(target.get('discovered_tables', []))}")
    print()
    
    # Step 3: Generate queries for each target table (best effort)
    all_query_results = []
    successful_queries = 0
    failed_queries = 0
    
    for i, target in enumerate(target_tables, 1):
        print("=" * 100)
        print(f"GENERATING QUERIES {i}/{len(target_tables)}")
        print("=" * 100)
        print()
        
        try:
            query_results = await client.generate_queries_for_target(
                target_table_name=target["target_table_name"],
                target_description=target["target_description"],
                target_columns=target["target_columns"],
                source_datasets=target["discovered_tables"],
                max_queries=max_queries,
                max_iterations=max_iterations
            )
            
            all_query_results.append({
                "target_table": target["target_table_name"],
                "results": query_results
            })
            
            # Print results
            print_query_results(
                query_results,
                client.output_dir,
                target["target_table_name"]
            )
            
            if query_results.get("total_validated", 0) > 0:
                successful_queries += 1
            else:
                failed_queries += 1
            
        except Exception as e:
            print(f"✗ Error generating queries for {target['target_table_name']}: {e}")
            print(f"   Exception type: {type(e).__name__}")
            print(f"   Exception args: {e.args}")
            print(f"   Detailed traceback:")
            import traceback
            traceback.print_exc()
            failed_queries += 1
        
        print()
    
    # Step 4: Generate GraphQL API (optional)
    graphql_results = []
    if not skip_graphql and successful_queries > 0:
        print("=" * 100)
        print("GENERATING GRAPHQL API")
        print("=" * 100)
        print()
        
        # Use project_name if provided, otherwise derive from first target table
        if not project_name:
            project_name = target_tables[0]["target_table_name"].replace("_", "-")
        
        print(f"📦 Project name: {project_name}")
        print()
        
        # Generate GraphQL for each target with valid queries
        for result in all_query_results:
            target_name = result["target_table"]
            queries = result["results"].get("queries", [])
            
            # Extract API-level metadata from query generation results
            api_metadata = {
                "insight": result["results"].get("insight"),
                "summary": result["results"].get("summary"),
                "dataset_count": result["results"].get("dataset_count"),
                "execution_time_ms": result["results"].get("execution_time_ms"),
            }
            
            # Only generate if there are valid queries
            valid_count = sum(1 for q in queries if q.get("validation_status") == "valid")
            
            if valid_count > 0:
                try:
                    graphql_result = await client.generate_graphql_api(
                        target_name=target_name,
                        validated_queries=queries,
                        project_name=project_name,
                        validation_level="standard",
                        api_metadata=api_metadata
                    )
                    
                    print(f"DEBUG: Raw GraphQL response for {target_name}: {json.dumps(graphql_result, indent=2)}")
                    
                    graphql_results.append({
                        "target_table": target_name,
                        "result": graphql_result
                    })
                    
                    # Print result
                    if graphql_result.get("success"):
                        print(f"✓ GraphQL API generated successfully for {target_name}")
                        print(f"   Output: {graphql_result.get('output_path', 'N/A')}")
                        print(f"   Files: {len(graphql_result.get('files_generated', []))}")
                        if graphql_result.get("message"):
                            print(f"   {graphql_result['message']}")
                    else:
                        print(f"✗ Failed to generate GraphQL API for {target_name}")
                        print(f"   Error: {graphql_result.get('error', 'Unknown error')}")
                    print()
                    
                except Exception as e:
                    print(f"✗ Error generating GraphQL for {target_name}: {e}")
                    print()
            else:
                print(f"⊘ Skipping GraphQL for {target_name} (no valid queries)")
                print()
        
        # Save GraphQL results summary
        if graphql_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            graphql_file = client.output_dir / f"graphql_results_{timestamp}.json"
            graphql_file.write_text(json.dumps(graphql_results, indent=2))
            print(f"💾 Saved GraphQL results to: {graphql_file}")
            print()
    
    # Step 5: Final Summary
    print("=" * 100)
    print("WORKFLOW COMPLETE")
    print("=" * 100)
    print()
    print(f"📊 Summary:")
    print(f"   Target tables: {len(target_tables)}")
    print(f"   Query generation - ✓ Successful: {successful_queries}, ✗ Failed: {failed_queries}")
    
    if not skip_graphql and graphql_results:
        successful_graphql = sum(1 for r in graphql_results if r["result"].get("success"))
        failed_graphql = len(graphql_results) - successful_graphql
        print(f"   GraphQL generation - ✓ Successful: {successful_graphql}, ✗ Failed: {failed_graphql}")
    elif skip_graphql:
        print(f"   GraphQL generation - ⊘ Skipped")
    
    print()
    
    if successful_queries > 0:
        print("Next steps:")
        print("   1. Review generated queries in output/ directory")
        
        if not skip_graphql and any(r["result"].get("success") for r in graphql_results):
            print("   2. Review generated GraphQL API code")
            print("   3. Deploy GraphQL server (see Docker files in output)")
            print("   4. Test GraphQL endpoints using generated test client")
        else:
            print("   2. Execute queries in BigQuery to validate results")
            print("   3. Verify query outputs match target schemas")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Integrated workflow: Planning → Discovery → Query Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use defaults (interactive Q&A)
  python integrated_workflow_example.py
  
  # Custom initial intent
  python integrated_workflow_example.py \\
    --initial-intent "Analyze customer behavior patterns"
  
  # Load existing PRP from file (skip planning)
  python integrated_workflow_example.py \\
    --prp-file output/prp_20250101_120000.md
  
  # Update existing PRP with modifications
  python integrated_workflow_example.py \\
    --update-prp output/prp_20250101_120000.md \\
    -i "Add geographic breakdown by state and customer segment analysis"
  
  # Fast path: Load query results and generate GraphQL only (skip all other steps)
  python integrated_workflow_example.py \\
    --query-results-file output/backtest_evaluation_view_20251024_190934.json \\
    --project-name backtest-analysis
  
  # Control query generation parameters
  python integrated_workflow_example.py \\
    --max-queries 5 \\
    --max-iterations 15
        """
    )
    
    # Planning parameters
    parser.add_argument(
        "-i", "--initial-intent",
        type=str,
        help="Initial business intent for planning (e.g., 'Analyze customer behavior') or requested changes when using --update-prp"
    )
    
    parser.add_argument(
        "-p", "--prp-file",
        type=str,
        help="Path to existing PRP markdown file (skips planning phase)"
    )
    
    parser.add_argument(
        "-u", "--update-prp",
        type=str,
        help="Path to existing PRP markdown file to update (requires -i for changes description)"
    )
    
    parser.add_argument(
        "-q", "--query-results-file",
        type=str,
        help="Path to query results JSON file (skips to GraphQL generation only)"
    )
    
    # Planning configuration
    parser.add_argument(
        "--max-planning-turns",
        type=int,
        default=10,
        help="Maximum Q&A turns in planning phase (default: 10)"
    )
    
    # Discovery configuration
    parser.add_argument(
        "--max-results-per-query",
        type=int,
        default=5,
        help="Maximum source tables to discover per search query (default: 5)"
    )
    
    # Query generation configuration
    parser.add_argument(
        "--max-queries",
        type=int,
        default=3,
        help="Maximum number of queries to generate per target (default: 3)"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum refinement iterations per query (default: 10)"
    )
    
    parser.add_argument(
        "--max-query-wait",
        type=float,
        default=900.0,
        help="Maximum time to wait for query generation in seconds (default: 900 = 15 minutes)"
    )
    
    # Service URLs
    parser.add_argument(
        "--planning-url",
        type=str,
        default="http://localhost:8082",
        help="Data planning agent URL (default: http://localhost:8082)"
    )
    
    parser.add_argument(
        "--discovery-url",
        type=str,
        default="http://localhost:8080",
        help="Data discovery agent URL (default: http://localhost:8080)"
    )
    
    parser.add_argument(
        "--query-gen-url",
        type=str,
        default="http://localhost:8081",
        help="Query generation agent URL (default: http://localhost:8081)"
    )
    
    parser.add_argument(
        "--graphql-url",
        type=str,
        default="http://localhost:8083",
        help="Data GraphQL agent URL (default: http://localhost:8083)"
    )
    
    # GraphQL configuration
    parser.add_argument(
        "--skip-graphql",
        action="store_true",
        help="Skip GraphQL API generation step"
    )
    
    parser.add_argument(
        "--project-name",
        type=str,
        help="Project name for GraphQL lineage tracking (defaults to first target table name)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Parse command-line arguments
    args = parse_args()
    
    print()
    if args.initial_intent:
        print("Running with custom parameters...")
    else:
        print("Running with default parameters...")
        print("(Use --help to see available options)")
    print()
    
    # Run main workflow
    asyncio.run(main(
        initial_intent=args.initial_intent,
        prp_file=args.prp_file,
        update_prp=args.update_prp,
        query_results_file=args.query_results_file,
        max_results_per_query=args.max_results_per_query,
        max_planning_turns=args.max_planning_turns,
        max_queries=args.max_queries,
        max_iterations=args.max_iterations,
        max_query_wait_seconds=args.max_query_wait,
        planning_url=args.planning_url,
        discovery_url=args.discovery_url,
        query_gen_url=args.query_gen_url,
        graphql_url=args.graphql_url,
        skip_graphql=args.skip_graphql,
        project_name=args.project_name
    ))

