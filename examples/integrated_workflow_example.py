"""
Integrated Workflow Example: Planning → Discovery → Query Generation

This example demonstrates the complete end-to-end workflow:
1. Gather requirements using data-planning-agent with interactive Q&A
2. Generate a structured Data Product Requirement Prompt (PRP)
3. Extract Section 9 data requirements and discover source tables
4. Generate SELECT queries for each target table specification
5. Save query results as JSON files in output directory

Prerequisites:
- data-planning-agent running on http://localhost:8082 (not needed if using --prp-file)
- data-discovery-agent running on http://localhost:8080
- query-generation-agent running on http://localhost:8081

Usage:
    # Use default initial intent (interactive Q&A)
    python integrated_workflow_example.py
    
    # Specify custom initial intent
    python integrated_workflow_example.py \
        --initial-intent "Analyze customer transaction patterns"
    
    # Load existing PRP from file (skip planning)
    python integrated_workflow_example.py \
        --prp-file output/prp_20250101_120000.md
    
    # Control query generation parameters
    python integrated_workflow_example.py \
        --max-queries 5 \
        --max-iterations 15
"""

import argparse
import asyncio
import json
import os
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import httpx


class IntegratedMCPClient:
    """Client for integrated planning, data discovery and query generation workflow."""
    
    def __init__(
        self,
        planning_url: str = "http://localhost:8082",
        discovery_url: str = "http://localhost:8080",
        query_gen_url: str = "http://localhost:8081",
        timeout: float = 600.0,
        output_dir: str = "output"
    ):
        """
        Initialize integrated MCP client.
        
        Args:
            planning_url: Data planning agent base URL
            discovery_url: Data discovery agent base URL
            query_gen_url: Query generation agent base URL
            timeout: Request timeout in seconds
            output_dir: Directory to save output files (default: "output")
        """
        self.planning_url = planning_url
        self.discovery_url = discovery_url
        self.query_gen_url = query_gen_url
        self.timeout = timeout
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
            async with httpx.AsyncClient(timeout=self.timeout) as client:
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
        Generate SELECT queries for a target table specification.
        
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
                "max_iterations": max_iterations
            }
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.query_gen_url}/mcp/call-tool",
                json=request
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Handle two response formats:
            # 1. MCP protocol wrapped: {"result": {"content": [{"text": "..."}]}}
            # 2. Direct JSON: {"queries": [...], "total_attempted": ...}
            
            # Check if it's MCP protocol wrapped
            if "result" in result and "content" in result["result"]:
                content_list = result["result"]["content"]
                if len(content_list) > 0 and "text" in content_list[0]:
                    result_text = content_list[0]["text"]
                    try:
                        return json.loads(result_text)
                    except json.JSONDecodeError as e:
                        print(f"✗ Error parsing MCP wrapped response: {e}")
                        print(f"Response text: {result_text[:500]}")
                        return {
                            "queries": [],
                            "total_attempted": 0,
                            "total_validated": 0,
                            "execution_time_ms": 0,
                            "warnings": ["JSON parse error"]
                        }
            
            # Check if it's direct JSON format (already unwrapped)
            elif "queries" in result:
                return result
            
            # Unknown format
            else:
                print(f"✗ Unexpected response structure from query generation agent")
                print(f"Response keys: {list(result.keys())}")
                print(f"Full response: {json.dumps(result, indent=2)[:500]}")
                return {
                    "queries": [],
                    "total_attempted": 0,
                    "total_validated": 0,
                    "execution_time_ms": 0,
                    "warnings": ["No queries generated"]
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
            "query_generation_agent": False
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
        max_turns: Maximum Q&A turns
    
    Returns:
        Generated PRP text
    """
    print(f"🎯 Starting planning session...")
    print(f"   Initial intent: '{initial_intent}'")
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
    
    # Generate PRP
    if is_complete:
        print("✓ Requirements gathering complete!")
    else:
        print("⚠️  Max turns reached, proceeding with available information")
    
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
    max_results_per_query: int = 5,
    max_planning_turns: int = 10,
    max_queries: int = 3,
    max_iterations: int = 10,
    planning_url: str = "http://localhost:8082",
    discovery_url: str = "http://localhost:8080",
    query_gen_url: str = "http://localhost:8081"
) -> None:
    """
    Run the integrated workflow example.
    
    Generates SELECT queries for all target tables defined in PRP Section 9.
    
    Args:
        initial_intent: Initial business intent for planning
        prp_file: Path to existing PRP file (skips planning phase)
        max_results_per_query: Maximum source tables to discover per target
        max_planning_turns: Maximum Q&A turns in planning phase
        max_queries: Maximum number of queries to generate per target
        max_iterations: Maximum refinement iterations per query
        planning_url: Data planning agent URL
        discovery_url: Data discovery agent URL
        query_gen_url: Query generation agent URL
    """
    print("=" * 100)
    print("INTEGRATED WORKFLOW: PLANNING → DISCOVERY → QUERY GENERATION")
    print("=" * 100)
    print()
    
    # Check for mutual exclusivity and warn if both provided
    if prp_file and initial_intent:
        print("⚠️  Both --prp-file and --initial-intent provided. Using --prp-file (skipping planning).")
        print()
    
    # Use default if not provided and no prp_file
    if initial_intent is None and prp_file is None:
        initial_intent = "We want to analyze customer transaction patterns to identify high-value customers"
        print(f"ℹ️  Using default intent: '{initial_intent}'")
        print()
    
    # Initialize client with output directory
    output_dir = "output"
    client = IntegratedMCPClient(
        planning_url=planning_url,
        discovery_url=discovery_url,
        query_gen_url=query_gen_url,
        output_dir=output_dir
    )
    
    print(f"📁 Output directory: {output_dir}/")
    print()
    
    # Check health (all 3 agents or skip planning if using prp_file)
    print("🏥 Checking service health...")
    health = await client.check_health()
    
    # Only check planning agent if not using prp_file
    if not prp_file:
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
    
    print()
    
    # Step 1: Get PRP (either from file or interactive planning)
    if prp_file:
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
            failed_queries += 1
        
        print()
    
    # Step 4: Final Summary
    print("=" * 100)
    print("WORKFLOW COMPLETE")
    print("=" * 100)
    print()
    print(f"📊 Summary:")
    print(f"   Target tables: {len(target_tables)}")
    print(f"   ✓ Successful: {successful_queries}")
    print(f"   ✗ Failed: {failed_queries}")
    print()
    
    if successful_queries > 0:
        print("Next steps:")
        print("   1. Review generated queries in output/ directory")
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
        help="Initial business intent for planning (e.g., 'Analyze customer behavior')"
    )
    
    parser.add_argument(
        "-p", "--prp-file",
        type=str,
        help="Path to existing PRP markdown file (skips planning phase)"
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
        max_results_per_query=args.max_results_per_query,
        max_planning_turns=args.max_planning_turns,
        max_queries=args.max_queries,
        max_iterations=args.max_iterations,
        planning_url=args.planning_url,
        discovery_url=args.discovery_url,
        query_gen_url=args.query_gen_url
    ))

