"""
Integrated Workflow Example: Planning → Data Discovery → Query Generation

This example demonstrates the complete end-to-end workflow:
1. Gather requirements using data-planning-agent with interactive Q&A
2. Generate a structured Data Product Requirement Prompt (PRP)
3. Discover datasets using data-discovery-agent based on the PRP
4. Generate SQL queries using query-generation-agent with the datasets and insight

Prerequisites:
- data-planning-agent running on http://localhost:8082
- data-discovery-agent running on http://localhost:8080
- query-generation-agent running on http://localhost:8081

Usage:
    # Use default initial intent and insight (interactive Q&A)
    python integrated_workflow_example.py
    
    # Specify custom initial intent and insight
    python integrated_workflow_example.py \
        --initial-intent "Analyze customer transaction patterns" \
        --query-insight "What are the top 10 customers by revenue?"
    
    # Control planning turns and results
    python integrated_workflow_example.py \
        --initial-intent "Track product performance" \
        --max-planning-turns 5 \
        --max-results 10 \
        --max-queries 3
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx


class IntegratedMCPClient:
    """Client for integrated planning, data discovery and query generation workflow."""
    
    def __init__(
        self,
        planning_url: str = "http://localhost:8082",
        discovery_url: str = "http://localhost:8080",
        query_gen_url: str = "http://localhost:8081",
        timeout: float = 300.0,
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
    
    async def start_planning_session(
        self,
        initial_intent: str
    ) -> tuple[str, str]:
        """
        Start a planning session with initial business intent.
        
        Args:
            initial_intent: High-level business goal or intent
            
        Returns:
            Tuple of (session_id, initial_questions)
        """
        request = {
            "name": "start_planning_session",
            "arguments": {
                "initial_intent": initial_intent
            }
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.planning_url}/mcp/call-tool",
                json=request
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract session ID and questions from response
            if "result" in result and len(result["result"]) > 0:
                result_text = result["result"][0]["text"]
                
                # Parse session ID from response (format: **Session ID:** `{uuid}`)
                import re
                session_match = re.search(r'`([a-f0-9-]+)`', result_text)
                if not session_match:
                    raise ValueError("Could not extract session ID from response")
                
                session_id = session_match.group(1)
                
                # Extract questions (everything after the first --- and before the final ---)
                parts = result_text.split("---")
                if len(parts) >= 2:
                    questions = parts[1].strip()
                else:
                    questions = result_text
                
                return (session_id, questions)
            else:
                raise ValueError("No response from planning agent")
    
    async def continue_planning_conversation(
        self,
        session_id: str,
        user_response: str
    ) -> tuple[str, bool]:
        """
        Continue a planning conversation with user response.
        
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
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.planning_url}/mcp/call-tool",
                json=request
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract questions and completion status
            if "result" in result and len(result["result"]) > 0:
                result_text = result["result"][0]["text"]
                
                # Check if requirements are complete
                is_complete = "requirements gathering complete" in result_text.lower() or \
                              "Requirements gathering is complete" in result_text or \
                              "requirements are complete" in result_text.lower()
                
                return (result_text, is_complete)
            else:
                raise ValueError("No response from planning agent")
    
    async def generate_prp(
        self,
        session_id: str
    ) -> str:
        """
        Generate Data PRP from completed planning session.
        
        Args:
            session_id: Planning session ID
            
        Returns:
            Generated PRP text in markdown format
        """
        request = {
            "name": "generate_data_prp",
            "arguments": {
                "session_id": session_id,
                "save_to_file": False
            }
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.planning_url}/mcp/call-tool",
                json=request
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract PRP text from response
            if "result" in result and len(result["result"]) > 0:
                result_text = result["result"][0]["text"]
                
                # Extract markdown content between code fences if present
                if "```markdown" in result_text:
                    import re
                    match = re.search(r'```markdown\n(.*?)\n```', result_text, re.DOTALL)
                    if match:
                        return match.group(1)
                
                # Otherwise return the full text
                return result_text
            else:
                raise ValueError("No PRP generated from planning agent")
    
    async def discover_datasets_from_prp(
        self,
        prp_text: str,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Discover datasets from PRP text.
        
        Args:
            prp_text: Generated PRP markdown text
            max_results: Maximum datasets to return
            
        Returns:
            List of discovered datasets with metadata
        """
        print(f"🔍 Discovering datasets from PRP...")
        print()
        
        request = {
            "name": "discover_datasets_for_prp",
            "arguments": {
                "prp_text": prp_text,
                "max_results": max_results
            }
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.discovery_url}/mcp/call-tool",
                json=request
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract datasets and metadata
            if "result" in result and len(result["result"]) > 0:
                result_text = result["result"][0]["text"]
                data = json.loads(result_text)
                
                # Save discovery JSON to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                discovery_file = self.output_dir / f"discovery_results_{timestamp}.json"
                discovery_file.write_text(json.dumps(data, indent=2))
                print(f"💾 Saved discovery results to: {discovery_file}")
                print()
                
                # Print the JSON response from data discovery agent
                print("=" * 80)
                print("DATA DISCOVERY AGENT JSON RESPONSE")
                print("=" * 80)
                print(json.dumps(data, indent=2))
                print("=" * 80)
                print()
                
                datasets = data.get("datasets", [])
                metadata = data.get("discovery_metadata", {})
                
                print(f"✓ Found {len(datasets)} datasets")
                
                # Show summary
                summary = metadata.get("summary", {})
                if summary:
                    print(f"   Queries executed: {summary.get('total_queries_generated', 0)}")
                    print(f"   Candidates found: {summary.get('total_candidates_found', 0)}")
                    print(f"   Execution time: {summary.get('total_execution_time_ms', 0):.0f}ms")
                
                for i, ds in enumerate(datasets, 1):
                    print(f"   {i}. {ds['project_id']}.{ds['dataset_id']}.{ds['table_id']}")
                print()
                
                return datasets
            else:
                print("✗ No datasets found")
                return []
    
    async def discover_datasets(
        self,
        query: str,
        page_size: int = 5,
        has_pii: Optional[bool] = None,
        has_phi: Optional[bool] = None,
        environment: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Discover datasets using natural language query.
        
        Args:
            query: Natural language search query (e.g., "customer transaction tables")
            page_size: Number of results to return
            has_pii: Filter by PII data presence
            has_phi: Filter by PHI data presence
            environment: Filter by environment (prod, staging, dev)
            
        Returns:
            List of discovered datasets in BigQuery schema format
        """
        print(f"🔍 Discovering datasets with query: '{query}'")
        print(f"   Filters: page_size={page_size}, has_pii={has_pii}, has_phi={has_phi}, environment={environment}")
        print()
        
        request = {
            "name": "get_datasets_for_query_generation",
            "arguments": {
                "query": query,
                "page_size": page_size
            }
        }
        
        # Add optional filters
        if has_pii is not None:
            request["arguments"]["has_pii"] = has_pii
        if has_phi is not None:
            request["arguments"]["has_phi"] = has_phi
        if environment:
            request["arguments"]["environment"] = environment
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.discovery_url}/mcp/call-tool",
                json=request
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract datasets from response
            if "result" in result and len(result["result"]) > 0:
                result_text = result["result"][0]["text"]
                data = json.loads(result_text)
                datasets = data.get("datasets", [])
                
                print(f"✓ Found {len(datasets)} datasets")
                for i, ds in enumerate(datasets, 1):
                    print(f"   {i}. {ds['project_id']}.{ds['dataset_id']}.{ds['table_id']}")
                    print(f"      Type: {ds['table_type']}, Rows: {ds.get('row_count', 'N/A')}, Columns: {ds.get('column_count', 'N/A')}")
                print()
                
                return datasets
            else:
                print("✗ No datasets found")
                return []
    
    async def generate_queries(
        self,
        insight: str,
        datasets: List[Dict[str, Any]],
        max_queries: int = 3,
        max_iterations: int = 10,
        poll_interval: int = 5
    ) -> Dict[str, Any]:
        """
        Generate SQL queries from insight and datasets using async pattern.
        
        Args:
            insight: Data science insight or question to answer
            datasets: List of datasets from discovery (in BigQuery schema format)
            max_queries: Maximum number of queries to generate
            max_iterations: Maximum refinement iterations per query
            poll_interval: Seconds between status checks (default: 5)
            
        Returns:
            Query generation results with validated SQL queries
        """
        print(f"🤖 Generating queries for insight:")
        print(f"   '{insight}'")
        print()
        print(f"   Using {len(datasets)} dataset(s)")
        print(f"   Max queries: {max_queries}, Max iterations: {max_iterations}")
        print()
        print("   This may take 1-2 minutes (generating, validating, and refining)...")
        print()
        
        request = {
            "name": "generate_queries",
            "arguments": {
                "insight": insight,
                "datasets": datasets,
                "max_queries": max_queries,
                "max_iterations": max_iterations
            }
        }
        
        # Use a longer timeout client for the entire operation, but short timeouts for individual requests
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=10.0)) as client:
            # 1. Try async endpoint first
            try:
                response = await client.post(
                    f"{self.query_gen_url}/mcp/call-tool-async",
                    json=request,
                    timeout=30.0  # Give more time for task creation
                )
                
                if response.status_code == 202:
                    # Async endpoint available - use polling pattern
                    task_data = response.json()
                    task_id = task_data["task_id"]
                    status_url = f"{self.query_gen_url}{task_data['status_url']}"
                    
                    print(f"   Task ID: {task_id}")
                    print(f"   Polling for status every {poll_interval}s...")
                    print()
                    
                    # 2. Poll for completion
                    start_time = asyncio.get_event_loop().time()
                    
                    while True:
                        elapsed = asyncio.get_event_loop().time() - start_time
                        
                        if elapsed > self.timeout:
                            raise TimeoutError(f"Task did not complete within {self.timeout}s")
                        
                        await asyncio.sleep(poll_interval)
                        
                        # Retry logic for status checks (server might be slow under load)
                        max_retries = 3
                        status_data = None
                        
                        for attempt in range(max_retries):
                            try:
                                # Increased timeout for status checks
                                status_response = await client.get(status_url, timeout=30.0)
                                status_data = status_response.json()
                                break
                            except httpx.ReadTimeout:
                                if attempt < max_retries - 1:
                                    print(f"   ⚠️  Status check timed out, retrying... ({attempt + 1}/{max_retries})")
                                    await asyncio.sleep(2)
                                else:
                                    print(f"   ✗ Status check failed after {max_retries} attempts")
                                    raise
                        
                        if status_data is None:
                            raise Exception("Failed to get status after retries")
                        
                        status = status_data["status"]
                        print(f"   Status: {status} ({elapsed:.0f}s elapsed)")
                        
                        if status == "completed":
                            # 3. Retrieve result
                            result_url = f"{self.query_gen_url}{status_data['result_url']}"
                            
                            # Also retry result retrieval
                            for attempt in range(max_retries):
                                try:
                                    result_response = await client.get(result_url, timeout=60.0)
                                    result_json = result_response.json()
                                    break
                                except httpx.ReadTimeout:
                                    if attempt < max_retries - 1:
                                        print(f"   ⚠️  Result retrieval timed out, retrying... ({attempt + 1}/{max_retries})")
                                        await asyncio.sleep(2)
                                    else:
                                        raise
                            
                            print()
                            print("✓ Query generation complete")
                            print()
                            
                            return result_json["result"]
                        
                        elif status == "failed":
                            error = status_data.get("error", "Unknown error")
                            raise Exception(f"Task failed: {error}")
                
                # If not 202, fall through to sync endpoint
            except httpx.HTTPStatusError as e:
                if e.response.status_code != 404:
                    raise
                # 404 means async endpoint doesn't exist, fall back to sync
            
            # Fallback to sync endpoint
            print("   ℹ️  Using synchronous endpoint (async not available)")
            response = await client.post(
                f"{self.query_gen_url}/mcp/call-tool",
                json=request,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            print("✓ Query generation complete")
            print()
            
            return result
    
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


def print_query_results(results: Dict[str, Any], output_dir: Path) -> None:
    """
    Pretty print query generation results.
    
    Args:
        results: Query generation results
        output_dir: Directory to save output files
    """
    # Save query results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    query_file = output_dir / f"query_results_{timestamp}.json"
    query_file.write_text(json.dumps(results, indent=2))
    print(f"💾 Saved query results to: {query_file}")
    print()
    
    print("=" * 100)
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
    
    # Summary recommendations
    valid_count = sum(1 for q in queries if q.get("validation_status") == "valid")
    
    if valid_count > 0:
        print()
        print(f"✓ {valid_count} valid {'query' if valid_count == 1 else 'queries'} ready for execution in BigQuery!")
        print()
        print("Next steps:")
        print("   1. Review the SQL queries above")
        print("   2. Execute in BigQuery console or via API")
        print("   3. Analyze results")
    else:
        print()
        print("⚠️  No valid queries generated. Consider:")
        print("   - Refining your insight to be more specific")
        print("   - Checking if datasets contain relevant fields")
        print("   - Reviewing error messages for guidance")


async def main(
    initial_intent: Optional[str] = None,
    insight: Optional[str] = None,
    max_results: int = 5,
    max_queries: int = 3,
    max_iterations: int = 10,
    max_planning_turns: int = 10,
    planning_url: str = "http://localhost:8082",
    discovery_url: str = "http://localhost:8080",
    query_gen_url: str = "http://localhost:8081"
) -> None:
    """
    Run the integrated workflow example.
    
    Args:
        initial_intent: Initial business intent for planning
        insight: Data science insight for query generation
        max_results: Maximum datasets to discover from PRP
        max_queries: Maximum number of queries to generate
        max_iterations: Maximum refinement iterations per query
        max_planning_turns: Maximum Q&A turns in planning phase
        planning_url: Data planning agent URL
        discovery_url: Data discovery agent URL
        query_gen_url: Query generation agent URL
    """
    print("=" * 100)
    print("INTEGRATED WORKFLOW: PLANNING → DISCOVERY → QUERY GENERATION")
    print("=" * 100)
    print()
    
    # Use defaults if not provided
    if initial_intent is None:
        initial_intent = "We want to analyze customer transaction patterns to identify high-value customers"
        print(f"ℹ️  Using default intent: '{initial_intent}'")
    if insight is None:
        insight = "What are the top 10 customers by total transaction amount in the last 30 days?"
        print(f"ℹ️  Using default insight: '{insight}'")
    
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
    
    # Check health (all 3 agents)
    print("🏥 Checking service health...")
    health = await client.check_health()
    
    # Check all three services
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
    
    # Step 1: Interactive Planning (Q&A)
    prp_text = await run_interactive_planning(client, initial_intent, max_planning_turns)
    
    # Step 2: Discover datasets from PRP
    datasets = await client.discover_datasets_from_prp(
        prp_text=prp_text,
        max_results=max_results
    )
    
    if not datasets:
        print("No datasets found. Exiting.")
        return
    
    # Step 3: Generate queries
    results = await client.generate_queries(
        insight=insight,
        datasets=datasets,
        max_queries=max_queries,
        max_iterations=max_iterations
    )
    
    # Step 4: Display results
    print_query_results(results, client.output_dir)


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
  
  # Custom initial intent and insight
  python integrated_workflow_example.py \\
    --initial-intent "Analyze customer behavior patterns" \\
    --query-insight "What are the top customer segments by revenue?"
  
  # Control planning and query generation
  python integrated_workflow_example.py \\
    --initial-intent "Track product performance metrics" \\
    --max-planning-turns 5 \\
    --max-results 10 \\
    --max-queries 3 \\
    --max-iterations 15
        """
    )
    
    # Planning parameters
    parser.add_argument(
        "-i", "--initial-intent",
        type=str,
        help="Initial business intent for planning (e.g., 'Analyze customer behavior')"
    )
    
    # Query generation parameters
    parser.add_argument(
        "-q", "--query-insight",
        type=str,
        help="Data science insight for query generation (e.g., 'Top 10 customers by revenue')"
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
        "--max-results",
        type=int,
        default=5,
        help="Maximum datasets to discover from PRP (default: 5)"
    )
    
    # Query generation configuration
    parser.add_argument(
        "--max-queries",
        type=int,
        default=3,
        help="Maximum number of queries to generate (default: 3)"
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
    # Parse command-line arguments
    args = parse_args()
    
    print()
    if args.initial_intent or args.query_insight:
        print("Running with custom parameters...")
    else:
        print("Running with default parameters...")
        print("(Use --help to see available options)")
    print()
    
    # Run main workflow
    asyncio.run(main(
        initial_intent=args.initial_intent,
        insight=args.query_insight,
        max_results=args.max_results,
        max_queries=args.max_queries,
        max_iterations=args.max_iterations,
        max_planning_turns=args.max_planning_turns,
        planning_url=args.planning_url,
        discovery_url=args.discovery_url,
        query_gen_url=args.query_gen_url
    ))

