"""
Integrated Workflow Example: Data Discovery → Query Generation

This example demonstrates the complete workflow:
1. Discover datasets using data-discovery-agent with a natural language query
2. Get datasets in BigQuery schema format (DiscoveredAssetDict)
3. Generate SQL queries using query-generation-agent with a data science insight
4. Display the validated queries ready for execution

Prerequisites:
- data-discovery-agent running on http://localhost:8080
- query-generation-agent running on http://localhost:8081

Usage:
    # Use default query and insight
    python integrated_workflow_example.py
    
    # Specify custom query and insight
    python integrated_workflow_example.py \
        --query "customer analytics tables" \
        --insight "What is the average order value by product category?"
    
    # Add filters
    python integrated_workflow_example.py \
        --query "revenue tables" \
        --insight "Calculate monthly revenue trend" \
        --page-size 5 \
        --environment prod \
        --no-pii
"""

import argparse
import asyncio
import json
from typing import Any, Dict, List, Optional

import httpx


class IntegratedMCPClient:
    """Client for integrated data discovery and query generation workflow."""
    
    def __init__(
        self,
        discovery_url: str = "http://localhost:8080",
        query_gen_url: str = "http://localhost:8081",
        timeout: float = 300.0
    ):
        """
        Initialize integrated MCP client.
        
        Args:
            discovery_url: Data discovery agent base URL
            query_gen_url: Query generation agent base URL
            timeout: Request timeout in seconds
        """
        self.discovery_url = discovery_url
        self.query_gen_url = query_gen_url
        self.timeout = timeout
    
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
        max_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Generate SQL queries from insight and datasets.
        
        Args:
            insight: Data science insight or question to answer
            datasets: List of datasets from discovery (in BigQuery schema format)
            max_queries: Maximum number of queries to generate
            max_iterations: Maximum refinement iterations per query
            
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
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.query_gen_url}/mcp/call-tool",
                json=request
            )
            response.raise_for_status()
            
            result = response.json()
            
            print("✓ Query generation complete")
            print()
            
            return result
    
    async def check_health(self) -> Dict[str, bool]:
        """
        Check health of both services.
        
        Returns:
            Dict with health status of each service
        """
        status = {
            "discovery_agent": False,
            "query_generation_agent": False
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
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


def print_query_results(results: Dict[str, Any]) -> None:
    """
    Pretty print query generation results.
    
    Args:
        results: Query generation results
    """
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
    query: Optional[str] = None,
    insight: Optional[str] = None,
    page_size: int = 3,
    has_pii: Optional[bool] = None,
    has_phi: Optional[bool] = None,
    environment: Optional[str] = None,
    max_queries: int = 3,
    max_iterations: int = 10,
    discovery_url: str = "http://localhost:8080",
    query_gen_url: str = "http://localhost:8081"
) -> None:
    """
    Run the integrated workflow example.
    
    Args:
        query: Discovery query (natural language)
        insight: Data science insight or question
        page_size: Number of datasets to retrieve
        has_pii: Filter for PII tables (True/False/None)
        has_phi: Filter for PHI tables (True/False/None)
        environment: Filter by environment (prod, staging, dev)
        max_queries: Maximum number of queries to generate
        max_iterations: Maximum refinement iterations per query
        discovery_url: Data discovery agent URL
        query_gen_url: Query generation agent URL
    """
    print("=" * 100)
    print("INTEGRATED WORKFLOW: DATA DISCOVERY → QUERY GENERATION")
    print("=" * 100)
    print()
    
    # Use defaults if not provided
    if query is None:
        query = "customer transaction tables"
        print(f"ℹ️  Using default query: '{query}'")
    if insight is None:
        insight = "What are the top 10 customers by total transaction amount in the last 30 days?"
        print(f"ℹ️  Using default insight: '{insight}'")
    
    if query is None or insight is None:
        print()
    
    # Initialize client
    client = IntegratedMCPClient(
        discovery_url=discovery_url,
        query_gen_url=query_gen_url
    )
    
    # Step 1: Check health
    print("🏥 Checking service health...")
    health = await client.check_health()
    
    if health["discovery_agent"]:
        print("   ✓ Data Discovery Agent: healthy")
    else:
        print("   ✗ Data Discovery Agent: unavailable")
        print()
        print("   Please start the data-discovery-agent:")
        print("   cd /home/user/git/data-discovery-agent")
        print("   poetry run python -m data_discovery_agent.mcp")
        return
    
    if health["query_generation_agent"]:
        print("   ✓ Query Generation Agent: healthy")
    else:
        print("   ✗ Query Generation Agent: unavailable")
        print()
        print("   Please start the query-generation-agent:")
        print("   cd /home/user/git/query-generation-agent")
        print("   poetry run python -m query_generation_agent.mcp")
        return
    
    print()
    
    # Step 2: Discover datasets
    datasets = await client.discover_datasets(
        query=query,
        page_size=page_size,
        has_pii=has_pii,
        has_phi=has_phi,
        environment=environment
    )
    
    if not datasets:
        print("No datasets found. Please check:")
        print("   - Data discovery agent has indexed data")
        print("   - Your search query matches available tables")
        print("   - Filters are not too restrictive")
        return
    
    # Step 3: Generate queries
    results = await client.generate_queries(
        insight=insight,
        datasets=datasets,
        max_queries=max_queries,
        max_iterations=max_iterations
    )
    
    # Step 4: Display results
    print_query_results(results)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Integrated workflow: Discover datasets and generate SQL queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use defaults
  python integrated_workflow_example.py
  
  # Custom query and insight
  python integrated_workflow_example.py \\
    --query "customer analytics tables" \\
    --insight "What is the average order value by product category?"
  
  # With filters
  python integrated_workflow_example.py \\
    --query "revenue tables" \\
    --insight "Calculate monthly revenue trend" \\
    --page-size 5 \\
    --environment prod \\
    --no-pii
  
  # More query alternatives
  python integrated_workflow_example.py \\
    --query "transaction data" \\
    --insight "Show daily transaction volume" \\
    --max-queries 5 \\
    --max-iterations 15
        """
    )
    
    # Discovery parameters
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Natural language discovery query (e.g., 'customer transaction tables')"
    )
    
    # Query generation parameters
    parser.add_argument(
        "-i", "--insight",
        type=str,
        help="Data science insight or question (e.g., 'What is the top 10 customers by revenue?')"
    )
    
    # Discovery filters
    parser.add_argument(
        "--page-size",
        type=int,
        default=3,
        help="Number of datasets to retrieve (default: 3)"
    )
    
    parser.add_argument(
        "--pii",
        action="store_true",
        help="Only include tables with PII data"
    )
    
    parser.add_argument(
        "--no-pii",
        action="store_true",
        help="Exclude tables with PII data"
    )
    
    parser.add_argument(
        "--phi",
        action="store_true",
        help="Only include tables with PHI data"
    )
    
    parser.add_argument(
        "--no-phi",
        action="store_true",
        help="Exclude tables with PHI data"
    )
    
    parser.add_argument(
        "--environment",
        type=str,
        choices=["prod", "production", "staging", "dev", "development"],
        help="Filter by environment"
    )
    
    # Query generation parameters
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
    
    # Determine PII/PHI filters
    has_pii = None
    if args.pii:
        has_pii = True
    elif args.no_pii:
        has_pii = False
    
    has_phi = None
    if args.phi:
        has_phi = True
    elif args.no_phi:
        has_phi = False
    
    print()
    if args.query or args.insight:
        print("Running with custom parameters...")
    else:
        print("Running with default parameters...")
        print("(Use --help to see available options)")
    print()
    
    # Run main workflow
    asyncio.run(main(
        query=args.query,
        insight=args.insight,
        page_size=args.page_size,
        has_pii=has_pii,
        has_phi=has_phi,
        environment=args.environment,
        max_queries=args.max_queries,
        max_iterations=args.max_iterations,
        discovery_url=args.discovery_url,
        query_gen_url=args.query_gen_url
    ))

