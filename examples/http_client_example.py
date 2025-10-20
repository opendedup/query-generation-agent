"""
Example: Using Query Generation Agent via HTTP transport

This example shows how to call the MCP server running in HTTP mode.
"""

import asyncio
import json

import httpx


async def main() -> None:
    """Run example HTTP client."""
    
    # Server URL (adjust as needed)
    base_url = "http://localhost:8081"
    
    # Example request data
    request = {
        "name": "generate_queries",
        "arguments": {
            "insight": "What is the average transaction value by payment method for Q4 2024?",
            "datasets": [
                {
                    "project_id": "my-project",
                    "dataset_id": "sales",
                    "table_id": "transactions",
                    "asset_type": "table",
                    "row_count": 1000000,
                    "size_bytes": 536870912,
                    "column_count": 8,
                    "schema_fields": [
                        {
                            "name": "transaction_id",
                            "type": "STRING",
                            "description": "Unique transaction identifier"
                        },
                        {
                            "name": "customer_id",
                            "type": "STRING",
                            "description": "Customer identifier"
                        },
                        {
                            "name": "payment_method",
                            "type": "STRING",
                            "description": "Payment method used"
                        },
                        {
                            "name": "amount",
                            "type": "FLOAT64",
                            "description": "Transaction amount in USD"
                        },
                        {
                            "name": "timestamp",
                            "type": "TIMESTAMP",
                            "description": "Transaction timestamp"
                        },
                        {
                            "name": "status",
                            "type": "STRING",
                            "description": "Transaction status"
                        },
                        {
                            "name": "currency",
                            "type": "STRING",
                            "description": "Currency code"
                        },
                        {
                            "name": "country",
                            "type": "STRING",
                            "description": "Customer country code"
                        }
                    ],
                    "full_markdown": """# Transactions Table

## Overview
Contains all customer transactions.

## Schema
- transaction_id: Unique identifier
- customer_id: Customer reference
- payment_method: Payment type
- amount: Transaction value in USD
- timestamp: Transaction time
- status: completed, pending, failed
- currency: Currency code
- country: Country code

## Usage
Filter by status='completed' for successful transactions.
Use timestamp for date range queries.
""",
                    "has_pii": True,
                    "has_phi": False,
                    "environment": "prod",
                    "owner_email": "data-team@example.com",
                    "tags": ["sales", "transactions"]
                }
            ],
            "max_queries": 3,
            "max_iterations": 10
        }
    }
    
    print("=" * 80)
    print("Query Generation Agent - HTTP Client Example")
    print("=" * 80)
    print()
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            # Check server health
            print("Checking server health...")
            health_response = await client.get(f"{base_url}/health")
            health_response.raise_for_status()
            print(f"✓ Server is healthy: {health_response.json()}")
            print()
            
            # List available tools
            print("Listing available tools...")
            tools_response = await client.get(f"{base_url}/mcp/tools")
            tools_response.raise_for_status()
            tools = tools_response.json()
            print(f"✓ Available tools: {len(tools['tools'])}")
            for tool in tools["tools"]:
                print(f"  - {tool['name']}: {tool['description'][:100]}...")
            print()
            
            # Call generate_queries tool
            print(f"Generating queries for insight:")
            print(f"  '{request['arguments']['insight']}'")
            print()
            print("This may take a minute or two...")
            print()
            
            call_response = await client.post(
                f"{base_url}/mcp/call-tool",
                json=request
            )
            call_response.raise_for_status()
            result = call_response.json()
            
            # Display results
            print("=" * 80)
            print("RESULTS")
            print("=" * 80)
            print()
            print(f"Total queries attempted: {result['total_attempted']}")
            print(f"Total queries validated: {result['total_validated']}")
            print(f"Execution time: {result['execution_time_ms']:.0f}ms")
            print()
            
            if result.get("warnings"):
                print("Warnings:")
                for warning in result["warnings"]:
                    print(f"  - {warning}")
                print()
            
            # Display each query
            for i, query in enumerate(result["queries"], 1):
                print(f"Query {i}:")
                print(f"  Status: {query['validation_status']}")
                print(f"  Alignment Score: {query['alignment_score']:.2f}")
                print(f"  Iterations: {query['iterations']}")
                print(f"  Generation Time: {query['generation_time_ms']:.0f}ms")
                print()
                
                if query["validation_status"] == "valid":
                    print(f"  Description: {query['description']}")
                    print()
                    print(f"  SQL:")
                    print("  " + "\n  ".join(query['sql'].split("\n")))
                    print()
                    
                    if query.get("estimated_cost_usd"):
                        print(f"  Estimated Cost: ${query['estimated_cost_usd']:.4f}")
                    
                    if query.get("estimated_bytes_processed"):
                        mb = query["estimated_bytes_processed"] / (1024 * 1024)
                        print(f"  Estimated Bytes: {mb:.2f} MB")
                    print()
                else:
                    error_msg = query["validation_details"].get("error_message", "Unknown error")
                    print(f"  Error: {error_msg}")
                    print()
                
                print("-" * 80)
                print()
            
            # Display summary
            if result.get("summary"):
                print("SUMMARY:")
                print(result["summary"])
            
        except httpx.HTTPError as e:
            print(f"✗ HTTP Error: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"Response: {e.response.text}")
        
        except Exception as e:
            print(f"✗ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())

