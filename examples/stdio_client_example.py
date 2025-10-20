"""
Example: Using Query Generation Agent via stdio transport

This example shows how to call the MCP server running in stdio mode.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path


async def main() -> None:
    """Run example stdio client."""
    
    # Example request data
    request = {
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
                        "description": "Payment method used (credit_card, paypal, bank_transfer, etc.)"
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
                        "description": "Transaction status (completed, pending, failed)"
                    },
                    {
                        "name": "currency",
                        "type": "STRING",
                        "description": "Currency code (USD, EUR, etc.)"
                    },
                    {
                        "name": "country",
                        "type": "STRING",
                        "description": "Customer country code"
                    }
                ],
                "full_markdown": """# Transactions Table

## Overview
Contains all customer transactions including payment methods and amounts.

## Schema
- transaction_id: Unique identifier for each transaction
- customer_id: Links to customers table
- payment_method: Credit card, PayPal, bank transfer, etc.
- amount: Transaction value in USD
- timestamp: When transaction occurred
- status: Current status (completed, pending, failed)
- currency: Currency used
- country: Customer's country

## Sample Data
Example rows showing typical payment methods and transaction values.

## Usage Notes
- Filter by status='completed' for successful transactions
- Use timestamp for date range queries
- payment_method is normalized to lowercase with underscores
""",
                "has_pii": True,
                "has_phi": False,
                "environment": "prod",
                "owner_email": "data-team@example.com",
                "tags": ["sales", "transactions", "payments"]
            }
        ],
        "max_queries": 3,
        "max_iterations": 10,
        "require_alignment_check": True,
        "allow_cross_dataset": True
    }
    
    print("=" * 80)
    print("Query Generation Agent - stdio Client Example")
    print("=" * 80)
    print()
    print(f"Insight: {request['insight']}")
    print(f"Datasets: {len(request['datasets'])}")
    print(f"Max queries: {request['max_queries']}")
    print()
    print("Generating queries...")
    print()
    
    # Note: This is a simplified example
    # In practice, you'd use the MCP SDK client libraries
    
    # For this example, we'll show how to invoke the server directly
    # In production, use proper MCP client libraries
    
    print("To run the MCP server in stdio mode:")
    print("  poetry run python -m query_generation_agent.mcp")
    print()
    print("To call the server from your MCP client:")
    print("  1. Initialize MCP client connection to the server process")
    print("  2. Call tool 'generate_queries' with the following arguments:")
    print()
    print(json.dumps(request, indent=2))
    print()
    print("Expected response structure:")
    print("""
{
  "queries": [
    {
      "sql": "SELECT payment_method, AVG(amount) as avg_amount ...",
      "description": "Calculate average transaction by payment method",
      "validation_status": "valid",
      "validation_details": {...},
      "alignment_score": 0.92,
      "iterations": 2,
      "generation_time_ms": 3500.0
    },
    ...
  ],
  "total_attempted": 3,
  "total_validated": 2,
  "execution_time_ms": 8500.0,
  "insight": "...",
  "dataset_count": 1
}
""")


if __name__ == "__main__":
    asyncio.run(main())

