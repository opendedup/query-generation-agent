# Query Generation Agent - Examples

This directory contains example scripts demonstrating how to use the Query Generation Agent MCP service.

## Available Examples

### 1. `http_client_example.py` - Basic HTTP Client
Simple example showing how to call the query generation service directly via HTTP.

**Use case**: You already have dataset metadata and want to generate queries.

```bash
# Make sure the query-generation-agent is running in HTTP mode
cd /home/user/git/query-generation-agent
poetry run python examples/http_client_example.py
```

**What it does**:
- Connects to query-generation-agent on `http://localhost:8081`
- Sends a predefined insight and dataset metadata
- Generates SQL queries
- Displays results

### 2. `integrated_workflow_example.py` - Complete Workflow ⭐
**Recommended for most users!**

Complete end-to-end workflow demonstrating the integration between data-discovery-agent and query-generation-agent.

**Use case**: You want to discover datasets first, then generate queries from them.

```bash
# Prerequisites: Both agents must be running

# Terminal 1: Start data-discovery-agent
cd /home/user/git/data-discovery-agent
poetry run python -m data_discovery_agent.mcp

# Terminal 2: Start query-generation-agent
cd /home/user/git/query-generation-agent
poetry run python -m query_generation_agent.mcp

# Terminal 3: Run the integrated example

# Use defaults
cd /home/user/git/query-generation-agent
poetry run python examples/integrated_workflow_example.py

# Or specify your own query and insight
poetry run python examples/integrated_workflow_example.py \
  --query "customer analytics tables" \
  --insight "What is the average order value by product category?"

# With filters
poetry run python examples/integrated_workflow_example.py \
  --query "revenue tables" \
  --insight "Calculate monthly revenue trend" \
  --page-size 5 \
  --environment prod \
  --no-pii

# See all options
poetry run python examples/integrated_workflow_example.py --help
```

**What it does**:
1. **Health Check**: Verifies both services are running
2. **Discovery**: Searches for datasets using natural language query
   - Example: "customer transaction tables"
   - Gets datasets in BigQuery schema format (DiscoveredAssetDict)
3. **Query Generation**: Generates SQL queries from insight
   - Example: "What are the top 10 customers by total transaction amount?"
   - Validates queries through dry-run and syntax checking
   - Refines queries iteratively for alignment
4. **Results**: Displays validated SQL queries with:
   - Alignment scores
   - Cost estimates
   - Validation status
   - Ready-to-execute SQL

### 3. `stdio_client_example.py` - Stdio Transport
Example for using the MCP server via stdio transport (for Cursor integration).

**Use case**: Testing stdio transport or debugging Cursor integration.

```bash
cd /home/user/git/query-generation-agent
poetry run python examples/stdio_client_example.py
```

## Command-Line Options

The integrated workflow example supports extensive command-line options:

```bash
# Discovery parameters
-q, --query              Natural language discovery query
-i, --insight           Data science insight or question
--page-size             Number of datasets to retrieve (default: 3)

# Filters
--pii                   Only include tables with PII data
--no-pii               Exclude tables with PII data
--phi                   Only include tables with PHI data
--no-phi               Exclude tables with PHI data
--environment          Filter by environment (prod, staging, dev)

# Query generation
--max-queries          Maximum queries to generate (default: 3)
--max-iterations       Maximum refinement iterations (default: 10)

# Service URLs (advanced)
--discovery-url        Data discovery agent URL
--query-gen-url        Query generation agent URL
```

### Example Commands

```bash
# Basic usage
poetry run python examples/integrated_workflow_example.py \
  -q "transaction tables" \
  -i "What are the top 10 customers by revenue?"

# Production data only, no PII
poetry run python examples/integrated_workflow_example.py \
  --query "analytics tables" \
  --insight "Calculate monthly active users" \
  --environment prod \
  --no-pii

# Get more alternatives with longer refinement
poetry run python examples/integrated_workflow_example.py \
  -q "revenue data" \
  -i "Show quarterly revenue trend by product" \
  --max-queries 5 \
  --max-iterations 15

# Get more datasets to work with
poetry run python examples/integrated_workflow_example.py \
  -q "customer behavior tables" \
  -i "Analyze customer retention by cohort" \
  --page-size 10
```

## Customizing via Code

You can also use the `IntegratedMCPClient` class programmatically:

```python
from integrated_workflow_example import IntegratedMCPClient, print_query_results
import asyncio

async def my_custom_workflow():
    client = IntegratedMCPClient()
    
    # Step 1: Discover datasets
    datasets = await client.discover_datasets(
        query="YOUR SEARCH QUERY HERE",
        page_size=5,
        has_pii=False,
        environment="prod"
    )
    
    # Step 2: Generate queries
    results = await client.generate_queries(
        insight="YOUR DATA SCIENCE QUESTION HERE",
        datasets=datasets,
        max_queries=3,
        max_iterations=10
    )
    
    # Step 3: Display results
    print_query_results(results)

asyncio.run(my_custom_workflow())
```


## Example Insights (Customize for Your Data)

### Analytics Insights
- "What are the top 10 customers by total revenue in the last quarter?"
- "Show monthly recurring revenue (MRR) trend over the past 12 months"
- "Calculate customer churn rate by cohort month"
- "What is the average order value by product category?"

### Operational Insights
- "How many failed transactions occurred in the last 7 days by payment method?"
- "Show daily active users trend for the past 30 days"
- "What are the most common error codes and their frequencies?"

### Segmentation Insights
- "Group customers by total spend into segments: high, medium, low"
- "What percentage of revenue comes from the top 20% of customers?"
- "Compare conversion rates between mobile and desktop users"

## Troubleshooting

### "Connection refused" or "Service unavailable"
- Make sure both agents are running in HTTP mode
- Check `.env` files have `MCP_TRANSPORT=http`
- Verify ports: data-discovery on 8080, query-generation on 8081

### "No datasets found"
- Check that data-discovery-agent has indexed your BigQuery tables
- Try broader search queries
- Remove restrictive filters (has_pii, environment, etc.)

### "No valid queries generated"
- Make sure your insight is specific and answerable with the discovered datasets
- Check that datasets contain relevant fields for your insight
- Review error messages in the output for guidance
- Try increasing `max_iterations` parameter

### "GEMINI_API_KEY environment variable is required"
- Get an API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
- Add to `.env`: `GEMINI_API_KEY=your-key-here`

### "BigQuery authentication errors"
```bash
gcloud auth application-default login
```

## Next Steps

1. **Review generated SQL**: Always review queries before executing in production
2. **Execute in BigQuery**: Copy validated queries to BigQuery console or use API
3. **Iterate**: Refine your insights based on results
4. **Integrate**: Use these patterns in your own applications

## Additional Resources

- [Main README](../README.md) - Full documentation
- [QUICKSTART](../QUICKSTART.md) - Quick setup guide
- [API Documentation](../docs/) - Detailed API reference
- [Tests](../tests/) - More usage examples

## Support

- Open issues on GitHub
- Review logs for detailed error messages
- Check existing issues for solutions

