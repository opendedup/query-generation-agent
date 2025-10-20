# Query Generation Agent - Quick Start Guide

Get up and running with the Query Generation Agent in minutes!

## Prerequisites

- Python 3.10+
- Poetry
- Google Cloud Project with BigQuery enabled
- Gemini API key

## Option 1: Local Development (stdio mode)

### 1. Clone and Install

```bash
git clone https://github.com/your-org/query-generation-agent.git
cd query-generation-agent
poetry install
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials:
# - GCP_PROJECT_ID
# - GEMINI_API_KEY
```

### 3. Run the Server

```bash
poetry run python -m query_generation_agent.mcp
```

The server is now listening on stdio for MCP protocol messages.

## Option 2: HTTP Server (for testing)

### 1-2. Same as above

### 3. Set HTTP mode

Edit `.env`:
```bash
MCP_TRANSPORT=http
MCP_PORT=8081
```

### 4. Run HTTP Server

```bash
poetry run python -m query_generation_agent.mcp.http_server
```

### 5. Test with curl

```bash
# Health check
curl http://localhost:8081/health

# List tools
curl http://localhost:8081/mcp/tools

# Call generate_queries (see examples/ for full payload)
curl -X POST http://localhost:8081/mcp/call-tool \
  -H "Content-Type: application/json" \
  -d @examples/sample_request.json
```

## Option 3: Docker Deployment

### 1. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 2. Build and Run

```bash
docker-compose up -d
```

### 3. Check Status

```bash
# View logs
docker-compose logs -f

# Health check
curl http://localhost:8081/health
```

### 4. Stop

```bash
docker-compose down
```

## Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=query_generation_agent --cov-report=html

# Run specific test file
poetry run pytest tests/unit/test_config.py
```

## Example Usage

See `examples/http_client_example.py` for a complete working example:

```bash
# Make sure server is running in HTTP mode
poetry run python examples/http_client_example.py
```

## Integration with Data Discovery Agent

1. **Discover datasets** using data-discovery-agent:
   ```python
   datasets = await discovery_agent.query_data_assets(
       query="transaction tables",
       include_full_content=True
   )
   ```

2. **Generate queries** using query-generation-agent:
   ```python
   queries = await query_gen_agent.generate_queries(
       insight="What is the average transaction value by payment method?",
       datasets=datasets
   )
   ```

3. **Execute queries** in BigQuery:
   ```python
   for query in queries:
       if query["validation_status"] == "valid":
           results = bigquery_client.query(query["sql"]).result()
   ```

## Troubleshooting

### "GCP_PROJECT_ID environment variable is required"
Make sure you've copied `.env.example` to `.env` and filled in your GCP project ID.

### "GEMINI_API_KEY environment variable is required"
Get a Gemini API key from Google AI Studio and add it to your `.env` file.

### BigQuery authentication errors
Ensure you're authenticated with GCP:
```bash
gcloud auth application-default login
```

### Query validation takes too long
Adjust these settings in `.env`:
- `MAX_QUERY_ITERATIONS` - Reduce for faster results (but lower quality)
- `QUERY_TIMEOUT_SECONDS` - Increase if queries are timing out

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore [examples/](examples/) for more usage patterns
- Check [tests/](tests/) for implementation examples
- Review [src/](src/) for the complete codebase

## Getting Help

- Open an issue on GitHub
- Check existing issues for solutions
- Review logs for detailed error messages

## Common Workflows

### Workflow 1: Simple Query Generation
```python
# 1. Prepare datasets (from data-discovery-agent)
# 2. Call generate_queries with insight
# 3. Get back validated SQL queries
# 4. Execute in BigQuery
```

### Workflow 2: Iterative Refinement
```python
# 1. Generate initial queries
# 2. Review alignment scores
# 3. Adjust alignment_threshold in config
# 4. Re-generate for better quality
```

### Workflow 3: Batch Processing
```python
# 1. Load multiple insights from requirements doc
# 2. For each insight, discover datasets
# 3. Generate queries for each insight
# 4. Collect all validated queries
# 5. Execute and analyze results
```

Happy querying! 🚀

