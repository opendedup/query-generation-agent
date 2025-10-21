# Query Generation Agent

MCP service that generates and validates BigQuery SQL queries from data science insights.

## Overview

The Query Generation Agent is a Model Context Protocol (MCP) service that takes dataset discovery results and data science insights, then generates and validates BigQuery SQL queries that accurately answer the insights. It uses an iterative refinement approach with Gemini 2.5 Pro to ensure query quality and alignment with the intended analysis.

## Key Features

- **Intelligent Query Generation**: Uses Gemini 2.5 Pro to generate multiple candidate SQL queries
- **Iterative Validation**: Refines queries through syntax validation, dry-run execution, and alignment checking
- **Quality Over Speed**: Prioritizes query accuracy and insight alignment over execution time
- **BigQuery Integration**: Direct integration with BigQuery for query validation and execution
- **MCP Protocol**: Supports both stdio and HTTP transports for flexible deployment
- **Comprehensive Validation**: Multi-stage validation including syntax, execution, and semantic alignment

## Architecture

### Input Format

JSON payload containing:
- **insight**: The data science question or requirement to satisfy
- **datasets**: Array of dataset metadata from data-discovery-agent
  - project_id, dataset_id, table_id
  - Schema information with field descriptions
  - Full markdown documentation

### Output Format

JSON response containing:
- **queries**: Array of validated query objects with:
  - SQL query text
  - Natural language description
  - Validation status and details
  - Alignment score (0-1)
  - Number of refinement iterations

### Validation Pipeline

1. **Query Ideation**: Generate 3-5 candidate queries using Gemini
2. **Syntax Validation**: Parse and check SQL syntax
3. **Dry-Run Execution**: Execute with BigQuery dry-run API
4. **Sample Execution**: Run with LIMIT 10 to get sample results
5. **Alignment Validation**: LLM evaluates if results match insight intent
6. **Iterative Refinement**: If validation fails, refine and retry (up to 10 iterations)

## Installation

### Prerequisites

- Python 3.10+
- Poetry for dependency management
- Google Cloud Project with BigQuery enabled
- Gemini API key

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/query-generation-agent.git
cd query-generation-agent

# Install dependencies with Poetry
poetry install

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your credentials

# Activate the Poetry environment
poetry shell
```

## Configuration

See `.env.example` for all available configuration options. Key variables:

- `GCP_PROJECT_ID`: Your Google Cloud Project ID
- `GEMINI_API_KEY`: Gemini API key for query generation
- `MAX_QUERY_ITERATIONS`: Maximum refinement attempts (default: 10)
- `ALIGNMENT_THRESHOLD`: Minimum score for query acceptance (default: 0.85)
- `GEMINI_MODEL`: Model to use (default: gemini-2.5-pro-latest)

## Usage

### As MCP Server (stdio)

```bash
# Start the MCP server in stdio mode (default)
# Ensure MCP_TRANSPORT=stdio in .env or leave unset
poetry run python -m query_generation_agent.mcp
```

### As HTTP Service

```bash
# Set MCP_TRANSPORT=http in .env, then start the server
poetry run python -m query_generation_agent.mcp

# Or with uvicorn directly
poetry run uvicorn query_generation_agent.mcp.http_server:app --host 0.0.0.0 --port 8081
```

### Python Client Example

```python
from query_generation_agent.models.request_models import (
    GenerateQueriesRequest,
    DatasetMetadata
)

# Prepare request
request = GenerateQueriesRequest(
    insight="What is the average transaction value by payment method for Q4 2024?",
    datasets=[
        DatasetMetadata(
            project_id="my-project",
            dataset_id="sales",
            table_id="transactions",
            asset_type="table",
            row_count=1000000,
            schema_fields=[...],
            full_markdown="# Transactions Table\n..."
        )
    ],
    max_queries=3,
    max_iterations=10
)

# Call the service (via MCP)
response = await mcp_client.call_tool("generate_queries", request.dict())

# Use the validated queries
for query in response["queries"]:
    if query["validation_status"] == "valid":
        print(f"SQL: {query['sql']}")
        print(f"Description: {query['description']}")
        print(f"Alignment: {query['alignment_score']}")
```

## Integration with Data Discovery Agent

This service is designed to work downstream from the data-discovery-agent:

1. Use data-discovery-agent to find relevant datasets for an insight
2. Extract dataset metadata and markdown documentation
3. Pass to query-generation-agent to generate validated SQL queries
4. Execute queries in BigQuery or present to user

See `examples/` directory for complete integration examples.

## Development

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=query_generation_agent --cov-report=html

# Run specific test suite
poetry run pytest tests/unit/
poetry run pytest tests/integration/
poetry run pytest tests/e2e/
```

### Code Quality

```bash
# Format code with Black
poetry run black src/ tests/

# Lint with Ruff
poetry run ruff check src/ tests/

# Type checking (if using mypy)
poetry run mypy src/
```

## Docker Deployment

```bash
# Build the Docker image
docker build -t query-generation-agent:latest .

# Run the container
docker run -d \
  --name query-gen-mcp \
  -p 8081:8081 \
  -e MCP_TRANSPORT=http \
  -e GCP_PROJECT_ID=your-project \
  -e GEMINI_API_KEY=your-key \
  query-generation-agent:latest
```

## Project Structure

```
query-generation-agent/
├── src/query_generation_agent/
│   ├── mcp/              # MCP server and handlers
│   ├── clients/          # BigQuery and Gemini clients
│   ├── models/           # Pydantic data models
│   ├── generation/       # Query generation logic
│   └── validation/       # Validation components
├── tests/                # Test suites
├── examples/             # Usage examples
└── docs/                 # Additional documentation
```

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

