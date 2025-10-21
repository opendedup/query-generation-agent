<!-- 6f5acd58-fcc6-4dba-9f97-5296b6dbe19e d3780bad-8610-4299-9f3c-40bd82d13f10 -->
# Query Generation and Validation MCP Service

## Project Overview

Create a standalone MCP service (`query-generation-agent`) that receives dataset discovery results from the data-discovery-agent and generates validated BigQuery SQL queries aligned with specific data science insights.

## Architecture

### Input Format

JSON payload containing:

- **insight**: The data science question or requirement to satisfy
- **datasets**: Array of dataset metadata from data-discovery-agent
  - Each dataset includes: project_id, dataset_id, table_id, schema info, row_count, size_bytes, etc.
  - Full markdown documentation with field descriptions, sample data, relationships

### Output Format

JSON response containing:

- **queries**: Array of validated query objects
  - **sql**: The BigQuery SQL query
  - **description**: Natural language explanation of what the query does
  - **validation_status**: "valid" or "failed"
  - **validation_details**: Details from dry-run execution (rows returned, columns, execution stats)
  - **alignment_score**: How well the query addresses the insight (0-1)
  - **iterations**: Number of attempts to generate valid query

### Core Workflow

1. **Query Ideation**: Use Gemini to generate 3-5 candidate queries based on insight + dataset metadata
2. **Iterative Validation Loop** (max N attempts, e.g., 5):

   - Generate/refine query using LLM
   - Validate syntax by parsing SQL
   - Execute dry-run with LIMIT 10 in BigQuery
   - Check if results align with insight intent using LLM
   - If validation fails, feed error back to LLM for refinement
   - If validation succeeds, add to output array

3. **Return Results**: Array of successfully validated queries with descriptions

## Project Structure

```
query-generation-agent/
├── src/
│   └── query_generation_agent/
│       ├── mcp/
│       │   ├── __init__.py
│       │   ├── server.py              # MCP server setup (stdio/http)
│       │   ├── http_server.py         # FastAPI HTTP endpoints
│       │   ├── handlers.py            # MCP tool handlers
│       │   ├── tools.py               # Tool definitions
│       │   └── config.py              # Environment config
│       ├── clients/
│       │   ├── bigquery_client.py     # BigQuery dry-run execution
│       │   └── gemini_client.py       # Gemini API for generation
│       ├── models/
│       │   ├── request_models.py      # Input models (InsightRequest, DatasetMetadata)
│       │   ├── response_models.py     # Output models (QueryResponse, ValidationResult)
│       │   └── validation_models.py   # Validation state tracking
│       ├── generation/
│       │   ├── query_ideator.py       # Initial query generation
│       │   ├── query_refiner.py       # Iterative refinement with feedback
│       │   └── prompt_templates.py    # Gemini prompts
│       └── validation/
│           ├── syntax_validator.py    # SQL syntax checking
│           ├── dryrun_validator.py    # BigQuery dry-run execution
│           └── alignment_validator.py # LLM-based intent alignment
├── tests/
│   └── (unit and integration tests)
├── examples/
│   ├── stdio_client_example.py
│   └── http_client_example.py
├── Dockerfile
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```

## Key Components

### 1. MCP Tools

Define one primary tool:

**Tool: `generate_queries`**

- **Input**: 
  - `insight` (string): The data science question
  - `datasets` (array): Dataset metadata and markdown docs
  - `max_queries` (int, default: 3): Max queries to generate
  - `max_iterations` (int, default: 5): Max refinement attempts per query
- **Output**: Array of validated queries with descriptions

### 2. Query Ideation (`query_ideator.py`)

- Parse dataset metadata and markdown documentation
- Extract schema, relationships, sample data
- Use Gemini with structured prompt:
  - Input: insight + all dataset schemas/docs
  - Output: 3-5 candidate SQL queries (as JSON)
- Prompt engineering: emphasize efficiency, correctness, alignment with insight

### 3. Iterative Validation Loop (`query_refiner.py`)

For each candidate query:

- **Iteration 0**: Start with ideated query
- **Iterations 1-N**: 
  - Validate syntax (check for obvious SQL errors)
  - Execute dry-run in BigQuery with LIMIT 10
  - Check schema of returned columns
  - Use LLM to evaluate if sample results align with insight
  - If fail: Generate refined query using error feedback
  - If pass: Move to output array and continue to next candidate

### 4. Validation Components

**Syntax Validator** (`syntax_validator.py`)

- Basic SQL parsing/linting
- Check for common mistakes (missing FROM, invalid joins, etc.)

**Dry-Run Validator** (`dryrun_validator.py`)

- Execute query with BigQuery Jobs API
- Use `dryRun=true` first to check validity without execution
- Then execute with LIMIT 10 to get sample results
- Capture: returned schema, row count, execution stats, any errors

**Alignment Validator** (`alignment_validator.py`)

- Use Gemini to compare:
  - Original insight
  - Dataset schemas
  - Query SQL
  - Sample results (first 10 rows)
- Prompt: "Does this query answer the insight? Score 0-1."
- Return alignment score + reasoning

### 5. BigQuery Client (`bigquery_client.py`)

- Authenticate with service account or ADC
- Execute queries with dry-run flag
- Execute queries with LIMIT for validation
- Parse results into structured format
- Handle errors gracefully

### 6. Gemini Client (`gemini_client.py`)

- Initialize Gemini API (gemini-1.5-pro or gemini-2.0-flash)
- Structured generation for SQL (JSON mode)
- Prompt management and token optimization
- Rate limiting and retry logic

### 7. Configuration (`config.py`)

Environment variables:

```python
GCP_PROJECT_ID=your-project-id
GEMINI_API_KEY=your-gemini-key
BQ_EXECUTION_PROJECT=project-for-query-execution
BQ_LOCATION=US
MCP_SERVER_NAME=query-generation-agent
MCP_SERVER_VERSION=1.0.0
MCP_TRANSPORT=stdio  # or http
MCP_HOST=0.0.0.0
MCP_PORT=8081
MAX_QUERY_ITERATIONS=5
MAX_QUERIES_PER_INSIGHT=3
QUERY_TIMEOUT_SECONDS=30
GEMINI_MODEL=gemini-1.5-pro-latest
LOG_LEVEL=INFO
```

### 8. Models

**Request Models** (`request_models.py`):

```python
class DatasetMetadata(BaseModel):
    project_id: str
    dataset_id: str
    table_id: str
    asset_type: str
    row_count: Optional[int]
    size_bytes: Optional[int]
    schema_fields: List[Dict[str, Any]]  # Field names, types, descriptions
    full_markdown: str  # Complete documentation
    
class GenerateQueriesRequest(BaseModel):
    insight: str
    datasets: List[DatasetMetadata]
    max_queries: int = 3
    max_iterations: int = 5
```

**Response Models** (`response_models.py`):

```python
class ValidationResult(BaseModel):
    is_valid: bool
    error_message: Optional[str]
    execution_stats: Optional[Dict[str, Any]]
    sample_results: Optional[List[Dict[str, Any]]]  # First 10 rows
    
class QueryResult(BaseModel):
    sql: str
    description: str
    validation_status: Literal["valid", "failed"]
    validation_details: ValidationResult
    alignment_score: float  # 0-1
    iterations: int
    
class GenerateQueriesResponse(BaseModel):
    queries: List[QueryResult]
    total_attempted: int
    total_validated: int
    execution_time_ms: float
```

## Implementation Steps

### Phase 0: GitHub Repository Setup

1. Create new GitHub repository `query-generation-agent`

   - **Description**: "MCP service that generates and validates BigQuery SQL queries from data science insights"
   - **Visibility**: Public or Private (as needed)
   - **Initialize with**: Empty (no README, .gitignore, or license yet - we'll create these)

2. Add GitHub topics/tags for discoverability:

   - `mcp-server`
   - `model-context-protocol`
   - `bigquery`
   - `query-generation`
   - `sql-validation`
   - `gemini-ai`
   - `data-science`
   - `python`
   - `llm`
   - `gcp`

3. Configure repository settings:

   - Enable Issues
   - Enable Discussions (optional)
   - Set up branch protection for `main` (optional)

### Phase 1: Project Setup

1. Clone the newly created repository locally
2. Initialize Python project with Poetry (`poetry init`)
3. Set up pyproject.toml with dependencies:

   - python (>=3.10,<3.13)
   - python-dotenv, pydantic, mcp, fastapi, uvicorn
   - google-cloud-bigquery, google-generativeai
   - pytest, pytest-asyncio, pytest-mock, black, ruff

4. Create .env.example, .gitignore, README.md
5. Set up project directory structure
6. Initial commit and push to GitHub

### Phase 2: Core Models & Config

1. Implement configuration loading (config.py)
2. Define request/response models
3. Define validation state models
4. Create base client classes

### Phase 3: BigQuery Client

1. Implement BigQueryClient with authentication
2. Add dry-run validation method
3. Add query execution with LIMIT method
4. Add result parsing and error handling
5. Write unit tests with mocks

### Phase 4: Gemini Client

1. Implement GeminiClient with API key auth
2. Create prompt templates for query generation
3. Create prompt templates for alignment validation
4. Add structured output parsing (JSON mode)
5. Write unit tests with mock responses

### Phase 5: Validation Components

1. Implement SyntaxValidator (basic SQL parsing)
2. Implement DryRunValidator (BigQuery execution)
3. Implement AlignmentValidator (LLM-based)
4. Write comprehensive unit tests

### Phase 6: Query Generation Logic

1. Implement QueryIdeator (initial generation)
2. Implement QueryRefiner (iterative loop)
3. Integrate all validators
4. Add iteration tracking and logging
5. Write integration tests

### Phase 7: MCP Server

1. Implement MCP server setup (server.py)
2. Define `generate_queries` tool (tools.py)
3. Implement tool handler (handlers.py)
4. Add FastAPI HTTP server (http_server.py)
5. Add stdio and HTTP transport support

### Phase 8: Testing & Examples

1. Write end-to-end tests
2. Create example client scripts
3. Test with real data-discovery-agent output
4. Performance testing and optimization

### Phase 9: Docker & Deployment

1. Create Dockerfile
2. Create docker-compose.yml (optional)
3. Document deployment process
4. Test containerized deployment

### Phase 10: Documentation

1. Complete README with usage examples
2. Document API endpoints and tools
3. Add architecture diagrams
4. Create integration guide with data-discovery-agent

## Integration with Data Discovery Agent

**Workflow:**

1. User creates data science requirement doc (out of scope)
2. System generates insights/questions from requirements (out of scope)
3. For each insight:

   - Call data-discovery-agent's `query_data_assets` tool
   - Collect all relevant datasets and markdown docs
   - Format as JSON input for query-generation-agent
   - Call query-generation-agent's `generate_queries` tool
   - Receive validated queries

4. Aggregate all queries across insights
5. Present to user for review/execution

**Example Integration:**

```python
# Step 1: Discover datasets for insight
discovery_response = await discovery_agent.call_tool(
    "query_data_assets",
    {
        "query": "customer transaction data with payment methods",
        "include_full_content": True,
        "output_format": "json"
    }
)

# Step 2: Extract datasets
datasets = [
    {
        "project_id": result["metadata"]["project_id"],
        "dataset_id": result["metadata"]["dataset_id"],
        "table_id": result["metadata"]["table_id"],
        "asset_type": result["metadata"]["asset_type"],
        "row_count": result["metadata"]["row_count"],
        "size_bytes": result["metadata"]["size_bytes"],
        "schema_fields": [...],  # Extract from markdown
        "full_markdown": result["full_content"]
    }
    for result in discovery_response["results"]
]

# Step 3: Generate queries
query_response = await query_agent.call_tool(
    "generate_queries",
    {
        "insight": "What is the average transaction value by payment method for Q4 2024?",
        "datasets": datasets,
        "max_queries": 3
    }
)

# Step 4: Use queries
for query_result in query_response["queries"]:
    if query_result["validation_status"] == "valid":
        print(f"Query: {query_result['sql']}")
        print(f"Description: {query_result['description']}")
        print(f"Alignment: {query_result['alignment_score']:.2f}")
```

## Key Technical Decisions

1. **LLM Selection**: Use Gemini 1.5 Pro for better reasoning and SQL generation
2. **Iteration Limit**: Default 5 iterations to balance quality and performance
3. **Validation Strategy**: Syntax → Dry-run → Alignment (fail fast)
4. **Transport Modes**: Support both stdio (local) and HTTP (container)
5. **Error Handling**: Graceful degradation - return partial results if some queries fail
6. **Logging**: Comprehensive logging for debugging iteration loops
7. **Token Optimization**: Summarize large markdown docs to fit in context window

## Success Criteria

- Successfully generates 2+ valid queries for 80% of insights
- Average validation time < 30 seconds per insight
- Queries align with intent (score > 0.7) in 90% of cases
- Handles schema complexity (10+ tables, 100+ fields)
- Gracefully handles validation failures
- Full test coverage for critical paths

### To-dos

- [ ] Create repository structure, initialize Poetry project, configure dependencies in pyproject.toml
- [ ] Implement configuration, request/response models, validation state models
- [ ] Implement BigQuery client with dry-run and execution capabilities
- [ ] Implement Gemini client with prompt templates and structured output
- [ ] Implement syntax, dry-run, and alignment validators
- [ ] Implement query ideation and iterative refinement logic
- [ ] Implement MCP server, tools, handlers, and HTTP endpoints
- [ ] Write comprehensive unit, integration, and end-to-end tests
- [ ] Create Dockerfile, test containerized deployment
- [ ] Complete README, API docs, architecture diagrams, integration guide