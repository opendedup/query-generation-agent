# Query Generation Agent - Dockerfile
# Multi-stage build for optimized production image

FROM python:3.11-slim as builder

# Set working directory
WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Copy project files
COPY pyproject.toml ./
COPY README.md ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-root

# Copy source code
COPY src ./src

# Install the package
RUN poetry install --no-dev

# ============================================================================
# Production stage
# ============================================================================
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY --from=builder /build/src ./src

# Set Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port for HTTP mode
EXPOSE 8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# Default to HTTP server
# Can be overridden to run stdio server
CMD ["python", "-m", "query_generation_agent.mcp.http_server"]

