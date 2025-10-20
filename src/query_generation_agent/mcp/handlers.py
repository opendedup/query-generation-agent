"""
MCP Tool Handlers

Implements the business logic for MCP tools.
"""

import json
import logging
import time
from typing import Any, Dict, Sequence

from mcp.types import TextContent

from ..clients.bigquery_client import BigQueryClient
from ..clients.gemini_client import GeminiClient
from ..generation.query_ideator import QueryIdeator
from ..generation.query_refiner import QueryRefiner
from ..models.request_models import DatasetMetadata, GenerateQueriesRequest
from ..models.response_models import GenerateQueriesResponse
from ..validation.alignment_validator import AlignmentValidator
from ..validation.dryrun_validator import DryRunValidator
from ..validation.syntax_validator import SyntaxValidator
from .config import QueryGenerationConfig

logger = logging.getLogger(__name__)


class MCPHandlers:
    """
    Handlers for MCP tool requests.
    
    Orchestrates query generation, validation, and refinement.
    """
    
    def __init__(
        self,
        config: QueryGenerationConfig,
        bigquery_client: BigQueryClient,
        gemini_client: GeminiClient
    ):
        """
        Initialize MCP handlers.
        
        Args:
            config: Configuration
            bigquery_client: BigQuery client
            gemini_client: Gemini client
        """
        self.config = config
        self.bigquery_client = bigquery_client
        self.gemini_client = gemini_client
        
        # Initialize validators
        self.syntax_validator = SyntaxValidator()
        self.dryrun_validator = DryRunValidator(
            bigquery_client=bigquery_client,
            max_sample_rows=config.max_sample_rows
        )
        self.alignment_validator = AlignmentValidator(
            gemini_client=gemini_client,
            alignment_threshold=config.alignment_threshold
        )
        
        # Initialize generation components
        self.query_ideator = QueryIdeator(gemini_client=gemini_client)
        self.query_refiner = QueryRefiner(
            gemini_client=gemini_client,
            syntax_validator=self.syntax_validator,
            dryrun_validator=self.dryrun_validator,
            alignment_validator=self.alignment_validator,
            max_iterations=config.max_iterations
        )
        
        logger.info("MCP handlers initialized")
    
    async def handle_generate_queries(self, arguments: Dict[str, Any]) -> Sequence[TextContent]:
        """
        Handle generate_queries tool request.
        
        Args:
            arguments: Tool arguments
            
        Returns:
            Sequence of TextContent with results
        """
        start_time = time.time()
        
        try:
            # Parse and validate request
            request = self._parse_request(arguments)
            
            logger.info(f"Generating queries for insight: {request.insight[:100]}...")
            logger.info(f"Datasets: {len(request.datasets)}")
            logger.info(f"Max queries: {request.max_queries}")
            logger.info(f"Max iterations: {request.max_iterations}")
            
            # Step 1: Generate initial query candidates
            success, error_msg, candidates = self.query_ideator.generate_candidates(
                insight=request.insight,
                datasets=request.datasets,
                num_queries=request.max_queries
            )
            
            if not success or not candidates:
                error_response = {
                    "error": f"Failed to generate query candidates: {error_msg}",
                    "queries": [],
                    "total_attempted": 0,
                    "total_validated": 0,
                    "execution_time_ms": (time.time() - start_time) * 1000
                }
                return [TextContent(type="text", text=json.dumps(error_response, indent=2))]
            
            logger.info(f"Generated {len(candidates)} initial candidates")
            
            # Step 2: Refine and validate each candidate
            validated_queries = []
            warnings = []
            
            for i, candidate in enumerate(candidates):
                candidate_id = f"query_{i+1}"
                
                try:
                    logger.info(f"Processing candidate {i+1}/{len(candidates)}")
                    
                    query_result = self.query_refiner.refine_and_validate(
                        candidate_id=candidate_id,
                        initial_sql=candidate["sql"],
                        description=candidate["description"],
                        insight=request.insight,
                        datasets=request.datasets
                    )
                    
                    validated_queries.append(query_result)
                    
                    if query_result.is_valid():
                        logger.info(
                            f"Candidate {i+1} validated successfully "
                            f"(alignment: {query_result.alignment_score:.2f}, "
                            f"iterations: {query_result.iterations})"
                        )
                    else:
                        logger.warning(
                            f"Candidate {i+1} failed validation after {query_result.iterations} iterations"
                        )
                        warnings.append(
                            f"Query candidate {i+1} failed: {query_result.validation_details.error_message}"
                        )
                
                except Exception as e:
                    logger.error(f"Error processing candidate {i+1}: {e}", exc_info=True)
                    warnings.append(f"Failed to process candidate {i+1}: {str(e)}")
            
            # Build response
            response = GenerateQueriesResponse(
                queries=validated_queries,
                total_attempted=len(candidates),
                total_validated=len([q for q in validated_queries if q.is_valid()]),
                execution_time_ms=(time.time() - start_time) * 1000,
                insight=request.insight,
                dataset_count=len(request.datasets),
                summary=None,
                warnings=warnings
            )
            
            # Generate summary
            response.summary = response.get_summary_text()
            
            logger.info(
                f"Query generation complete. "
                f"Validated: {response.total_validated}/{response.total_attempted} "
                f"in {response.execution_time_ms:.0f}ms"
            )
            
            # Return as JSON
            response_json = response.model_dump(mode="json")
            return [TextContent(type="text", text=json.dumps(response_json, indent=2))]
            
        except Exception as e:
            logger.error(f"Error in handle_generate_queries: {e}", exc_info=True)
            error_response = {
                "error": f"Internal error: {str(e)}",
                "queries": [],
                "total_attempted": 0,
                "total_validated": 0,
                "execution_time_ms": (time.time() - start_time) * 1000
            }
            return [TextContent(type="text", text=json.dumps(error_response, indent=2))]
    
    def _parse_request(self, arguments: Dict[str, Any]) -> GenerateQueriesRequest:
        """
        Parse and validate request arguments.
        
        Args:
            arguments: Raw request arguments
            
        Returns:
            Validated GenerateQueriesRequest
            
        Raises:
            ValueError: If arguments are invalid
        """
        # Parse datasets
        datasets_data = arguments.get("datasets", [])
        datasets = []
        
        for ds_data in datasets_data:
            # Set defaults for optional fields
            ds_data.setdefault("row_count", None)
            ds_data.setdefault("size_bytes", None)
            ds_data.setdefault("column_count", None)
            ds_data.setdefault("has_pii", False)
            ds_data.setdefault("has_phi", False)
            ds_data.setdefault("environment", None)
            ds_data.setdefault("owner_email", None)
            ds_data.setdefault("tags", [])
            
            dataset = DatasetMetadata(**ds_data)
            datasets.append(dataset)
        
        # Build request
        request = GenerateQueriesRequest(
            insight=arguments["insight"],
            datasets=datasets,
            max_queries=arguments.get("max_queries", 3),
            max_iterations=arguments.get("max_iterations", self.config.max_query_iterations),
            require_alignment_check=arguments.get("require_alignment_check", True),
            allow_cross_dataset=arguments.get("allow_cross_dataset", True)
        )
        
        return request

