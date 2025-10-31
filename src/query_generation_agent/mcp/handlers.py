"""
MCP Tool Handlers

Implements the business logic for MCP tools.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Sequence

from mcp.types import TextContent

from ..clients.bigquery_client import BigQueryClient
from ..clients.gemini_client import GeminiClient
from ..generation.query_ideator import QueryIdeator
from ..generation.query_refiner import QueryRefiner
from ..generation.view_generator import ViewGenerator
from ..models.request_models import DatasetMetadata, GenerateQueriesRequest, GenerateViewsRequest
from ..models.response_models import GenerateQueriesResponse, QueryResult, TokenUsage
from ..parsers.prp_parser import parse_prp_section_9
from ..validation.alignment_validator import AlignmentValidator
from ..validation.dryrun_validator import DryRunValidator
from ..validation.view_validator import ViewValidator
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
            dryrun_validator=self.dryrun_validator,
            alignment_validator=self.alignment_validator,
            max_iterations=config.max_query_iterations,
            query_naming_strategy=config.query_naming_strategy
        )
        
        # Initialize view generation components
        self.view_generator = ViewGenerator(gemini_client=gemini_client)
        self.view_validator = ViewValidator(bigquery_client=bigquery_client)
        
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
            
            # Initialize token usage tracking
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_llm_calls = 0
            generation_tokens = 0
            refinement_tokens = 0
            alignment_tokens = 0
            
            # Step 1: Generate initial query candidates
            # Run in thread pool to avoid blocking the async event loop
            success, error_msg, candidates, gen_usage = await asyncio.to_thread(
                self.query_ideator.generate_candidates,
                insight=request.insight,
                datasets=request.datasets,
                num_queries=request.max_queries
            )
            
            # Track generation usage
            total_prompt_tokens += gen_usage.get("prompt_tokens", 0)
            total_completion_tokens += gen_usage.get("completion_tokens", 0)
            total_llm_calls += 1
            generation_tokens += gen_usage.get("total_tokens", 0)
            
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
                    
                    # Run in thread pool to avoid blocking the async event loop
                    query_result = await asyncio.to_thread(
                        self.query_refiner.refine_and_validate,
                        candidate_id=candidate_id,
                        initial_sql=candidate["sql"],
                        description=candidate["description"],
                        insight=request.insight,
                        datasets=request.datasets,
                        target_table_name=request.target_table_name,
                        query_index=i
                    )
                    
                    validated_queries.append(query_result)
                    
                    # Aggregate per-query token usage
                    if query_result.token_usage:
                        query_usage = query_result.token_usage
                        refinement_tokens += query_usage.get("refinement_tokens", 0)
                        alignment_tokens += query_usage.get("alignment_tokens", 0)
                        total_llm_calls += query_usage.get("llm_calls", 0)
                        # Also add to total prompt/completion tokens
                        # (approximation since we don't split refine vs align in per-query)
                        total_prompt_tokens += query_usage.get("refinement_tokens", 0) // 2
                        total_completion_tokens += query_usage.get("refinement_tokens", 0) // 2
                        total_prompt_tokens += query_usage.get("alignment_tokens", 0) // 2
                        total_completion_tokens += query_usage.get("alignment_tokens", 0) // 2
                    
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
            
            # Derive project_name from target_table_name
            project_name = None
            if request.target_table_name:
                project_name = request.target_table_name.replace("_", "-")
            
            # Build token usage summary
            total_tokens = total_prompt_tokens + total_completion_tokens
            token_usage = TokenUsage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_tokens,
                generation_tokens=generation_tokens,
                refinement_tokens=refinement_tokens,
                alignment_tokens=alignment_tokens,
                llm_calls=total_llm_calls
            )
            
            # Generate diagnostic report if all candidates failed
            failure_diagnostic_markdown = None
            total_validated_count = len([q for q in validated_queries if q.is_valid()])
            
            if total_validated_count == 0 and validated_queries:
                failure_diagnostic_markdown = self._generate_failure_diagnostic_markdown(
                    candidates=candidates,
                    validated_queries=validated_queries,
                    insight=request.insight
                )
                
                # Add to warnings for backward compatibility
                warnings.append("All candidates failed - see diagnostic report")
                
                logger.warning(f"All {len(candidates)} candidates failed. Diagnostic report generated.")
                logger.info(f"Failure diagnostic:\n{failure_diagnostic_markdown}")
            
            # Build response
            response = GenerateQueriesResponse(
                project_name=project_name,
                queries=validated_queries,
                total_attempted=len(candidates),
                total_validated=total_validated_count,
                execution_time_ms=(time.time() - start_time) * 1000,
                insight=request.insight,
                dataset_count=len(request.datasets),
                summary=None,
                warnings=warnings,
                token_usage=token_usage,
                failure_diagnostic=failure_diagnostic_markdown
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
        Parse and validate request arguments from data-discovery-agent.
        
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
            # Map data-discovery-agent output to our DatasetMetadata model
            dataset = DatasetMetadata(
                project_id=ds_data.get("project_id"),
                dataset_id=ds_data.get("dataset_id"),
                table_id=ds_data.get("table_id"),
                asset_type=ds_data.get("asset_type", ds_data.get("table_type", "TABLE")),
                description=ds_data.get("description"),
                row_count=ds_data.get("row_count"),
                size_bytes=ds_data.get("size_bytes"),
                column_count=ds_data.get("column_count"),
                created=ds_data.get("created"),
                last_modified=ds_data.get("last_modified"),
                insert_timestamp=ds_data.get("insert_timestamp"),
                schema=ds_data.get("schema", []),
                column_profiles=ds_data.get("column_profiles", []),
                lineage=ds_data.get("lineage", []),
                analytical_insights=ds_data.get("analytical_insights", []),
                key_metrics=ds_data.get("key_metrics", []),
                full_markdown=ds_data.get("full_markdown", ""),
                has_pii=ds_data.get("has_pii", False),
                has_phi=ds_data.get("has_phi", False),
                environment=ds_data.get("environment"),
                owner_email=ds_data.get("owner_email"),
                tags=ds_data.get("tags", [])
            )
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
    
    def _generate_failure_diagnostic_markdown(
        self,
        candidates: List[Dict[str, Any]],
        validated_queries: List[QueryResult],
        insight: str
    ) -> str:
        """
        Generate markdown-formatted failure diagnostic when all candidates fail.
        
        Analyzes failure patterns, table combinations, and provides recommendations.
        
        Args:
            candidates: Original candidate queries
            validated_queries: Validated query results
            insight: Original insight request
            
        Returns:
            Markdown-formatted diagnostic report
        """
        # Track failure patterns
        all_zero_rows = True
        all_syntax_errors = True
        alignment_scores = []
        common_errors: Dict[str, int] = {}
        all_tables_used = set()
        table_combination_failures: Dict[tuple, Dict[str, Any]] = {}
        
        candidates_analysis = []
        
        for i, query_result in enumerate(validated_queries):
            # Track all tables used
            all_tables_used.update(query_result.source_tables)
            
            # Track table combination failures
            table_combo = tuple(sorted(query_result.source_tables))
            if table_combo:
                if table_combo not in table_combination_failures:
                    table_combination_failures[table_combo] = {
                        "count": 0,
                        "zero_rows": 0,
                        "candidates": []
                    }
                table_combination_failures[table_combo]["count"] += 1
                table_combination_failures[table_combo]["candidates"].append(i + 1)
            
            # Analyze validation details
            val_details = query_result.validation_details
            
            # Check for zero rows
            rows_returned = 0
            if val_details.sample_results is not None:
                rows_returned = len(val_details.sample_results)
                if rows_returned > 0:
                    all_zero_rows = False
                else:
                    # Track zero row failures by table combination
                    if table_combo:
                        table_combination_failures[table_combo]["zero_rows"] += 1
            else:
                # Track zero row failures by table combination
                if table_combo:
                    table_combination_failures[table_combo]["zero_rows"] += 1
            
            # Check syntax/dryrun
            if val_details.syntax_valid and val_details.dryrun_valid:
                all_syntax_errors = False
            
            # Collect errors
            if val_details.error_message:
                error_type = val_details.error_type or "unknown"
                common_errors[error_type] = common_errors.get(error_type, 0) + 1
            
            # Collect alignment info
            if val_details.alignment_score is not None:
                alignment_scores.append(val_details.alignment_score)
            
            # Build candidate analysis
            candidates_analysis.append({
                "number": i + 1,
                "iterations": query_result.iterations,
                "rows_returned": rows_returned,
                "alignment_score": query_result.alignment_score,
                "tables": query_result.source_tables,
                "error": val_details.error_message,
                "reasoning": val_details.alignment_reasoning
            })
        
        # Build markdown output
        md_lines = [
            "# Query Generation Failure Analysis",
            "",
            f"**Insight**: {insight[:200]}{'...' if len(insight) > 200 else ''}",
            f"**Candidates Attempted**: {len(candidates)}",
            f"**Successful**: 0",
            f"**Execution Time**: {sum(q.generation_time_ms for q in validated_queries) / 1000:.1f}s",
            "",
            "---",
            "",
            "## Tables Attempted",
            ""
        ]
        
        # List all tables
        for table in sorted(all_tables_used):
            md_lines.append(f"- `{table}`")
        
        # Table combination analysis
        if table_combination_failures:
            md_lines.extend([
                "",
                "## Table Combination Analysis",
                "",
                "| Tables | Attempts | Zero Row Results | Candidates |",
                "|--------|----------|------------------|------------|"
            ])
            
            for table_combo, stats in sorted(table_combination_failures.items()):
                table_names = [t.split('.')[-1] for t in table_combo]
                tables_str = " + ".join(table_names)
                candidates_str = ", ".join([f"#{c}" for c in stats["candidates"]])
                md_lines.append(
                    f"| {tables_str} | {stats['count']} | {stats['zero_rows']} | {candidates_str} |"
                )
        
        # Root causes
        md_lines.extend([
            "",
            "## Root Causes",
            ""
        ])
        
        cause_num = 1
        if all_zero_rows:
            md_lines.append(
                f"{cause_num}. **All queries returned 0 rows** - this indicates a data availability issue"
            )
            cause_num += 1
        
        if all_syntax_errors:
            md_lines.append(f"{cause_num}. **Syntax or schema errors** prevented query execution")
            cause_num += 1
        
        if alignment_scores and not all_zero_rows:
            avg_score = sum(alignment_scores) / len(alignment_scores)
            md_lines.append(
                f"{cause_num}. **Queries executed but didn't align with insight** "
                f"(avg alignment score: {avg_score:.2f})"
            )
            cause_num += 1
        
        if len(all_tables_used) > 1 and all_zero_rows:
            md_lines.append(f"{cause_num}. **JOIN keys between tables may be incompatible**")
            cause_num += 1
        
        # Recommendations
        md_lines.extend([
            "",
            "## Recommendations",
            ""
        ])
        
        rec_num = 1
        if all_zero_rows:
            md_lines.extend([
                f"{rec_num}. Verify that the requested data exists in the source tables",
                f"{rec_num + 1}. Check if temporal filters (e.g., 'recent dates', 'future dates') are excluding all data",
                f"{rec_num + 2}. Confirm that JOIN keys between tables are compatible (matching IDs, timestamps, etc.)"
            ])
            rec_num += 3
            
            if len(all_tables_used) > 1:
                table_names = [t.split('.')[-1] for t in sorted(all_tables_used)]
                md_lines.append(
                    f"{rec_num}. Verify JOIN compatibility between: "
                    f"{', '.join([f'`{t}`' for t in table_names])}"
                )
                rec_num += 1
            
            # Identify problematic table combinations
            for table_combo, stats in table_combination_failures.items():
                if stats["zero_rows"] == stats["count"] and stats["count"] > 1:
                    table_names = [t.split('.')[-1] for t in table_combo]
                    md_lines.append(
                        f"{rec_num}. **Critical**: Table combination "
                        f"`{' + '.join(table_names)}` consistently failed "
                        f"(tried {stats['count']} times, 0 rows every time)"
                    )
                    rec_num += 1
        
        if all_syntax_errors:
            md_lines.append(f"{rec_num}. Review table schemas and column names for accuracy")
            rec_num += 1
        
        # Candidate details
        md_lines.extend([
            "",
            "---",
            "",
            "## Candidate Details",
            ""
        ])
        
        for analysis in candidates_analysis:
            table_names = [t.split('.')[-1] for t in analysis["tables"]]
            tables_str = ", ".join([f"`{t}`" for t in table_names]) if table_names else "none"
            
            md_lines.extend([
                f"### Candidate {analysis['number']}",
                f"- **Iterations**: {analysis['iterations']}",
                f"- **Rows Returned**: {analysis['rows_returned']}",
                f"- **Alignment Score**: {analysis['alignment_score']:.2f}",
                f"- **Tables**: {tables_str}"
            ])
            
            if analysis["error"]:
                # Truncate long error messages
                error_msg = analysis["error"][:300]
                if len(analysis["error"]) > 300:
                    error_msg += "..."
                md_lines.append(f"- **Error**: {error_msg}")
            
            if analysis["reasoning"]:
                # Truncate long reasoning
                reasoning = analysis["reasoning"][:300]
                if len(analysis["reasoning"]) > 300:
                    reasoning += "..."
                md_lines.append(f"- **Reasoning**: {reasoning}")
            
            md_lines.append("")
        
        return "\n".join(md_lines)
    
    async def handle_generate_queries_async(
        self,
        task_id: str,
        arguments: Dict[str, Any],
        task_manager: Any  # Avoid circular import
    ) -> None:
        """
        Handle generate_queries tool request asynchronously.
        
        Executes query generation in background and updates task status.
        Used for long-running operations to prevent client timeouts.
        
        Args:
            task_id: Task identifier
            arguments: Tool arguments
            task_manager: TaskManager instance for updating status
        """
        from .task_manager import TaskStatus
        
        try:
            logger.info(f"Starting async query generation for task {task_id}")
            task_manager.update_task_status(task_id, TaskStatus.RUNNING)
            
            # Call existing handle_generate_queries logic
            result = await self.handle_generate_queries(arguments)
            
            # Update task with result
            task_manager.update_task_status(
                task_id,
                TaskStatus.COMPLETED,
                result=result
            )
            logger.info(f"Async query generation completed for task {task_id}")
            
        except Exception as e:
            logger.error(f"Async query generation failed for task {task_id}: {e}", exc_info=True)
            task_manager.update_task_status(
                task_id,
                TaskStatus.FAILED,
                error=str(e)
            )
    
    async def handle_generate_views(self, arguments: Dict[str, Any]) -> Sequence[TextContent]:
        """
        Handle generate_views tool request.
        
        Generates CREATE VIEW DDL statements from PRP Section 9 data requirements.
        
        Args:
            arguments: Tool arguments with prp_markdown and source_datasets
            
        Returns:
            Sequence of TextContent with results in GenerateQueriesResponse format
        """
        start_time = time.time()
        
        try:
            # Parse and validate request
            request = self._parse_views_request(arguments)
            
            logger.info("Generating VIEW DDL from PRP Section 9")
            logger.info(f"Source datasets: {len(request.source_datasets)}")
            logger.info(f"Target location: {request.get_target_location() or 'not specified'}")
            
            # Parse PRP to extract target views
            target_views = parse_prp_section_9(request.prp_markdown)
            
            if not target_views:
                logger.error("No target views found in PRP markdown")
                error_response = GenerateQueriesResponse(
                    queries=[],
                    total_attempted=0,
                    total_validated=0,
                    execution_time_ms=0,
                    insight="Generate views from PRP Section 9",
                    dataset_count=len(request.source_datasets),
                    warnings=["No target views found in PRP Section 9"]
                )
                return [TextContent(type="text", text=json.dumps(error_response.model_dump(), indent=2))]
            
            logger.info(f"Found {len(target_views)} target views in PRP")
            
            # Generate DDL for each view
            view_results = []
            for target_view in target_views:
                view_start_time = time.time()
                
                logger.info(f"Generating DDL for view: {target_view.view_name}")
                
                # Generate DDL (run in thread pool to avoid blocking)
                success, error_msg, ddl = await asyncio.to_thread(
                    self.view_generator.generate_view_ddl,
                    target_view=target_view,
                    source_datasets=request.source_datasets,
                    target_location=request.get_target_location()
                )
                
                if not success:
                    logger.error(f"Failed to generate DDL for {target_view.view_name}: {error_msg}")
                    # Create failed result
                    from ..models.response_models import ValidationResult
                    validation = ValidationResult(
                        is_valid=False,
                        error_message=error_msg,
                        error_type="generation",
                        syntax_valid=False,
                        dryrun_valid=False,
                        execution_valid=False,
                        alignment_valid=False
                    )
                    
                    result = QueryResult(
                        sql="-- Failed to generate DDL",
                        description=target_view.description,
                        source_tables=[],
                        validation_status="failed",
                        validation_details=validation,
                        alignment_score=0.0,
                        iterations=0,
                        generation_time_ms=(time.time() - view_start_time) * 1000
                    )
                    view_results.append(result)
                    continue
                
                # Validate schema matches target (run in thread pool)
                validation = await asyncio.to_thread(
                    self.view_validator.validate_view_ddl,
                    view_ddl=ddl,
                    target_schema=target_view.columns
                )
                
                # Extract source tables from DDL
                source_tables = self._extract_source_tables_from_ddl(ddl)
                
                # Build QueryResult (same format as generate_queries)
                result = QueryResult(
                    sql=ddl,
                    description=target_view.description,
                    source_tables=source_tables,
                    validation_status="valid" if validation.is_valid else "failed",
                    validation_details=validation,
                    alignment_score=validation.alignment_score or (1.0 if validation.is_valid else 0.0),
                    iterations=0,  # No iterative refinement for views yet
                    generation_time_ms=(time.time() - view_start_time) * 1000
                )
                
                view_results.append(result)
                logger.info(
                    f"View {target_view.view_name}: "
                    f"{result.validation_status} "
                    f"({result.generation_time_ms:.0f}ms)"
                )
            
            # Build response in GenerateQueriesResponse format
            total_time = (time.time() - start_time) * 1000
            response = GenerateQueriesResponse(
                queries=view_results,
                total_attempted=len(target_views),
                total_validated=len([r for r in view_results if r.validation_status == "valid"]),
                execution_time_ms=total_time,
                insight="Generate views from PRP Section 9",
                dataset_count=len(request.source_datasets)
            )
            
            logger.info(
                f"Generated {len(view_results)} views "
                f"({response.total_validated} valid) in {total_time:.0f}ms"
            )
            
            return [TextContent(type="text", text=json.dumps(response.model_dump(), indent=2))]
            
        except Exception as e:
            logger.error(f"Error generating views: {e}", exc_info=True)
            error_response = GenerateQueriesResponse(
                queries=[],
                total_attempted=0,
                total_validated=0,
                execution_time_ms=(time.time() - start_time) * 1000,
                insight="Generate views from PRP Section 9",
                dataset_count=0,
                warnings=[f"Error: {str(e)}"]
            )
            return [TextContent(type="text", text=json.dumps(error_response.model_dump(), indent=2))]
    
    def _parse_views_request(self, arguments: Dict[str, Any]) -> GenerateViewsRequest:
        """
        Parse and validate generate_views request arguments.
        
        Args:
            arguments: Raw request arguments
            
        Returns:
            Validated GenerateViewsRequest
            
        Raises:
            ValueError: If arguments are invalid
        """
        # Parse datasets
        datasets_data = arguments.get("source_datasets", [])
        datasets = []
        
        for ds_data in datasets_data:
            # Map data-discovery-agent output to our DatasetMetadata model
            dataset = DatasetMetadata(
                project_id=ds_data.get("project_id"),
                dataset_id=ds_data.get("dataset_id"),
                table_id=ds_data.get("table_id"),
                asset_type=ds_data.get("asset_type", ds_data.get("table_type", "TABLE")),
                description=ds_data.get("description"),
                row_count=ds_data.get("row_count"),
                size_bytes=ds_data.get("size_bytes"),
                column_count=ds_data.get("column_count"),
                created=ds_data.get("created"),
                last_modified=ds_data.get("last_modified"),
                insert_timestamp=ds_data.get("insert_timestamp"),
                schema=ds_data.get("schema", []),
                column_profiles=ds_data.get("column_profiles", []),
                lineage=ds_data.get("lineage", []),
                analytical_insights=ds_data.get("analytical_insights", []),
                key_metrics=ds_data.get("key_metrics", []),
                full_markdown=ds_data.get("full_markdown", ""),
                has_pii=ds_data.get("has_pii", False),
                has_phi=ds_data.get("has_phi", False),
                environment=ds_data.get("environment"),
                owner_email=ds_data.get("owner_email"),
                tags=ds_data.get("tags", [])
            )
            datasets.append(dataset)
        
        # Build request
        request = GenerateViewsRequest(
            prp_markdown=arguments["prp_markdown"],
            source_datasets=datasets,
            target_project=arguments.get("target_project"),
            target_dataset=arguments.get("target_dataset")
        )
        
        return request
    
    def _extract_source_tables_from_ddl(self, ddl: str) -> list[str]:
        """
        Extract source table references from DDL.
        
        Args:
            ddl: SQL DDL statement
            
        Returns:
            List of fully qualified table names
        """
        import re
        
        # Pattern to match fully qualified table names: `project.dataset.table` or project.dataset.table
        pattern = r'`?([a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+)`?'
        
        matches = re.findall(pattern, ddl)
        
        # Deduplicate and return
        return list(set(matches))

