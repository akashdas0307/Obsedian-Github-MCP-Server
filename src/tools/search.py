"""Semantic search MCP tool — the 15th tool."""

import json

import structlog

from mcp.server.fastmcp import FastMCP

log = structlog.get_logger()


def _get_search_engine():
    from src.main import search_engine
    return search_engine


def register_search_tool(mcp: FastMCP):
    """Register the semantic_search tool on the given FastMCP server."""

    @mcp.tool()
    async def semantic_search(query: str, top_k: int = 10) -> str:
        """Semantically search the repository content using AI embeddings.

        Uses nomic-embed-text-v1.5 for embedding and cross-encoder reranking
        to find the most relevant file sections for a given query.

        Args:
            query: Natural language search query.
            top_k: Number of results to return (default 10, max 20).

        Returns:
            JSON array of results, each with:
              - file_name: Relative file path
              - position_range: Character range in format "0-1000"
              - score: Relevance score 0.0-1.0
              - preview: Short text preview of the matching section
        """
        engine = _get_search_engine()
        if engine is None:
            return json.dumps({"error": "Semantic search engine not initialised"})

        top_k = max(1, min(top_k, 20))

        try:
            results = await engine.search(query, top_k=top_k)
            return json.dumps(results, indent=2)
        except Exception as e:
            log.exception("semantic_search_tool_error", query=query)
            return json.dumps({"error": f"Search failed: {str(e)}"})
