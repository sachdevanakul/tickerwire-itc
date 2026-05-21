"""
src/mcp/server.py
MCP server exposing the get_financial_kpi tool.
Run: python src/mcp/server.py
"""
from __future__ import annotations
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
from src.mcp.tools import get_financial_kpi
from src.utils.logging import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

app = Server("tickerwire-itc-mcp")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_financial_kpi",
            description=(
                "Look up an exact financial KPI for ITC Limited from annual reports FY22–FY25. "
                "Use this for precise numbers: revenue, PAT, EBITDA, segment revenues, ROCE, EPS. "
                "Always prefer this tool over free-text retrieval for specific numbers."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "description": "The financial metric to look up, e.g. 'revenue', 'PAT', 'EBITDA', 'cigarette segment revenue', 'ROCE', 'EPS'",
                    },
                    "fiscal_year": {
                        "type": "string",
                        "description": "Fiscal year: FY22, FY23, FY24, or FY25",
                        "enum": ["FY22", "FY23", "FY24", "FY25"],
                    },
                },
                "required": ["metric", "fiscal_year"],
            },
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name != "get_financial_kpi":
        raise ValueError(f"Unknown tool: {name}")

    result = get_financial_kpi(
        metric=arguments["metric"],
        fiscal_year=arguments["fiscal_year"],
    )
    logger.info("mcp_tool_called", tool=name, args=arguments, result=result)

    if result["found"]:
        text = f"ITC {result['metric']} for {result['fiscal_year']}: {result['value']}"
    else:
        text = f"Not found: {result.get('error', 'Unknown error')}"

    return [types.TextContent(type="text", text=text)]


async def main():
    logger.info("mcp_server_starting")
    async with stdio_server() as streams:
        await app.run(streams[0], streams[1], app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
