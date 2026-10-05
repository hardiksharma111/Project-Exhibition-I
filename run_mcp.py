"""
Model Context Protocol (MCP) Standard Server Launcher
Runs the MCP server over standard input/output (stdio) for external MCP clients.
"""

from core.mcp.server import mcp

if __name__ == "__main__":
    print("[MCP Server] Starting Intelligent Multi-Model AI MCP Server via stdio...")
    mcp.run(transport="stdio")
