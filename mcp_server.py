#!/usr/bin/env python3
"""
MCP Server for Obsidiantools - Exposes all functionality from the obsidiantools Python package
for analyzing Obsidian.md vaults through the Model Context Protocol.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import pandas as pd
import networkx as nx
import yaml
import numpy as np
np.NaN = np.nan


# MCP SDK imports
from mcp import types as mcp_types
from mcp.server import Server, NotificationOptions, InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.server.models import InitializationOptions
# Specific types for clarity
from mcp.types import (
    Tool,
    TextContent,
    # ImageContent, # Not used in return types, but available
    # EmbeddedResource, # Not used in return types, but available
    ListToolsRequest,
    ListToolsResult,
    CallToolRequest,
    CallToolResult,
    ErrorData,
    ServerCapabilities,
    NotificationParams,
    Request,
    Result,
    ErrorData
)

# Obsidiantools imports
import obsidiantools.api as otools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPError(Exception):
    """Custom exception for MCP errors."""
    def __init__(self, error_data: ErrorData):
        self.error_data = error_data
        super().__init__(error_data.message)


class ObsidianToolsServer:
    """MCP Server for Obsidiantools functionality"""

    def __init__(self):
        self.vault: Optional[otools.Vault] = None
        self.vault_path: Optional[str] = "/Users/mohit/Library/Mobile Documents/iCloud~md~obsidian/Documents/mo"
        self.tool_definitions = self._get_tool_definitions()

        # Define server capabilities
        capabilities = mcp_types.ServerCapabilities(
            tools=mcp_types.ToolsCapability(enabled=True),
            resources=mcp_types.ResourcesCapability(enabled=False),
            prompts=mcp_types.PromptsCapability(enabled=False)
        )

        self.server = Server("obsidiantools-mcp-server")

        # Register handlers using decorators
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            return self.tool_definitions

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Optional[Dict[str, Any]] = None) -> List[Union[TextContent, ErrorData]]:
            if not arguments:
                arguments = {}
            
            content_list: Optional[List[Union[TextContent, ErrorData]]] = None
            error_val: Optional[ErrorData] = None

            try:
                result_data: Any
                if name == "init_vault":
                    result_data = await self.init_vault(**arguments)
                elif name == "connect_vault":
                    result_data = await self.connect_vault()
                elif name == "gather_vault":
                    result_data = await self.gather_vault()
                elif name == "get_graph_stats":
                    result_data = await self.get_graph_stats() # No args
                elif name == "export_graph":
                    result_data = await self.export_graph(**arguments)
                elif name == "get_note_metadata":
                    result_data = await self.get_note_metadata(**arguments)
                elif name == "get_media_metadata":
                    result_data = await self.get_media_metadata(**arguments)
                elif name == "get_canvas_metadata":
                    result_data = await self.get_canvas_metadata(**arguments)
                elif name == "get_file_indices":
                    result_data = await self.get_file_indices() # No args
                elif name == "get_nonexistent_files":
                    result_data = await self.get_nonexistent_files() # No args
                elif name == "get_isolated_files":
                    result_data = await self.get_isolated_files() # No args
                elif name == "get_all_links":
                    result_data = await self.get_all_links() # No args
                elif name == "get_note_links":
                    result_data = await self.get_note_links(**arguments)
                elif name == "get_note_content":
                    result_data = await self.get_note_content(**arguments)
                elif name == "get_all_tags":
                    result_data = await self.get_all_tags() # No args
                elif name == "find_notes_by_tag":
                    result_data = await self.find_notes_by_tag(**arguments)
                elif name == "get_canvas_content":
                    result_data = await self.get_canvas_content(**arguments)
                elif name == "search_notes":
                    result_data = await self.search_notes(**arguments)
                elif name == "analyze_vault_structure":
                    result_data = await self.analyze_vault_structure() # No args
                else:
                    error_val = ErrorData(kind="UnknownTool", message=f"Tool '{name}' not found.", code=404)

                if not error_val: # If tool was found and executed
                    if isinstance(result_data, dict) and "error" in result_data:
                        # Tool indicated an error in its return value
                        error_val = ErrorData(kind="ToolExecutionError", message=str(result_data["error"]), data=result_data, code=500)
                    elif isinstance(result_data, list) and result_data and isinstance(result_data[0], dict) and "error" in result_data[0]:
                         # Handle cases like search_notes returning [{"error": "..."}]
                        error_val = ErrorData(kind="ToolExecutionError", message=str(result_data[0]["error"]), data=result_data[0], code=500)
                    elif isinstance(result_data, str):
                        if result_data.lower().startswith("error:"):
                            error_val = ErrorData(kind="ToolExecutionError", message=result_data, code=500)
                        else:
                            content_list = [TextContent(type="text", text=result_data)]
                    elif isinstance(result_data, (dict, list)):
                        content_list = [TextContent(type="text", text=json.dumps(result_data, indent=2))]
                    else: # Fallback for other types, convert to string
                        content_list = [TextContent(type="text", text=str(result_data))]

            except Exception as e:
                logger.error(f"Error calling tool {name} with args {arguments}: {e}", exc_info=True)
                error_val = ErrorData(kind="InternalError", message=f"An unexpected error occurred while executing {name}: {str(e)}", code=500)

            if error_val:
                raise MCPError(error_val)
            elif content_list:
                return content_list
            else:
                # Should not happen if logic above is correct, but as a fallback:
                error_val = ErrorData(kind="InternalError", message=f"Tool '{name}' did not produce recognizable output or error.", code=500)
                raise MCPError(error_val)

    def _get_tool_definitions(self) -> List[Tool]:
        """Defines all tools available on this server."""
        return [
            Tool(
                name="init_vault",
                description="Initialize an Obsidian vault for analysis. Must be called before using other tools.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "vault_path": {
                            "type": "string",
                            "description": "Path to the Obsidian vault directory"
                        },
                        "include_subdirs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of subdirectories to include in analysis",
                            "default": None
                        },
                        "include_root": {
                            "type": "boolean",
                            "description": "Include files that are directly in the vault directory",
                            "default": True
                        }
                    },
                    "required": ["vault_path"]
                }
            ),
            Tool(
                name="connect_vault",
                description="Connect notes in the vault and build the graph structure. Call after init_vault.",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            Tool(
                name="gather_vault",
                description="Gather plaintext content from notes. Call after connect_vault for full functionality.",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            Tool(
                name="get_graph_stats",
                description="Get statistics about the vault's graph structure. Returns JSON.",
                inputSchema={"type": "object", "properties": {}}
            ),
            Tool(
                name="export_graph",
                description="Export the vault graph in various formats.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string",
                            "enum": ["gexf", "graphml", "json", "yaml"],
                            "description": "Export format for the graph"
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Path where to save the exported graph"
                        }
                    },
                    "required": ["format", "output_path"]
                }
            ),
            Tool(
                name="get_note_metadata",
                description="Get metadata about all notes in the vault. Returns JSON by default.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "as_json": {
                            "type": "boolean",
                            "description": "Return as JSON (list of records) instead of DataFrame string",
                            "default": True
                        }
                    }
                }
            ),
            Tool(
                name="get_media_metadata",
                description="Get metadata about media files in the vault. Returns JSON by default.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "as_json": {
                            "type": "boolean",
                            "description": "Return as JSON (list of records) instead of DataFrame string",
                            "default": True
                        }
                    }
                }
            ),
            Tool(
                name="get_canvas_metadata",
                description="Get metadata about canvas files in the vault. Returns JSON by default.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "as_json": {
                            "type": "boolean",
                            "description": "Return as JSON (list of records) instead of DataFrame string",
                            "default": True
                        }
                    }
                }
            ),
            Tool(
                name="get_file_indices",
                description="Get all file indices (md, media, canvas). Returns JSON.",
                inputSchema={"type": "object", "properties": {}}
            ),
            Tool(
                name="get_nonexistent_files",
                description="Get lists of files that are linked but don't exist. Returns JSON.",
                inputSchema={"type": "object", "properties": {}}
            ),
            Tool(
                name="get_isolated_files",
                description="Get lists of orphan files (not linked from anywhere). Returns JSON.",
                inputSchema={"type": "object", "properties": {}}
            ),
            Tool(
                name="get_all_links",
                description="Get all links in the vault organized by type. Returns JSON.",
                inputSchema={"type": "object", "properties": {}}
            ),
            Tool(
                name="get_note_links",
                description="Get all links for a specific note. Returns JSON.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "note_name": {
                            "type": "string",
                            "description": "Name of the note (without .md extension)"
                        }
                    },
                    "required": ["note_name"]
                }
            ),
            Tool(
                name="get_note_content",
                description="Get various forms of content for a specific note. Returns text or JSON.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "note_name": {
                            "type": "string",
                            "description": "Name of the note (without .md extension)"
                        },
                        "content_type": {
                            "type": "string",
                            "enum": ["source", "readable", "front_matter", "tags", "math"],
                            "description": "Type of content to retrieve"
                        }
                    },
                    "required": ["note_name", "content_type"]
                }
            ),
            Tool(
                name="get_all_tags",
                description="Get all tags used in the vault with their frequencies. Returns JSON.",
                inputSchema={"type": "object", "properties": {}}
            ),
            Tool(
                name="find_notes_by_tag",
                description="Find all notes that contain a specific tag. Returns a list of note names (JSON).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tag": {
                            "type": "string",
                            "description": "Tag to search for (with or without #)"
                        }
                    },
                    "required": ["tag"]
                }
            ),
            Tool(
                name="get_canvas_content",
                description="Get the content of a specific canvas file. Returns JSON.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "canvas_name": {
                            "type": "string",
                            "description": "Name of the canvas file (without .canvas extension)"
                        }
                    },
                    "required": ["canvas_name"]
                }
            ),
            Tool(
                name="search_notes",
                description="Search for notes containing specific text. Returns a list of matches (JSON).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Text to search for in notes"
                        },
                        "search_type": {
                            "type": "string",
                            "enum": ["source", "readable"],
                            "description": "Search in source or readable text",
                            "default": "readable"
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="analyze_vault_structure",
                description="Get a comprehensive analysis of the vault structure. Returns JSON.",
                inputSchema={"type": "object", "properties": {}}
            )
        ]

    async def init_vault(self, vault_path: str, include_subdirs: Optional[List[str]] = None, include_root: bool = True) -> str:
        logger.info(f"Initializing vault: {vault_path}, include_subdirs: {include_subdirs}, include_root: {include_root}")
        try:
            if not os.path.exists(vault_path):
                return f"Error: Vault path '{vault_path}' does not exist"

            from pathlib import Path
            self.vault = otools.Vault(
                dirpath=Path(vault_path),
                include_subdirs=include_subdirs,
                include_root=include_root
            )
            self.vault_path = vault_path
            return f"Successfully initialized vault at '{vault_path}'"
        except Exception as e:
            logger.error(f"Error initializing vault: {e}", exc_info=True)
            return f"Error initializing vault: {str(e)}"

    async def connect_vault(self) -> str:
        """Connect to an existing Obsidian vault."""
        if not self.vault_path:
            raise MCPError(ErrorData(code=400, message="No vault path specified. Call init_vault first."))
        try:
            self.vault = otools.Vault(self.vault_path)
            self.vault.connect()
            return f"Successfully connected to vault at {self.vault_path}"
        except Exception as e:
            raise MCPError(ErrorData(code=500, message=f"Failed to connect to vault: {str(e)}"))

    async def gather_vault(self) -> str:
        logger.info("Gathering vault")
        if not self.vault:
            return "Error: Vault not initialized. Call init_vault first."
        try:
            self.vault.gather()
            return "Successfully gathered vault content"
        except Exception as e:
            logger.error(f"Error gathering vault: {e}", exc_info=True)
            return f"Error gathering vault content: {str(e)}"

    async def get_graph_stats(self) -> Dict[str, Any]:
        logger.info("Getting graph stats")
        if not self.vault or not hasattr(self.vault, 'graph') or self.vault.graph is None:
            return {"error": "Vault not connected. Call connect_vault first."}
        try:
            graph = self.vault.graph
            if not isinstance(graph, nx.Graph): # Ensure graph is a NetworkX graph
                 return {"error": "Vault graph is not a valid NetworkX graph object."}
            if not graph.number_of_nodes():
                return {"message": "Graph is empty. No notes or connections found.", "num_nodes": 0, "num_edges": 0}

            stats = {
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
            }
            if graph.number_of_nodes() > 0: # Avoid division by zero for dense graphs or single node graphs
                 stats.update({
                    "density": nx.density(graph),
                    "is_connected": nx.is_weakly_connected(graph), # Use weakly for directed, is_connected for undirected
                    "num_components": nx.number_weakly_connected_components(graph),
                    "avg_degree": sum(dict(graph.degree()).values()) / graph.number_of_nodes()
                 })
                 # Get most connected notes
                 degree_centrality = nx.degree_centrality(graph)
                 top_notes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
                 stats["most_connected_notes"] = [{"note": note, "centrality": cent} for note, cent in top_notes]
            else:
                 stats.update({
                    "density": 0,
                    "is_connected": False,
                    "num_components": 0,
                    "avg_degree": 0,
                    "most_connected_notes": []
                 })
            return stats
        except Exception as e:
            logger.error(f"Error getting graph stats: {e}", exc_info=True)
            return {"error": f"Error getting graph stats: {str(e)}"}

    async def export_graph(self, format: str, output_path: str) -> str:
        logger.info(f"Exporting graph to {output_path} in {format} format")
        if not self.vault or not hasattr(self.vault, 'graph') or self.vault.graph is None:
            return "Error: Vault not connected. Call connect_vault first."
        try:
            graph = self.vault.graph
            if not isinstance(graph, nx.Graph):
                 return "Error: Vault graph is not a valid NetworkX graph object."

            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            if format == "gexf":
                nx.write_gexf(graph, output_path)
            elif format == "graphml":
                nx.write_graphml(graph, output_path)
            elif format == "json":
                data = nx.node_link_data(graph)
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)
            elif format == "yaml":
                data = nx.node_link_data(graph)
                with open(output_path, 'w') as f:
                    yaml.dump(data, f)
            else:
                return f"Error: Unsupported format '{format}'"
            return f"Successfully exported graph to '{output_path}' in {format} format"
        except Exception as e:
            logger.error(f"Error exporting graph: {e}", exc_info=True)
            return f"Error exporting graph: {str(e)}"

    async def get_note_metadata(self, as_json: bool = True) -> Union[str, List[Dict[str, Any]], Dict[str,str]]:
        logger.info(f"Getting note metadata, as_json: {as_json}")
        if not self.vault:
            return {"error": "Vault not initialized"} if as_json else "Error: Vault not initialized"
        try:
            # Ensure vault path is a Path object
            if not isinstance(self.vault._dirpath, Path):
                self.vault._dirpath = Path(self.vault._dirpath)
            
            # Get the raw metadata
            df = self.vault.get_note_metadata()
            if not isinstance(df, pd.DataFrame):
                msg = "Failed to retrieve metadata as a DataFrame."
                return {"error": msg} if as_json else f"Error: {msg}"
            
            # Log initial state
            logger.info(f"Initial DataFrame shape: {df.shape}")
            logger.info(f"Initial DataFrame columns: {df.columns.tolist()}")
            logger.info(f"Initial DataFrame dtypes:\n{df.dtypes}")
            
            # Create a copy to avoid modifying the original
            df = df.copy()
            
            # Handle numeric columns
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
            logger.info(f"Numeric columns: {numeric_columns.tolist()}")
            
            # Handle object columns that might be numeric
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    # Try to convert to numeric, coercing errors to NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    logger.info(f"Converted column {col} to numeric")
                except Exception as e:
                    logger.warning(f"Could not convert column {col} to numeric: {e}")
            
            # Handle NaN values
            df = df.replace({np.nan: None})
            
            if as_json:
                try:
                    # Convert to records
                    records = df.to_dict(orient='records')
                    logger.info(f"Successfully converted to {len(records)} records")
                    return records
                except Exception as e:
                    logger.error(f"Error converting DataFrame to JSON: {e}", exc_info=True)
                    return {"error": f"Error converting metadata to JSON: {str(e)}"}
            else:
                return df.to_string()
        except Exception as e:
            logger.error(f"Error getting metadata: {e}", exc_info=True)
            error_msg = f"Error getting metadata: {str(e)}"
            return {"error": error_msg} if as_json else error_msg

    async def get_media_metadata(self, as_json: bool = True) -> Union[str, List[Dict[str, Any]], Dict[str,str]]:
        logger.info(f"Getting media metadata, as_json: {as_json}")
        return await self._get_metadata_df(lambda: self.vault.get_media_file_metadata(), as_json)

    async def get_canvas_metadata(self, as_json: bool = True) -> Union[str, List[Dict[str, Any]], Dict[str,str]]:
        logger.info(f"Getting canvas metadata, as_json: {as_json}")
        return await self._get_metadata_df(lambda: self.vault.get_canvas_file_metadata(), as_json)


    async def get_file_indices(self) -> Dict[str, Any]:
        logger.info("Getting file indices")
        if not self.vault:
            return {"error": "Vault not initialized"}
        try:
            return {
                "md_files": list(getattr(self.vault, 'md_file_index', {}).keys()),
                "media_files": list(getattr(self.vault, 'media_file_index', {}).keys()),
                "canvas_files": list(getattr(self.vault, 'canvas_file_index', {}).keys())
            }
        except Exception as e:
            logger.error(f"Error getting file indices: {e}", exc_info=True)
            return {"error": f"Error getting file indices: {str(e)}"}

    async def get_nonexistent_files(self) -> Dict[str, Any]:
        logger.info("Getting non-existent files")
        if not self.vault:
            return {"error": "Vault not initialized"}
        try:
            return {
                "nonexistent_notes": list(getattr(self.vault, 'nonexistent_notes', [])),
                "nonexistent_media": list(getattr(self.vault, 'nonexistent_media_files', [])),
                "nonexistent_canvas": list(getattr(self.vault, 'nonexistent_canvas_files', []))
            }
        except Exception as e:
            logger.error(f"Error getting non-existent files: {e}", exc_info=True)
            return {"error": f"Error getting nonexistent files: {str(e)}"}

    async def get_isolated_files(self) -> Dict[str, Any]:
        logger.info("Getting isolated files")
        if not self.vault:
            return {"error": "Vault not initialized"}
        try:
            return {
                "isolated_notes": list(getattr(self.vault, 'isolated_notes', [])),
                "isolated_media": list(getattr(self.vault, 'isolated_media_files', [])),
                "isolated_canvas": list(getattr(self.vault, 'isolated_canvas_files', []))
            }
        except Exception as e:
            logger.error(f"Error getting isolated files: {e}", exc_info=True)
            return {"error": f"Error getting isolated files: {str(e)}"}

    async def get_all_links(self) -> Dict[str, Any]:
        logger.info("Getting all links")
        if not self.vault:
            return {"error": "Vault not initialized"}
        try:
            result: Dict[str, Any] = {}
            if hasattr(self.vault, 'backlinks_index') and self.vault.backlinks_index:
                result['backlinks'] = {k: list(v) for k, v in self.vault.backlinks_index.items()}
            if hasattr(self.vault, 'wikilinks_index') and self.vault.wikilinks_index:
                result['wikilinks'] = {k: list(v) for k, v in self.vault.wikilinks_index.items()}
            if hasattr(self.vault, 'embedded_files_index') and self.vault.embedded_files_index:
                result['embedded_files'] = {k: list(v) for k, v in self.vault.embedded_files_index.items()}
            if hasattr(self.vault, 'md_links_index') and self.vault.md_links_index:
                result['markdown_links'] = {k: list(v) for k, v in self.vault.md_links_index.items()}
            if not result:
                return {"message": "No link data found or vault not sufficiently processed (e.g., connect_vault)."}
            return result
        except Exception as e:
            logger.error(f"Error getting all links: {e}", exc_info=True)
            return {"error": f"Error getting links: {str(e)}"}

    async def get_note_links(self, note_name: str) -> Dict[str, Any]:
        logger.info(f"Getting links for note: {note_name}")
        if not self.vault:
            return {"error": "Vault not initialized"}
        try:
            result: Dict[str, Any] = {}
            # Ensure methods exist and handle potential errors if note_name not found by obsidiantools
            for link_type, method_name in [
                ('backlinks', 'get_backlinks'),
                ('wikilinks', 'get_wikilinks'),
                ('embedded_files', 'get_embedded_files'),
                ('markdown_links', 'get_md_links')
            ]:
                if hasattr(self.vault, method_name):
                    try:
                        links = getattr(self.vault, method_name)(note_name)
                        result[link_type] = list(links) if links else []
                    except KeyError: # otools might raise KeyError if note not found
                        result[link_type] = []
                        logger.warning(f"Note '{note_name}' not found when getting {link_type}.")
                    except Exception as ex:
                        logger.error(f"Error getting {link_type} for {note_name}: {ex}", exc_info=True)
                        result[link_type] = [{"error": f"Could not retrieve {link_type} for {note_name}"}]
            if not result:
                 return {"message": f"No links found or note '{note_name}' does not exist."}
            return result
        except Exception as e:
            logger.error(f"Error getting note links for {note_name}: {e}", exc_info=True)
            return {"error": f"Error getting note links for {note_name}: {str(e)}"}


    async def get_note_content(self, note_name: str, content_type: str) -> Union[str, Dict, List, Dict[str,str]]:
        logger.info(f"Getting content for note: {note_name}, type: {content_type}")
        if not self.vault:
            return {"error": "Vault not initialized"}

        method_map = {
            "source": "get_source_text",
            "readable": "get_readable_text",
            "front_matter": "get_front_matter",
            "tags": "get_tags",
            "math": "get_math"
        }

        default_unavailable_map = {
            "source": "Source text not available. Call gather_vault first.",
            "readable": "Readable text not available. Call gather_vault first.",
            "front_matter": {},
            "tags": [],
            "math": []
        }

        if content_type not in method_map:
            return {"error": f"Unknown content type: {content_type}"}

        try:
            method_name = method_map[content_type]
            if hasattr(self.vault, method_name):
                data = getattr(self.vault, method_name)(note_name)
                # Ensure tags and math are lists, as otools might return sets or other iterables
                if content_type in ["tags", "math"] and data is not None:
                    return list(data)
                return data if data is not None else default_unavailable_map[content_type]
            else:
                return default_unavailable_map[content_type]
        except KeyError: # Note not found by obsidiantools
             return {"error": f"Note '{note_name}' not found for content type '{content_type}'."}
        except Exception as e:
            logger.error(f"Error getting note content for {note_name} ({content_type}): {e}", exc_info=True)
            return {"error": f"Error getting note content for {note_name} ({content_type}): {str(e)}"}


    async def get_all_tags(self) -> Dict[str, Any]:
        logger.info("Getting all tags")
        if not self.vault:
            return {"error": "Vault not initialized"}
        try:
            if hasattr(self.vault, 'tags_index') and self.vault.tags_index:
                tag_counts: Dict[str, int] = {}
                for _note, tags_in_note in self.vault.tags_index.items():
                    for tag in tags_in_note:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
                return dict(sorted(tag_counts.items(), key=lambda item: item[1], reverse=True))
            else:
                return {"message": "No tags found or tags_index not available (e.g. call connect_vault)."}
        except Exception as e:
            logger.error(f"Error getting all tags: {e}", exc_info=True)
            return {"error": f"Error getting tags: {str(e)}"}

    async def find_notes_by_tag(self, tag: str) -> Union[List[str], Dict[str,str]]:
        logger.info(f"Finding notes by tag: {tag}")
        if not self.vault:
            return {"error": "Vault not initialized"} # Return dict for consistency with error handling
        try:
            normalized_tag = tag if tag.startswith('#') else f'#{tag}'
            notes_with_tag: List[str] = []
            if hasattr(self.vault, 'tags_index') and self.vault.tags_index:
                for note, tags_in_note in self.vault.tags_index.items():
                    if normalized_tag in tags_in_note:
                        notes_with_tag.append(note)
                return sorted(notes_with_tag)
            else: # No tags_index or it's empty
                return [] # Return empty list if no notes found or index unavailable
        except Exception as e:
            logger.error(f"Error finding notes by tag {tag}: {e}", exc_info=True)
            return {"error": f"Error finding notes by tag: {str(e)}"}


    async def get_canvas_content(self, canvas_name: str) -> Dict[str, Any]:
        logger.info(f"Getting content for canvas: {canvas_name}")
        if not self.vault:
            return {"error": "Vault not initialized"}
        try:
            if hasattr(self.vault, 'canvas_content_index'):
                if canvas_name in self.vault.canvas_content_index:
                    return self.vault.canvas_content_index[canvas_name]
                else:
                    return {"error": f"Canvas '{canvas_name}' not found in canvas_content_index."}
            else:
                return {"error": "Canvas content_index not available. Ensure vault is initialized and connected."}
        except Exception as e:
            logger.error(f"Error getting canvas content for {canvas_name}: {e}", exc_info=True)
            return {"error": f"Error getting canvas content: {str(e)}"}

    async def search_notes(self, query: str, search_type: str = "readable") -> Union[List[Dict[str, str]], Dict[str,str]]:
        logger.info(f"Searching notes with query '{query}', type: {search_type}")
        if not self.vault:
            return {"error": "Vault not initialized"} # Return dict for consistency

        results: List[Dict[str, str]] = []
        query_lower = query.lower()
        text_index_attr = f'{search_type}_text_index' # 'source_text_index' or 'readable_text_index'

        try:
            if not hasattr(self.vault, text_index_attr):
                return {"error": f"{search_type.capitalize()} text index not available. Call gather_vault first."}

            text_index = getattr(self.vault, text_index_attr, {})
            if not text_index: # Index exists but is empty
                 return [] # No notes to search

            for note, text in text_index.items():
                if text and query_lower in text.lower(): # Ensure text is not None
                    idx = text.lower().find(query_lower)
                    start = max(0, idx - 50)
                    end = min(len(text), idx + len(query) + 50)
                    context = text[start:end]
                    results.append({
                        "note": note,
                        "match_context": f"...{context}...",
                        "match_type": f"{search_type}_text"
                    })
            return results
        except Exception as e:
            logger.error(f"Error searching notes with query '{query}': {e}", exc_info=True)
            return {"error": f"Error searching notes: {str(e)}"}


    async def analyze_vault_structure(self) -> Dict[str, Any]:
        logger.info("Analyzing vault structure")
        if not self.vault:
            return {"error": "Vault not initialized"}

        try:
            analysis: Dict[str, Any] = {
                "vault_path": self.vault_path,
                "total_notes": len(getattr(self.vault, 'md_file_index', {})),
                "total_media": len(getattr(self.vault, 'media_file_index', {})),
                "total_canvas": len(getattr(self.vault, 'canvas_file_index', {})),
            }

            # Initialize graph_metrics with a default message
            analysis["graph_metrics"] = {"message": "Graph not connected or not available. Call connect_vault first."}

            if hasattr(self.vault, 'graph') and self.vault.graph is not None and isinstance(self.vault.graph, nx.Graph):
                graph = self.vault.graph
                if graph.number_of_nodes() > 0:
                    analysis["graph_metrics"] = {
                        "nodes": graph.number_of_nodes(),
                        "edges": graph.number_of_edges(),
                        "density": nx.density(graph),
                        "components": nx.number_weakly_connected_components(graph)
                    }

            wikilinks_index = getattr(self.vault, 'wikilinks_index', {})
            analysis["total_wikilinks"] = sum(len(links) for links in wikilinks_index.values())

            backlinks_index = getattr(self.vault, 'backlinks_index', {})
            if backlinks_index:
                analysis["avg_backlinks_per_note_with_backlinks"] = sum(len(links) for links in backlinks_index.values()) / len(backlinks_index)
            else:
                analysis["avg_backlinks_per_note_with_backlinks"] = 0

            tags_idx = getattr(self.vault, 'tags_index', {})
            all_tags = set()
            for tags_in_note in tags_idx.values():
                all_tags.update(tags_in_note)
            analysis["unique_tags"] = len(all_tags)

            return analysis
        except Exception as e:
            logger.error(f"Error analyzing vault structure: {e}", exc_info=True)
            return {"error": f"Error analyzing vault: {str(e)}"}


    async def run(self):
        """Run the MCP server using stdio."""
        logger.info("Starting ObsidianTools MCP server...")
        async with stdio_server() as (reader, writer):
            await self.server.run(
                reader,
                writer,
                InitializationOptions(
                    server_name="obsidiantools-mcp-server",
                    server_version="1.0.1",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
        logger.info("ObsidianTools MCP server stopped.")


def main():
    """Main entry point"""
    server_instance = ObsidianToolsServer()
    try:
        asyncio.run(server_instance.run())
    except KeyboardInterrupt:
        logger.info("ObsidianTools MCP server shutting down due to KeyboardInterrupt.")
    except Exception as e:
        logger.critical(f"ObsidianTools MCP server exited with a critical error: {e}", exc_info=True)

if __name__ == "__main__":
    main()