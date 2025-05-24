#!/usr/bin/env python3
"""
MCP Server for Obsidiantools - Exposes all functionality from the obsidiantools Python package
for analyzing Obsidian.md vaults through the Model Context Protocol.

Handles automatic vault initialization based on client-provided settings.
"""

import asyncio
import json
import logging
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import pandas as pd
import networkx as nx
import yaml
import numpy as np
np.NaN = np.nan # type: ignore # pd.NA is preferred but otools might use np.nan

# MCP SDK imports
from mcp import types as mcp_types
from mcp.server import Server, NotificationOptions # InitializationOptions is from mcp.server.models
from mcp.server.stdio import stdio_server
from mcp.server.models import InitializationOptions as ServerInitOptions # Renamed to avoid clash
# Specific types for clarity
from mcp.types import (
    Tool,
    TextContent,
    ListToolsRequest,
    ListToolsResult,
    CallToolRequest,
    CallToolResult,
    ErrorData,
    ServerCapabilities,
    NotificationParams,
    Request,
    Result,
    InitializeRequestParams, # For type hinting client params
    InitializeResult
)

# Obsidiantools imports
import obsidiantools.api as otools

# Configure logging
logging.basicConfig(stream=sys.stderr,level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
        self.vault_path: Optional[str] = os.environ.get("VAULT_PATH")
        self.tool_definitions = self._get_tool_definitions()
        self.auto_init = os.environ.get("AUTO_INIT", "true").lower() == "true"
        self.include_subdirs = os.environ.get("INCLUDE_SUBDIRS", "true").lower() == "true"
        self.include_root = os.environ.get("INCLUDE_ROOT", "true").lower() == "true"
    
        self.server = Server("obsidiantools-mcp-server")        
        logger.info(f"ObsidianToolsServer initialized. Auto-init: {self.auto_init}")

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
                # Dispatch to the appropriate tool method
                if name == "init_vault":
                    result_data = await self.init_vault(**arguments)
                elif name == "connect_vault":
                    result_data = await self.connect_vault() # No args for the tool itself
                elif name == "gather_vault":
                    result_data = await self.gather_vault() # No args
                elif name == "get_graph_stats":
                    result_data = await self.get_graph_stats()
                elif name == "export_graph":
                    result_data = await self.export_graph(**arguments)
                elif name == "get_note_metadata":
                    result_data = await self.get_note_metadata(**arguments)
                elif name == "get_media_metadata":
                    result_data = await self.get_media_metadata(**arguments)
                elif name == "get_canvas_metadata":
                    result_data = await self.get_canvas_metadata(**arguments)
                elif name == "get_file_indices":
                    result_data = await self.get_file_indices()
                elif name == "get_nonexistent_files":
                    result_data = await self.get_nonexistent_files()
                elif name == "get_isolated_files":
                    result_data = await self.get_isolated_files()
                elif name == "get_all_links":
                    result_data = await self.get_all_links()
                elif name == "get_note_links":
                    result_data = await self.get_note_links(**arguments)
                elif name == "get_note_content":
                    result_data = await self.get_note_content(**arguments)
                elif name == "get_all_tags":
                    result_data = await self.get_all_tags()
                elif name == "find_notes_by_tag":
                    result_data = await self.find_notes_by_tag(**arguments)
                elif name == "get_canvas_content":
                    result_data = await self.get_canvas_content(**arguments)
                elif name == "search_notes":
                    result_data = await self.search_notes(**arguments)
                elif name == "analyze_vault_structure":
                    result_data = await self.analyze_vault_structure()
                else:
                    error_val = ErrorData(kind="UnknownTool", message=f"Tool '{name}' not found.", code=404)

                # Process result_data if no error_val yet
                if not error_val:
                    if isinstance(result_data, dict) and "error" in result_data:
                        error_val = ErrorData(kind="ToolExecutionError", message=str(result_data["error"]), data=result_data, code=500)
                    elif isinstance(result_data, list) and result_data and isinstance(result_data[0], dict) and "error" in result_data[0]:
                        error_val = ErrorData(kind="ToolExecutionError", message=str(result_data[0]["error"]), data=result_data[0], code=500)
                    elif isinstance(result_data, str):
                        if result_data.lower().startswith("error:"):
                            error_val = ErrorData(kind="ToolExecutionError", message=result_data, code=500)
                        else:
                            content_list = [TextContent(type="text", text=result_data)]
                    elif isinstance(result_data, (dict, list)):
                        content_list = [TextContent(type="text", text=json.dumps(result_data, indent=2))]
                    else: # Fallback for other types
                        content_list = [TextContent(type="text", text=str(result_data))]
            
            except MCPError as mcp_e: # Catch MCPError if raised by tools like the original connect_vault
                logger.error(f"MCPError calling tool {name} with args {arguments}: {mcp_e.error_data.message}", exc_info=False) # exc_info=False as MCPError is structured
                error_val = mcp_e.error_data
            except Exception as e:
                logger.error(f"Error calling tool {name} with args {arguments}: {e}", exc_info=True)
                error_val = ErrorData(kind="InternalError", message=f"An unexpected error occurred while executing {name}: {str(e)}", code=500)

            if error_val:
                # The MCP SDK expects the handler to raise for errors to be converted to ErrorResult
                raise MCPError(error_val) 
            elif content_list:
                return content_list
            else:
                # This case should ideally not be reached if logic is correct
                fallback_error = ErrorData(kind="InternalError", message=f"Tool '{name}' did not produce recognizable output or error.", code=500)
                raise MCPError(fallback_error)
        

    async def perform_automatic_setup(self):
        """
        Performs automatic vault initialization, connection, and gathering based on client config.
        Sends notifications to the client about progress.
        """
        vault_path = self.vault_path
        if not vault_path:
            logger.error("Auto-init: vault_path missing in client configuration.")
            raise MCPError(ErrorData(kind="InvalidConfiguration", message="Auto-init failed: vault_path not provided by client.", code=400))
            return

        try:
            # Step 1: Init Vault
            init_args = {
                "vault_path": vault_path,
                "include_subdirs": None if self.include_subdirs else [],  # Convert bool to None/empty list
                "include_root": self.include_root
            }

            logger.info(f"Auto-init: Calling init_vault with {init_args}")
            init_result = await self.init_vault(**init_args)

            if init_result.lower().startswith("error:"):
                raise Exception(f"init_vault failed: {init_result}")
            logger.info(f"Auto-init: init_vault successful. {init_result}")


            # Step 2: Connect Vault
            logger.info("Auto-init: Calling connect_vault")
            connect_result = await self.connect_vault() # Uses self.vault_path set by init_vault
            if connect_result.lower().startswith("error:"):
                raise Exception(f"connect_vault failed: {connect_result}")
            logger.info(f"Auto-init: connect_vault successful. {connect_result}")

            # Step 3: Gather Vault
            logger.info("Auto-init: Calling gather_vault")
            gather_result = await self.gather_vault()
            if gather_result.lower().startswith("error:"):
                raise Exception(f"gather_vault failed: {gather_result}")
            logger.info(f"Auto-init: gather_vault successful. {gather_result}")

            logger.info("Auto-init: All steps completed successfully.")

        except Exception as e:
            logger.error(f"Auto-init sequence failed: {e}", exc_info=True)
            raise MCPError(ErrorData(kind="InternalError", message=f"Auto-init sequence failed: {str(e)}", code=500))

    def _get_tool_definitions(self) -> List[Tool]:
        """Defines all tools available on this server."""
        # (Tool definitions remain the same as in the original code)
        # ... (omitted for brevity, but should be the same as provided)
        return [
            Tool(
                name="init_vault",
                description="Initialize an Obsidian vault for analysis. Must be called before using other tools. Not required if auto-init is enabled.",
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
                description="Connect notes in the vault and build the graph structure. Call after init_vault. Not required if auto-init is enabled.",
                inputSchema={ # No arguments needed for the tool call itself
                    "type": "object",
                    "properties": {}
                }
            ),
            Tool(
                name="gather_vault",
                description="Gather plaintext content from notes. Call after connect_vault for full functionality. Not required if auto-init is enabled.",
                inputSchema={ # No arguments needed
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

            self.vault = otools.Vault(
                dirpath=Path(vault_path), # obsidiantools expects Path here
                include_subdirs=include_subdirs,
                include_root=include_root
            )
            self.vault_path = vault_path # Store the string path for connect_vault and others
            logger.info(f"Successfully initialized vault at '{vault_path}'")
            return f"Successfully initialized vault at '{vault_path}'"
        except Exception as e:
            logger.error(f"Error initializing vault: {e}", exc_info=True)
            return f"Error: Error initializing vault: {str(e)}"

    async def connect_vault(self) -> str:
        """Connects to an existing Obsidian vault. Uses self.vault_path set by init_vault."""
        logger.info(f"Attempting to connect vault using path: {self.vault_path}")
        if not self.vault_path:
            logger.warning("connect_vault called but self.vault_path is not set. Call init_vault first.")
            return "Error: No vault path specified. Call init_vault first."
        try:
            # otools.Vault can take a string path.
            # This will create a new Vault instance and connect it.
            self.vault = otools.Vault(self.vault_path)
            self.vault.connect() # This performs the actual scanning and graph building.
            logger.info(f"Successfully connected to vault at {self.vault_path}")
            return f"Successfully connected to vault at {self.vault_path}"
        except Exception as e:
            logger.error(f"Error connecting vault at {self.vault_path}: {e}", exc_info=True)
            return f"Error: Failed to connect to vault: {str(e)}"

    async def gather_vault(self) -> str:
        logger.info("Gathering vault content")
        if not self.vault:
            return "Error: Vault not initialized or connected. Call init_vault and connect_vault first."
        if not hasattr(self.vault, 'md_file_index') or not self.vault.md_file_index:
             # Check if connect() was successful by looking for an attribute it populates
            return "Error: Vault might not be properly connected (e.g., no files found or connect failed). Call connect_vault first."
        try:
            self.vault.gather() # Gathers plaintext content
            logger.info("Successfully gathered vault content")
            return "Successfully gathered vault content"
        except Exception as e:
            logger.error(f"Error gathering vault: {e}", exc_info=True)
            return f"Error: Error gathering vault content: {str(e)}"

    # --- Other tool methods (get_graph_stats, export_graph, etc.) ---
    # These methods remain largely the same as in the original code.
    # Minor adjustments for error messages or consistency if needed.
    # For brevity, only showing a few key ones or ones with potential adjustments.

    async def get_graph_stats(self) -> Dict[str, Any]:
        logger.info("Getting graph stats")
        if not self.vault or not hasattr(self.vault, 'graph') or self.vault.graph is None:
            return {"error": "Vault not connected or graph not built. Call init_vault and connect_vault first."}
        try:
            graph = self.vault.graph
            if not isinstance(graph, nx.Graph): 
                return {"error": "Vault graph is not a valid NetworkX graph object."}
            if not graph.number_of_nodes(): # Changed from `if not graph.number_of_nodes():` to handle empty graph
                return {"message": "Graph is empty. No notes or connections found.", "num_nodes": 0, "num_edges": 0}

            stats = {
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
            }
            if graph.number_of_nodes() > 0:
                stats.update({
                    "density": nx.density(graph),
                    "is_connected": nx.is_weakly_connected(graph) if graph.is_directed() else nx.is_connected(graph),
                    "num_components": nx.number_weakly_connected_components(graph) if graph.is_directed() else nx.number_connected_components(graph),
                    "avg_degree": sum(dict(graph.degree()).values()) / float(graph.number_of_nodes()) # Ensure float division
                })
                degree_centrality = nx.degree_centrality(graph)
                top_notes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
                stats["most_connected_notes"] = [{"note": note, "centrality": float(cent)} for note, cent in top_notes] # Ensure float
            else: # Should be covered by the empty graph check above, but good for robustness
                stats.update({
                    "density": 0.0, "is_connected": False, "num_components": 0, 
                    "avg_degree": 0.0, "most_connected_notes": []
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

            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

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
            return f"Error: Error exporting graph: {str(e)}"

    async def _get_metadata_df_common(self, df_provider_func: callable, as_json: bool) -> Union[str, List[Dict[str, Any]], Dict[str,str]]:
        """Helper for DataFrame based metadata getters."""
        if not self.vault:
            return {"error": "Vault not initialized"} if as_json else "Error: Vault not initialized"
        try:
            df = df_provider_func()
            if not isinstance(df, pd.DataFrame):
                msg = "Failed to retrieve metadata as a DataFrame."
                return {"error": msg} if as_json else f"Error: {msg}"
            
            df = df.copy() # Avoid SettingWithCopyWarning

            # Convert datetime columns to ISO format strings
            for col in df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
                df[col] = df[col].dt.strftime('%Y-%m-%dT%H:%M:%S%z') # ISO 8601 with timezone
            
            # Replace pd.NA and np.nan with None for JSON serialization
            # df = df.replace({pd.NA: None, np.nan: None}) # More robust NaN/NA handling
            df = df.astype(object).where(pd.notnull(df), None)


            if as_json:
                try:
                    records = df.to_dict(orient='records')
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
            
    async def get_note_metadata(self, as_json: bool = True) -> Union[str, List[Dict[str, Any]], Dict[str,str]]:
        logger.info(f"Getting note metadata, as_json: {as_json}")
        return await self._get_metadata_df_common(lambda: self.vault.get_note_metadata(), as_json)

    async def get_media_metadata(self, as_json: bool = True) -> Union[str, List[Dict[str, Any]], Dict[str,str]]:
        logger.info(f"Getting media metadata, as_json: {as_json}")
        return await self._get_metadata_df_common(lambda: self.vault.get_media_file_metadata(), as_json)

    async def get_canvas_metadata(self, as_json: bool = True) -> Union[str, List[Dict[str, Any]], Dict[str,str]]:
        logger.info(f"Getting canvas metadata, as_json: {as_json}")
        return await self._get_metadata_df_common(lambda: self.vault.get_canvas_file_metadata(), as_json)

    async def get_file_indices(self) -> Dict[str, Any]:
        logger.info("Getting file indices")
        if not self.vault: return {"error": "Vault not initialized"}
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
        if not self.vault: return {"error": "Vault not initialized"}
        if not hasattr(self.vault, 'nonexistent_notes'): # Check if connect() has run
             return {"error": "Vault not connected. Call connect_vault first."}
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
        if not self.vault: return {"error": "Vault not initialized"}
        if not hasattr(self.vault, 'isolated_notes'): # Check if connect() has run
             return {"error": "Vault not connected. Call connect_vault first."}
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
        if not self.vault: return {"error": "Vault not initialized"}
        if not hasattr(self.vault, 'wikilinks_index'): # Check if connect() has run
             return {"error": "Vault not connected. Call connect_vault first."}
        try:
            result: Dict[str, Any] = {}
            # Helper to convert sets to lists for JSON serialization
            def get_links_from_index(index_attr_name: str) -> Dict[str, List[str]]:
                index_data = getattr(self.vault, index_attr_name, {})
                return {k: list(v) for k, v in index_data.items()}

            result['backlinks'] = get_links_from_index('backlinks_index')
            result['wikilinks'] = get_links_from_index('wikilinks_index')
            result['embedded_files'] = get_links_from_index('embedded_files_index')
            result['markdown_links'] = get_links_from_index('md_links_index')
            
            if not any(result.values()): # Check if all link lists are empty
                return {"message": "No link data found or vault not sufficiently processed (e.g., connect_vault)."}
            return result
        except Exception as e:
            logger.error(f"Error getting all links: {e}", exc_info=True)
            return {"error": f"Error getting links: {str(e)}"}

    async def get_note_links(self, note_name: str) -> Dict[str, Any]:
        logger.info(f"Getting links for note: {note_name}")
        if not self.vault: return {"error": "Vault not initialized"}
        if not hasattr(self.vault, 'md_file_index') or note_name not in self.vault.md_file_index:
            return {"error": f"Note '{note_name}' not found in vault or vault not connected."}
        try:
            result: Dict[str, Any] = {}
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
                    except KeyError: 
                        result[link_type] = []
                        logger.warning(f"Note '{note_name}' caused KeyError when getting {link_type} (might be ok if no such links).")
                    except Exception as ex_inner:
                        logger.error(f"Error getting {link_type} for {note_name}: {ex_inner}", exc_info=True)
                        result[link_type] = [{"error": f"Could not retrieve {link_type} for {note_name}: {str(ex_inner)}"}]
            if not result: # Should not happen if note exists
                 return {"message": f"No link data structure available for note '{note_name}'."}
            return result
        except Exception as e:
            logger.error(f"Error getting note links for {note_name}: {e}", exc_info=True)
            return {"error": f"Error getting note links for {note_name}: {str(e)}"}

    async def get_note_content(self, note_name: str, content_type: str) -> Union[str, Dict, List, Dict[str,str]]:
        logger.info(f"Getting content for note: {note_name}, type: {content_type}")
        if not self.vault: return {"error": "Vault not initialized"}
        if not hasattr(self.vault, 'md_file_index') or note_name not in self.vault.md_file_index:
             return {"error": f"Note '{note_name}' not found in vault."}
        if content_type in ["source", "readable"] and (not hasattr(self.vault, 'source_text_index') or not hasattr(self.vault, 'readable_text_index')):
             return {"error": f"Text content not gathered for note '{note_name}'. Call gather_vault first."}


        method_map = {
            "source": "get_source_text", "readable": "get_readable_text",
            "front_matter": "get_front_matter", "tags": "get_tags", "math": "get_math"
        }
        default_unavailable_map = { # Default values if method returns None
            "source": "Source text not available.", "readable": "Readable text not available.",
            "front_matter": {}, "tags": [], "math": []
        }

        if content_type not in method_map:
            return {"error": f"Unknown content type: {content_type}"}
        
        try:
            method_name = method_map[content_type]
            if hasattr(self.vault, method_name):
                data = getattr(self.vault, method_name)(note_name)
                if data is None: # otools methods might return None
                    return default_unavailable_map[content_type]
                if content_type in ["tags", "math"] and not isinstance(data, list):
                    return list(data) # Ensure list for tags/math
                return data
            else: # Should not happen if vault object is standard
                return {"error": f"Method {method_name} not found on vault object."}
        except KeyError: # Note not found by specific otools getter
             return {"error": f"Note '{note_name}' not found for content type '{content_type}' by underlying method."}
        except Exception as e:
            logger.error(f"Error getting note content for {note_name} ({content_type}): {e}", exc_info=True)
            return {"error": f"Error getting note content for {note_name} ({content_type}): {str(e)}"}

    async def get_all_tags(self) -> Dict[str, Any]:
        logger.info("Getting all tags")
        if not self.vault: return {"error": "Vault not initialized"}
        if not hasattr(self.vault, 'tags_index'): # Check if connect() has run and populated tags_index
             return {"error": "Tags index not available. Call connect_vault first."}
        try:
            if self.vault.tags_index:
                tag_counts: Dict[str, int] = {}
                for _note, tags_in_note in self.vault.tags_index.items():
                    for tag in tags_in_note:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
                return dict(sorted(tag_counts.items(), key=lambda item: item[1], reverse=True))
            else:
                return {"message": "No tags found in the vault."}
        except Exception as e:
            logger.error(f"Error getting all tags: {e}", exc_info=True)
            return {"error": f"Error getting tags: {str(e)}"}

    async def find_notes_by_tag(self, tag: str) -> Union[List[str], Dict[str,str]]:
        logger.info(f"Finding notes by tag: {tag}")
        if not self.vault: return {"error": "Vault not initialized"}
        if not hasattr(self.vault, 'tags_index'):
             return {"error": "Tags index not available. Call connect_vault first."}
        try:
            normalized_tag = tag if tag.startswith('#') else f'#{tag}'
            notes_with_tag: List[str] = []
            if self.vault.tags_index:
                for note, tags_in_note in self.vault.tags_index.items():
                    if normalized_tag in tags_in_note:
                        notes_with_tag.append(note)
                return sorted(notes_with_tag)
            else: # tags_index exists but is empty
                return [] 
        except Exception as e:
            logger.error(f"Error finding notes by tag {tag}: {e}", exc_info=True)
            return {"error": f"Error finding notes by tag: {str(e)}"}

    async def get_canvas_content(self, canvas_name: str) -> Dict[str, Any]:
        logger.info(f"Getting content for canvas: {canvas_name}")
        if not self.vault: return {"error": "Vault not initialized"}
        if not hasattr(self.vault, 'canvas_content_index'):
             return {"error": "Canvas content_index not available. Call connect_vault and gather_vault first."}
        try:
            if canvas_name in self.vault.canvas_content_index:
                return self.vault.canvas_content_index[canvas_name]
            else:
                return {"error": f"Canvas '{canvas_name}' not found in canvas_content_index."}
        except Exception as e:
            logger.error(f"Error getting canvas content for {canvas_name}: {e}", exc_info=True)
            return {"error": f"Error getting canvas content: {str(e)}"}

    async def search_notes(self, query: str, search_type: str = "readable") -> Union[List[Dict[str, str]], Dict[str,str]]:
        logger.info(f"Searching notes with query '{query}', type: {search_type}")
        if not self.vault: return {"error": "Vault not initialized"}

        text_index_attr = f'{search_type}_text_index'
        if not hasattr(self.vault, text_index_attr):
            return {"error": f"{search_type.capitalize()} text index not available. Call gather_vault first."}
        
        results: List[Dict[str, str]] = []
        query_lower = query.lower()
        try:
            text_index = getattr(self.vault, text_index_attr, {})
            if not text_index: return [] # No notes to search or index empty

            for note, text in text_index.items():
                if text and query_lower in text.lower():
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
        if not self.vault: return {"error": "Vault not initialized"}

        try:
            analysis: Dict[str, Any] = {
                "vault_path": self.vault_path,
                "total_notes": len(getattr(self.vault, 'md_file_index', {})),
                "total_media": len(getattr(self.vault, 'media_file_index', {})),
                "total_canvas": len(getattr(self.vault, 'canvas_file_index', {})),
            }
            analysis["graph_metrics"] = {"message": "Graph not connected or not available. Call connect_vault first."}
            if hasattr(self.vault, 'graph') and self.vault.graph is not None and isinstance(self.vault.graph, nx.Graph):
                graph = self.vault.graph
                if graph.number_of_nodes() > 0:
                    analysis["graph_metrics"] = {
                        "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
                        "density": nx.density(graph),
                        "components": nx.number_weakly_connected_components(graph) if graph.is_directed() else nx.number_connected_components(graph)
                    }
            
            wikilinks_index = getattr(self.vault, 'wikilinks_index', {})
            analysis["total_wikilinks"] = sum(len(links) for links in wikilinks_index.values())

            backlinks_index = getattr(self.vault, 'backlinks_index', {})
            if backlinks_index: # Avoid division by zero
                analysis["avg_backlinks_per_note_with_backlinks"] = sum(len(links) for links in backlinks_index.values()) / float(len(backlinks_index))
            else:
                analysis["avg_backlinks_per_note_with_backlinks"] = 0.0
            
            tags_idx = getattr(self.vault, 'tags_index', {})
            all_tags_set = set()
            for tags_in_note in tags_idx.values(): all_tags_set.update(tags_in_note)
            analysis["unique_tags"] = len(all_tags_set)

            return analysis
        except Exception as e:
            logger.error(f"Error analyzing vault structure: {e}", exc_info=True)
            return {"error": f"Error analyzing vault: {str(e)}"}

    async def run(self):
        """Run the MCP server using stdio."""
        logger.info("Starting ObsidianTools MCP server...")
        # Define server capabilities (these are determined by registered handlers by the Server class)
        # The ServerInitOptions capabilities field will be populated by self.server.get_capabilities()
        
        # Define server's own initialization options to send to client
        # The capabilities are dynamically generated by self.server.get_capabilities()
        server_init_opts = ServerInitOptions(
            server_name="obsidiantools-mcp-server",
            server_version="1.0.2", # Incremented version
            capabilities=self.server.get_capabilities( # Let the server instance build its capabilities
                notification_options=NotificationOptions(), # Example: configure notifications if needed
                experimental_capabilities={},
            ),
            instructions="""
            This server is an MCP server for ObsidianTools.
            It is used to analyze Obsidian vaults and provide information about the vault structure.
            It is not required to use the tools, but it is recommended to call init_vault and connect_vault before using the tools.
            """ + (f"""
            Auto-init is enabled.
            """ if self.auto_init else "")
        )
        logger.info(f"Server capabilities to be sent: {server_init_opts.capabilities.model_dump_json(indent=2)}")
        
        if self.auto_init:
            await self.perform_automatic_setup()

        async with stdio_server() as (reader, writer):
            await self.server.run(
                reader,
                writer,
                server_init_opts, # Pass the server's own init options
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
