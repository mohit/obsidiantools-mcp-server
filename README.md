# Obsidiantools MCP Server

An MCP (Model Context Protocol) server that exposes all functionality from the [obsidiantools](https://github.com/mfarragher/obsidiantools) Python package for analyzing Obsidian.md vaults. This allows AI assistants to analyze and interact with Obsidian vaults programmatically.

## Features

The MCP server provides comprehensive access to obsidiantools functionality:

### Vault Initialization & Connection
- **init_vault**: Initialize a vault for analysis with optional subdirectory filtering
- **connect_vault**: Build the graph structure with optional attachment inclusion
- **gather_vault**: Gather plaintext content from notes with optional code removal

### Graph Analysis
- **get_graph_stats**: Get comprehensive graph statistics (nodes, edges, density, centrality)
- **export_graph**: Export vault graph in multiple formats (GEXF, GraphML, JSON, YAML)

### Metadata Extraction
- **get_note_metadata**: Get detailed metadata for all notes as DataFrame/JSON
- **get_media_metadata**: Get metadata for media files
- **get_canvas_metadata**: Get metadata for canvas files

### File Management
- **get_file_indices**: List all files by type (notes, media, canvas)
- **get_nonexistent_files**: Find broken links to non-existent files
- **get_isolated_files**: Find orphan files not linked from anywhere

### Link Analysis
- **get_all_links**: Get all links organized by type (wikilinks, backlinks, embedded, markdown)
- **get_note_links**: Get all links for a specific note

### Content Extraction
- **get_note_content**: Extract various content types (source, readable, front matter, tags, math)
- **search_notes**: Full-text search across notes

### Tag Management
- **get_all_tags**: Get all tags with usage frequencies
- **find_notes_by_tag**: Find all notes containing a specific tag

### Canvas Support
- **get_canvas_content**: Get JSON content of canvas files

### Analysis
- **analyze_vault_structure**: Get comprehensive vault structure analysis

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd obsidiantools-mcp-server
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make the server executable:
```bash
chmod +x obsidiantools_mcp_server.py
```

## Usage

### As an MCP Server

1. Add to your MCP client configuration (e.g., Claude Desktop):

```json
{
  "mcpServers": {
    "obsidiantools": {
      "command": "python",
      "args": ["/path/to/obsidiantools_mcp_server.py"]
    }
  }
}
```

2. The server will be available to your AI assistant for vault analysis.

### Direct Python Usage

```python
import obsidiantools.api as otools

# Initialize and analyze a vault
vault = otools.Vault('/path/to/obsidian/vault')
vault.connect().gather()

# Get metadata
df = vault.get_note_metadata()
print(df)

# Access the graph
import networkx as nx
print(nx.info(vault.graph))
```

## Example Workflow

Here's a typical workflow for analyzing an Obsidian vault:

1. **Initialize the vault**:
   ```
   init_vault(vault_path="/path/to/vault")
   ```

2. **Connect to build graph**:
   ```
   connect_vault(include_attachments=true)
   ```

3. **Gather content** (optional, needed for text analysis):
   ```
   gather_vault(remove_code=false)
   ```

4. **Analyze the vault**:
   ```
   get_graph_stats()
   analyze_vault_structure()
   get_note_metadata(as_json=true)
   ```

5. **Search and explore**:
   ```
   search_notes(query="machine learning")
   find_notes_by_tag(tag="research")
   ```

## Tool Reference

### Initialization Tools

#### init_vault
- **Parameters**: 
  - `vault_path` (required): Path to Obsidian vault
  - `include_subdirs` (optional): List of subdirectories to include
- **Returns**: Success/error message

#### connect_vault
- **Parameters**:
  - `include_attachments` (optional, default: false): Include attachments in graph
- **Returns**: Graph connection info

#### gather_vault
- **Parameters**:
  - `remove_code` (optional, default: false): Remove code blocks from text
- **Returns**: Success/error message

### Analysis Tools

#### get_graph_stats
- **Parameters**: None
- **Returns**: Dictionary with graph metrics and most connected notes

#### analyze_vault_structure
- **Parameters**: None
- **Returns**: Comprehensive vault analysis with counts and statistics

### Metadata Tools

#### get_note_metadata
- **Parameters**:
  - `as_json` (optional, default: true): Return as JSON or DataFrame string
- **Returns**: Note metadata including links, tags, and structure

### Search Tools

#### search_notes
- **Parameters**:
  - `query` (required): Text to search for
  - `search_type` (optional): "source" or "readable"
- **Returns**: List of matches with context

#### find_notes_by_tag
- **Parameters**:
  - `tag` (required): Tag to search for (with or without #)
- **Returns**: List of note names

## Requirements

- Python 3.9 or higher
- Obsidian vault with markdown files
- All dependencies listed in requirements.txt

## Limitations

- The package works best with vaults using standard wikilink syntax
- Relative paths in wikilinks (like `[[../note]]`) are not fully supported
- The graph may differ slightly from Obsidian's graph view (see [obsidiantools wiki](https://github.com/mfarragher/obsidiantools/wiki))

## Troubleshooting

### "Vault not initialized" error
Make sure to call `init_vault` with a valid vault path before using other tools.

### "Vault not connected" error
Call `connect_vault` after `init_vault` to build the graph structure.

### Missing content errors
Call `gather_vault` after connecting to enable text search and content extraction.

### Graph export errors
Ensure you have write permissions to the output path and the required libraries (e.g., PyYAML for YAML export).

## License

This MCP server follows the licensing of the obsidiantools package (Modified BSD 3-clause).

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

This MCP server is built on top of the excellent [obsidiantools](https://github.com/mfarragher/obsidiantools) package by Mark Farragher