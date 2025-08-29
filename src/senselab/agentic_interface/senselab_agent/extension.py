"""This is the senselab AI agentic interface."""

import asyncio
import logging
import threading

from notebook_intelligence import (
    ChatCommand,
    ChatRequest,
    ChatResponse,
    Host,
    MarkdownData,
    NotebookIntelligenceExtension,
    Tool,
)
from notebook_intelligence.mcp_manager import (
    MCPChatParticipant,
    MCPServer,
    MCPServerImpl,
    SSEServerParameters,
    StdioServerParameters,
)

PARTICIPANT_ICON_URL = "https://avatars.githubusercontent.com/u/47326880?s=200&v=4"

from .notebook_share_tool import NotebookShareTool

log = logging.getLogger(__name__)


import os
from typing import Union

from notebook_intelligence import ToolPreInvokeResponse


class CreateNewNotebookTool(Tool):
    def __init__(self, auto_approve: bool = False):
        self._auto_approve = auto_approve
        super().__init__()

    @property
    def name(self) -> str:
        return "create_new_notebook"

    @property
    def title(self) -> str:
        return "Create new notebook with the provided code and markdown cells"

    @property
    def tags(self) -> list[str]:
        return ["senselab-ai"]

    @property
    def description(self) -> str:
        return "Create a new notebook, then append the provided cells in order."

    @property
    def schema(self) -> dict:
        # IMPORTANT: include `required` inside the `items` object
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell_sources": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "cell_type": {
                                        "type": "string",
                                        "enum": ["code", "markdown"],
                                        "description": "Type of the cell to create."
                                    },
                                    "source": {
                                        "type": "string",
                                        "description": "The content of the cell."
                                    }
                                },
                                "required": ["cell_type", "source"],
                                "additionalProperties": False
                            },
                            "description": "Ordered list of cells to add to the new notebook."
                        }
                    },
                    "required": ["cell_sources"],
                    "additionalProperties": False
                }
            }
        }

    def pre_invoke(self, request: ChatRequest, tool_args: dict) -> Union[ToolPreInvokeResponse, None]:
        if self._auto_approve:
            return ToolPreInvokeResponse(f"Calling tool '{self.name}'")
        return ToolPreInvokeResponse(
            f"Calling tool '{self.name}'", "Approve", "Are you sure you want to create a new notebook?"
        )

    async def handle_tool_call(self, request: ChatRequest, response: ChatResponse, tool_context: dict, tool_args: dict) -> str:
        cell_sources = tool_args.get("cell_sources", [])

        # Create an empty notebook (the UI command returns its path)
        ui_cmd_response = await response.run_ui_command(
            "notebook-intelligence:create-new-notebook-from-py",
            {"code": ""}  # empty python file -> empty notebook
        )
        file_path = ui_cmd_response["path"]

        # Append cells in order
        for cell in cell_sources:
            ctype = cell["cell_type"]
            src = cell["source"]
            if ctype == "markdown":
                await response.run_ui_command(
                    "notebook-intelligence:add-markdown-cell-to-notebook",
                    {"markdown": src, "path": file_path}
                )
            else:  # code
                await response.run_ui_command(
                    "notebook-intelligence:add-code-cell-to-notebook",
                    {"code": src, "path": file_path}
                )

        return f"Notebook created successfully at {file_path}"


class AddMarkdownCellToNotebookTool(Tool):
    def __init__(self, auto_approve: bool = False):
        self._auto_approve = auto_approve
        super().__init__()

    @property
    def name(self) -> str:
        return "add_markdown_cell_to_notebook"

    @property
    def title(self) -> str:
        return "Add markdown cell to notebook"

    @property
    def tags(self) -> list[str]:
        return ["senselab-ai"]

    @property
    def description(self) -> str:
        return "Append a markdown cell to an existing notebook."

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "notebook_file_path": {
                            "type": "string",
                            "description": "Absolute or server-root-relative notebook path."
                        },
                        "markdown_cell_source": {
                            "type": "string",
                            "description": "Markdown to append."
                        }
                    },
                    "required": ["notebook_file_path", "markdown_cell_source"],
                    "additionalProperties": False
                }
            }
        }

    def pre_invoke(self, request: ChatRequest, tool_args: dict) -> Union[ToolPreInvokeResponse, None]:
        if self._auto_approve:
            return ToolPreInvokeResponse(f"Calling tool '{self.name}'")
        return ToolPreInvokeResponse(
            f"Calling tool '{self.name}'", "Approve", "Append markdown to the notebook?"
        )

    async def handle_tool_call(self, request: ChatRequest, response: ChatResponse, tool_context: dict, tool_args: dict) -> str:
        notebook_file_path = tool_args["notebook_file_path"]
        server_root_dir = request.host.nbi_config.server_root_dir

        # Normalize to server-root-relative path if needed
        if notebook_file_path.startswith(server_root_dir):
            notebook_file_path = os.path.relpath(notebook_file_path, server_root_dir)

        src = tool_args["markdown_cell_source"]
        await response.run_ui_command(
            "notebook-intelligence:add-markdown-cell-to-notebook",
            {"markdown": src, "path": notebook_file_path}
        )
        return "Added markdown cell to notebook"


class AddCodeCellTool(Tool):
    def __init__(self, auto_approve: bool = False):
        self._auto_approve = auto_approve
        super().__init__()

    @property
    def name(self) -> str:
        return "add_code_cell_to_notebook"

    @property
    def title(self) -> str:
        return "Add code cell to notebook"

    @property
    def tags(self) -> list[str]:
        return ["senselab-ai"]

    @property
    def description(self) -> str:
        return "Append a code cell to an existing notebook."

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "notebook_file_path": {
                            "type": "string",
                            "description": "Absolute or server-root-relative notebook path."
                        },
                        "code_cell_source": {
                            "type": "string",
                            "description": "Code to append."
                        }
                    },
                    "required": ["notebook_file_path", "code_cell_source"],
                    "additionalProperties": False
                }
            }
        }

    def pre_invoke(self, request: ChatRequest, tool_args: dict) -> Union[ToolPreInvokeResponse, None]:
        if self._auto_approve:
            return ToolPreInvokeResponse(f"Calling tool '{self.name}'")
        return ToolPreInvokeResponse(
            f"Calling tool '{self.name}'", "Approve", "Append code to the notebook?"
        )

    async def handle_tool_call(self, request: ChatRequest, response: ChatResponse, tool_context: dict, tool_args: dict) -> str:
        notebook_file_path = tool_args["notebook_file_path"]
        server_root_dir = request.host.nbi_config.server_root_dir

        if notebook_file_path.startswith(server_root_dir):
            notebook_file_path = os.path.relpath(notebook_file_path, server_root_dir)

        src = tool_args["code_cell_source"]
        await response.run_ui_command(
            "notebook-intelligence:add-code-cell-to-notebook",
            {"code": src, "path": notebook_file_path}
        )
        return "Added code cell to notebook"


class RunCellAtIndexTool(Tool):
    def __init__(self, auto_approve: bool = False):
        self._auto_approve = auto_approve
        super().__init__()

    @property
    def name(self) -> str:
        return "run_cell_at_index"

    @property
    def title(self) -> str:
        return "Run cell at index in the active notebook"

    @property
    def tags(self) -> list[str]:
        return ["senselab-ai"]

    @property
    def description(self) -> str:
        return "Execute the cell at the given zero-based index in the ACTIVE notebook."

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell_index": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Zero-based index of the cell to run in the active notebook."
                        }
                    },
                    "required": ["cell_index"],
                    "additionalProperties": False
                }
            }
        }

    def pre_invoke(self, request: ChatRequest, tool_args: dict) -> Union[ToolPreInvokeResponse, None]:
        if self._auto_approve:
            return ToolPreInvokeResponse(f"Calling tool '{self.name}'")
        return ToolPreInvokeResponse(
            f"Calling tool '{self.name}'",
            "Approve",
            f"Run cell at index {tool_args.get('cell_index', '?')} in the active notebook?"
        )

    async def handle_tool_call(self, request: ChatRequest, response: ChatResponse, tool_context: dict, tool_args: dict) -> str:
        cell_index = tool_args["cell_index"]
        if cell_index < 0:
            raise ValueError("cell_index must be >= 0")

        # Runs against the ACTIVE notebook (no path needed)
        await response.run_ui_command(
            "notebook-intelligence:run-cell-at-index",
            {"cellIndex": cell_index}
        )
        return f"Ran the cell at index: {cell_index}"


class RenameNotebookTool(Tool):
    def __init__(self, auto_approve: bool = False):
        self._auto_approve = auto_approve
        super().__init__()

    @property
    def name(self) -> str:
        return "rename_notebook"

    @property
    def title(self) -> str:
        return "Rename the active notebook"

    @property
    def tags(self) -> list[str]:
        return ["senselab-ai"]

    @property
    def description(self) -> str:
        return "Renames the active notebook to the provided name."

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "new_name": {
                            "type": "string",
                            "minLength": 1,
                            "description": "New filename for the notebook (e.g., 'analysis.ipynb')."
                        }
                    },
                    "required": ["new_name"],
                    "additionalProperties": False
                }
            }
        }

    def pre_invoke(self, request: ChatRequest, tool_args: dict):
        if self._auto_approve:
            return ToolPreInvokeResponse(f"Calling tool '{self.name}'")
        return ToolPreInvokeResponse(
            f"Calling tool '{self.name}'",
            "Approve",
            f"Rename the active notebook to '{tool_args.get('new_name', '')}'?"
        )

    async def handle_tool_call(self, request: ChatRequest, response: ChatResponse, tool_context: dict, tool_args: dict) -> str:
        new_name = tool_args["new_name"]
        await response.run_ui_command('notebook-intelligence:rename-notebook', {"newName": new_name})
        return f"Successfully renamed notebook to '{new_name}'."

class GetNumberOfCellsTool(Tool):
    def __init__(self, auto_approve: bool = True):
        self._auto_approve = auto_approve
        super().__init__()

    @property
    def name(self) -> str:
        return "get_number_of_cells"

    @property
    def title(self) -> str:
        return "Get number of cells in the active notebook"

    @property
    def tags(self) -> list[str]:
        return ["senselab-ai"]

    @property
    def description(self) -> str:
        return "Returns the count of cells in the ACTIVE notebook."

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": True,
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False}
            }
        }

    def pre_invoke(self, request: ChatRequest, tool_args: dict):
        return ToolPreInvokeResponse(f"Calling tool '{self.name}'") if self._auto_approve else ToolPreInvokeResponse(
            f"Calling tool '{self.name}'", "Approve", "Get number of cells in the active notebook?"
        )

    async def handle_tool_call(self, request: ChatRequest, response: ChatResponse, tool_context: dict, tool_args: dict) -> str:
        ui_cmd_response = await response.run_ui_command('notebook-intelligence:get-number-of-cells', {})
        return str(ui_cmd_response)


class GetCellTypeAndSourceTool(Tool):
    def __init__(self, auto_approve: bool = True):
        self._auto_approve = auto_approve
        super().__init__()

    @property
    def name(self) -> str:
        return "get_cell_type_and_source"

    @property
    def title(self) -> str:
        return "Get cell type and source at index"

    @property
    def tags(self) -> list[str]:
        return ["senselab-ai"]

    @property
    def description(self) -> str:
        return "Returns the type ('code' or 'markdown') and the source for the given cell index."

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell_index": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Zero-based index of the cell."
                        }
                    },
                    "required": ["cell_index"],
                    "additionalProperties": False
                }
            }
        }

    def pre_invoke(self, request: ChatRequest, tool_args: dict):
        return ToolPreInvokeResponse(f"Calling tool '{self.name}'") if self._auto_approve else ToolPreInvokeResponse(
            f"Calling tool '{self.name}'", "Approve", f"Get type and source for cell {tool_args.get('cell_index', '?')}?"
        )

    async def handle_tool_call(self, request: ChatRequest, response: ChatResponse, tool_context: dict, tool_args: dict) -> str:
        cell_index = tool_args["cell_index"]
        ui_cmd_response = await response.run_ui_command('notebook-intelligence:get-cell-type-and-source', {"cellIndex": cell_index})
        return str(ui_cmd_response)


class GetCellOutputTool(Tool):
    def __init__(self, auto_approve: bool = True):
        self._auto_approve = auto_approve
        super().__init__()

    @property
    def name(self) -> str:
        return "get_cell_output"

    @property
    def title(self) -> str:
        return "Get output of cell at index"

    @property
    def tags(self) -> list[str]:
        return ["senselab-ai"]

    @property
    def description(self) -> str:
        return "Returns the output of the cell at the given index."

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell_index": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Zero-based index of the cell."
                        }
                    },
                    "required": ["cell_index"],
                    "additionalProperties": False
                }
            }
        }

    def pre_invoke(self, request: ChatRequest, tool_args: dict):
        return ToolPreInvokeResponse(f"Calling tool '{self.name}'") if self._auto_approve else ToolPreInvokeResponse(
            f"Calling tool '{self.name}'", "Approve", f"Get output for cell {tool_args.get('cell_index', '?')}?"
        )

    async def handle_tool_call(self, request: ChatRequest, response: ChatResponse, tool_context: dict, tool_args: dict) -> str:
        cell_index = tool_args["cell_index"]
        ui_cmd_response = await response.run_ui_command('notebook-intelligence:get-cell-output', {"cellIndex": cell_index})
        return str(ui_cmd_response)


class SetCellTypeAndSourceTool(Tool):
    def __init__(self, auto_approve: bool = False):
        self._auto_approve = auto_approve
        super().__init__()

    @property
    def name(self) -> str:
        return "set_cell_type_and_source"

    @property
    def title(self) -> str:
        return "Set cell type and source at index"

    @property
    def tags(self) -> list[str]:
        return ["senselab-ai"]

    @property
    def description(self) -> str:
        return "Sets the cell's type ('code' or 'markdown') and its source at the given index."

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell_index": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Zero-based index of the cell."
                        },
                        "cell_type": {
                            "type": "string",
                            "enum": ["code", "markdown"],
                            "description": "Cell type to set."
                        },
                        "source": {
                            "type": "string",
                            "description": "Cell content (Python code or Markdown)."
                        }
                    },
                    "required": ["cell_index", "cell_type", "source"],
                    "additionalProperties": False
                }
            }
        }

    def pre_invoke(self, request: ChatRequest, tool_args: dict):
        if self._auto_approve:
            return ToolPreInvokeResponse(f"Calling tool '{self.name}'")
        return ToolPreInvokeResponse(
            f"Calling tool '{self.name}'",
            "Approve",
            f"Set cell {tool_args.get('cell_index','?')} to type {tool_args.get('cell_type','?')}?"
        )

    async def handle_tool_call(self, request: ChatRequest, response: ChatResponse, tool_context: dict, tool_args: dict) -> str:
        ui_cmd_response = await response.run_ui_command(
            'notebook-intelligence:set-cell-type-and-source',
            {
                "cellIndex": tool_args["cell_index"],
                "cellType": tool_args["cell_type"],
                "source": tool_args["source"]
            }
        )
        return str(ui_cmd_response)


class DeleteCellAtIndexTool(Tool):
    def __init__(self, auto_approve: bool = False):
        self._auto_approve = auto_approve
        super().__init__()

    @property
    def name(self) -> str:
        return "delete_cell_at_index"

    @property
    def title(self) -> str:
        return "Delete cell at index"

    @property
    def tags(self) -> list[str]:
        return ["senselab-ai"]

    @property
    def description(self) -> str:
        return "Deletes the cell at the given index in the ACTIVE notebook."

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell_index": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Zero-based index of the cell to delete."
                        }
                    },
                    "required": ["cell_index"],
                    "additionalProperties": False
                }
            }
        }

    def pre_invoke(self, request: ChatRequest, tool_args: dict):
        if self._auto_approve:
            return ToolPreInvokeResponse(f"Calling tool '{self.name}'")
        return ToolPreInvokeResponse(
            f"Calling tool '{self.name}'",
            "Approve",
            f"Delete the cell at index {tool_args.get('cell_index', '?')}?"
        )

    async def handle_tool_call(self, request: ChatRequest, response: ChatResponse, tool_context: dict, tool_args: dict) -> str:
        cell_index = tool_args["cell_index"]
        await response.run_ui_command('notebook-intelligence:delete-cell-at-index', {"cellIndex": cell_index})
        return f"Deleted the cell at index: {cell_index}"


class InsertCellAtIndexTool(Tool):
    def __init__(self, auto_approve: bool = False):
        self._auto_approve = auto_approve
        super().__init__()

    @property
    def name(self) -> str:
        return "insert_cell_at_index"

    @property
    def title(self) -> str:
        return "Insert a cell at index"

    @property
    def tags(self) -> list[str]:
        return ["senselab-ai"]

    @property
    def description(self) -> str:
        return "Inserts a new cell (code or markdown) with given source at the index."

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell_index": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Zero-based index where the new cell will be inserted."
                        },
                        "cell_type": {
                            "type": "string",
                            "enum": ["code", "markdown"],
                            "description": "Type of the new cell."
                        },
                        "source": {
                            "type": "string",
                            "description": "Content for the new cell."
                        }
                    },
                    "required": ["cell_index", "cell_type", "source"],
                    "additionalProperties": False
                }
            }
        }

    def pre_invoke(self, request: ChatRequest, tool_args: dict):
        if self._auto_approve:
            return ToolPreInvokeResponse(f"Calling tool '{self.name}'")
        return ToolPreInvokeResponse(
            f"Calling tool '{self.name}'",
            "Approve",
            f"Insert a {tool_args.get('cell_type','?')} cell at index {tool_args.get('cell_index','?')}?"
        )

    async def handle_tool_call(self, request: ChatRequest, response: ChatResponse, tool_context: dict, tool_args: dict) -> str:
        ui_cmd_response = await response.run_ui_command(
            'notebook-intelligence:insert-cell-at-index',
            {
                "cellIndex": tool_args["cell_index"],
                "cellType": tool_args["cell_type"],
                "source": tool_args["source"]
            }
        )
        return str(ui_cmd_response)


class SaveNotebookTool(Tool):
    def __init__(self, auto_approve: bool = True):
        self._auto_approve = auto_approve
        super().__init__()

    @property
    def name(self) -> str:
        return "save_notebook"

    @property
    def title(self) -> str:
        return "Save the active notebook to disk"

    @property
    def tags(self) -> list[str]:
        return ["senselab-ai"]

    @property
    def description(self) -> str:
        return "Saves changes in the ACTIVE notebook."

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": True,
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False}
            }
        }

    def pre_invoke(self, request: ChatRequest, tool_args: dict):
        return ToolPreInvokeResponse(f"Calling tool '{self.name}'") if self._auto_approve else ToolPreInvokeResponse(
            f"Calling tool '{self.name}'", "Approve", "Save the active notebook?"
        )

    async def handle_tool_call(self, request: ChatRequest, response: ChatResponse, tool_context: dict, tool_args: dict) -> str:
        await response.run_ui_command('docmanager:save')
        return "Saved the notebook"


class CreateNewPythonFileTool(Tool):
    def __init__(self, auto_approve: bool = False):
        self._auto_approve = auto_approve
        super().__init__()

    @property
    def name(self) -> str:
        return "create_new_python_file"

    @property
    def title(self) -> str:
        return "Create a new Python (.py) file"

    @property
    def tags(self) -> list[str]:
        return ["senselab-ai"]

    @property
    def description(self) -> str:
        return "Creates a new Python file with the provided code, returning the file path."

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python source code for the new file."
                        }
                    },
                    "required": ["code"],
                    "additionalProperties": False
                }
            }
        }

    def pre_invoke(self, request: ChatRequest, tool_args: dict):
        if self._auto_approve:
            return ToolPreInvokeResponse(f"Calling tool '{self.name}'")
        return ToolPreInvokeResponse(
            f"Calling tool '{self.name}'", "Approve", "Create a new Python file with the provided code?"
        )

    async def handle_tool_call(self, request: ChatRequest, response: ChatResponse, tool_context: dict, tool_args: dict) -> str:
        ui_cmd_response = await response.run_ui_command('notebook-intelligence:create-new-file', {'code': tool_args["code"]})
        file_path = ui_cmd_response.get('path', '')
        return f"Created new Python file at {file_path}"


class GetCurrentFileContentTool(Tool):
    def __init__(self, auto_approve: bool = True):
        self._auto_approve = auto_approve
        super().__init__()

    @property
    def name(self) -> str:
        return "get_current_file_content"

    @property
    def title(self) -> str:
        return "Get content of the current file"

    @property
    def tags(self) -> list[str]:
        return ["senselab-ai"]

    @property
    def description(self) -> str:
        return "Returns the content of the currently active file (notebook or text)."

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": True,
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False}
            }
        }

    def pre_invoke(self, request: ChatRequest, tool_args: dict):
        return ToolPreInvokeResponse(f"Calling tool '{self.name}'") if self._auto_approve else ToolPreInvokeResponse(
            f"Calling tool '{self.name}'", "Approve", "Get content of the current file?"
        )

    async def handle_tool_call(self, request: ChatRequest, response: ChatResponse, tool_context: dict, tool_args: dict) -> str:
        ui_cmd_response = await response.run_ui_command('notebook-intelligence:get-current-file-content', {})
        return str(ui_cmd_response)


class SetCurrentFileContentTool(Tool):
    def __init__(self, auto_approve: bool = False):
        self._auto_approve = auto_approve
        super().__init__()

    @property
    def name(self) -> str:
        return "set_current_file_content"

    @property
    def title(self) -> str:
        return "Set content of the current file"

    @property
    def tags(self) -> list[str]:
        return ["senselab-ai"]

    @property
    def description(self) -> str:
        return "Sets the content of the currently active file (notebook or text)."

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "New content to write into the current file."
                        }
                    },
                    "required": ["content"],
                    "additionalProperties": False
                }
            }
        }

    def pre_invoke(self, request: ChatRequest, tool_args: dict):
        if self._auto_approve:
            return ToolPreInvokeResponse(f"Calling tool '{self.name}'")
        return ToolPreInvokeResponse(
            f"Calling tool '{self.name}'", "Approve", "Overwrite the current file's content?"
        )

    async def handle_tool_call(self, request: ChatRequest, response: ChatResponse, tool_context: dict, tool_args: dict) -> str:
        await response.run_ui_command('notebook-intelligence:set-current-file-content', {"content": tool_args["content"]})
        return "Set the file content"


class AIAgentChatParticipant(MCPChatParticipant):
    PREPROMPT = """You are senselab AI. Solve every user request using the senselab Python package (assume it is installed and importable).
    Before writing ANY import, function call, or data structure, first consult the Deepwiki MCP server (sensein/senselab, server name: `deepwiki`) to confirm the correct senselab utilities/classes/modules to use. Do not invent functions or APIs. If anything is unclear or ambiguous, STOP and query Deepwiki before coding.

    Follow an iterative loop: Think → Act → Observe, repeating as needed.
    1) Plan: briefly state your step-by-step plan for solving the task with senselab.
    2) Act (discovery): call Deepwiki tools to confirm the exact senselab symbols (module paths, class/function names, expected arguments, return types) and relevant usage notes.
    3) Observe: summarize what you learned from Deepwiki and update your plan if needed.
    4) Plan II: refine your plan based on the new knowledge.
    5) Coding: write code one Jupyter cell at a time. Each cell must be self-contained and minimal (anatomic), with only what is necessary for that step.
    6) Execute: Run immediately after writing each cell. 
    7) If it fails, debug IN PLACE until it runs successfully. Only then proceed to the next step. If user input is needed (e.g., paths, parameters, assumptions), explicitly ask the user before continuing.

    While acting, always name which Deepwiki tool you used (e.g., ask_question, read_wiki_structure, read_wiki_contents) and what you asked. Keep responses concise and action-oriented."""

    POSTPROMPT = """Output format and discipline:
    - Start with **Plan** (numbered steps).
    - For each discovery step, show **Deepwiki query** (tool + question) and **Observation** (what you learned).
    - For each coding step, output exactly one Jupyter cell, then immediately run it; if an error occurs, show the error briefly and fix the SAME cell until it succeeds before moving on.
    - Prefer runnable, minimal snippets over long narratives.
    - Never fabricate senselab APIs; if uncertain, consult Deepwiki again.

    Finalization:
    - Conclude with a **Methods** and **Results** section in the Notebook, as in a scientific report (concise, reproducible).
    - In **Methods**, list senselab components used, key parameters, data handling, and the sequence of steps.
    - In **Results**, report outputs/metrics/artifacts produced and any validations performed.
    - If there are limitations or open questions, include a brief **Discussion/Next steps**.

    Remember: senselab-first; Deepwiki for confirmation before coding; iterate with think→act→observe; one self-contained cell at a time; run and fix before proceeding."""

    def __init__(self, host: Host, servers: list[MCPServer]):
        self.host = host
        super().__init__(
            id="senselab-ai",
            name="senselab AI",
            servers=servers,
            nbi_tools=[]
        )

    @property
    def id(self) -> str:
        return "senselab-ai"

    @property
    def name(self) -> str:
        return "senselab AI"
    
    @property
    def description(self) -> str:
        return "The senselab AI that assists with using senselab for audio, video, and text analysis."

    @property
    def icon_path(self) -> str:
        return PARTICIPANT_ICON_URL

    @property
    def commands(self) -> list[ChatCommand]:
        return [
            *super().commands,  # keep anything parent exposes
            ChatCommand(name="help", description="Show help"),
        ]
    
    @property
    def tools(self) -> list[Tool]:
        return [*super().tools,  # keep anything parent exposes
                CreateNewNotebookTool(auto_approve=False),
                AddMarkdownCellToNotebookTool(auto_approve=False),
                AddCodeCellTool(auto_approve=False),
                RunCellAtIndexTool(auto_approve=False),
                RenameNotebookTool(auto_approve=False),
                GetNumberOfCellsTool(auto_approve=True),
                GetCellTypeAndSourceTool(auto_approve=True),
                GetCellOutputTool(auto_approve=True),
                SetCellTypeAndSourceTool(auto_approve=False),
                DeleteCellAtIndexTool(auto_approve=False),
                InsertCellAtIndexTool(auto_approve=False),
                SaveNotebookTool(auto_approve=False),
                CreateNewPythonFileTool(auto_approve=False),
                GetCurrentFileContentTool(auto_approve=True),
                SetCurrentFileContentTool(auto_approve=False),
                NotebookShareTool()]

    async def handle_chat_request(self, request: ChatRequest, response: ChatResponse, options: dict = {}) -> None:
        if request.command == 'help':
            response.stream(
                MarkdownData(
                    """I am the senselab AI. I can help you with audio, text, and video analysis.
                    Here are some example prompts you can try:\n
                    \n```text\n@senselab-ai Read the audio files in '<directory_path>', make them mono, resample them to 16KHz, and extract audio features.\n```\n
                    \n```text\n@senselab-ai Read the audio files in '<directory_path>', make them mono, resample them to 16KHz, and transcribe them with word-level timestamps.\n```\n
                    \n```text\n@senselab-ai Share this notebook publicly.\n```\n


                    """)
                )
            response.finish()
            return

        system_prompt = """
        Try to answer the questions with `senselab` Python package (you can assume it's installed). 
        You **MUST** query the Deepwiki server (`deepwiki`) to verify every class, function, or module **BEFORE** writing any code that uses it (repo: sensein/senselab).
        """

        options = options.copy()
        options["system_prompt"] = system_prompt

        await self.handle_chat_request_with_tools(request, response, options)


class AIAgentExtension(NotebookIntelligenceExtension):
    @property
    def id(self) -> str:
        return "senselab-agentic-interface-extension"

    @property
    def name(self) -> str:
        return "senselab AI Extension"

    @property
    def provider(self) -> str:
        return "sensein/senselab"

    @property
    def url(self) -> str:
        return "https://github.com/sensein/senselab"

    def _make_deepwiki_server(self) -> MCPServer:
        return MCPServerImpl(
            name="deepwiki",
            sse_params=SSEServerParameters(url="https://mcp.deepwiki.com/sse"),
            auto_approve_tools=[],
        )

    def _init_tool_lists_bg(self, servers: list[MCPServer]) -> None:
        async def _go():
            for s in servers:
                try:
                    await s.update_tool_list()
                    # Optional: log discovered tools for quick inspection
                    try:
                        tools = s.get_tools()
                        names = [t.name for t in tools]
                        log.info(f"[MCP:{s.name}] tools discovered: {names}")
                    except Exception as e:
                        log.warning(f"[MCP:{s.name}] failed listing tools locally: {e}")
                except Exception as e:
                    log.error(f"Error initializing tool list for server {s.name}: {e}")

        threading.Thread(target=lambda: asyncio.run(_go()), daemon=True).start()

    def activate(self, host: Host) -> None:
        # Build the single Deepwiki server in code
        deepwiki = self._make_deepwiki_server()
        servers = [deepwiki]

        # Create + register your participant
        participant = AIAgentChatParticipant(host, servers=servers)
        host.register_chat_participant(participant)
        self.participant = participant

        # Populate MCP tool lists in the background so tools appear promptly
        self._init_tool_lists_bg(servers)

        log.info("senselab AI Extension activated.")