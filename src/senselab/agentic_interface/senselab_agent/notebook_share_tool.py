"""This module provides a tool for sharing Jupyter notebooks.

The code has been adapted from the original implementation by Mehmet Bektas <mbektasgh@outlook.com>.
See here: https://github.com/notebook-intelligence/nbi-ai-agent-example/blob/main/nbi_ai_agent_example/notebook_share_tool.py
"""

from os import path
from typing import Union

import nbss_upload
from notebook_intelligence import AnchorData, ChatRequest, ChatResponse, Tool, ToolPreInvokeResponse


class NotebookShareTool(Tool):
    """A tool for sharing Jupyter notebooks."""

    @property
    def name(self) -> str:
        """Name of the tool.

        Return:
            str: The name of the tool.
        """
        return "share_notebook"

    @property
    def title(self) -> str:
        """Title of the tool.

        Return:
            str: The title of the tool.
        """
        return "Share notebook publicly"

    @property
    def tags(self) -> list[str]:
        """Return a list of tags associated with the tool.

        Return:
            list[str]: A list of tags associated with the tool.
        """
        return ["senselab-ai-example-tool"]

    @property
    def description(self) -> str:
        """Return a description of the tool's purpose.

        Return:
            str: A description of the tool's purpose.
        """
        return "This is a tool that shares a notebook publicly and creates a link"

    @property
    def schema(self) -> dict:
        """Return the JSON schema for the tool's arguments.

        Return:
            dict: The JSON schema for the tool's arguments.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "notebook_file_path": {
                            "type": "string",
                            "description": "File path of the notebook to share",
                        }
                    },
                    "required": ["notebook_file_path"],
                    "additionalProperties": False,
                },
            },
        }

    def pre_invoke(self, request: ChatRequest, tool_args: dict) -> Union[ToolPreInvokeResponse, None]:
        """Prepare to share a notebook by confirming the action.

        Args:
            request (ChatRequest): The chat request object.
            tool_args (dict): The arguments for the tool invocation.

        Return:
            ToolPreInvokeResponse: The response object for the pre-invocation step.
        """
        file_path = tool_args.get("notebook_file_path")
        assert isinstance(file_path, str)
        file_name = path.basename(file_path)

        return ToolPreInvokeResponse(
            message=f"Sharing notebook '{file_name}'",
            confirmationTitle="Confirm sharing",
            confirmationMessage=(
                f"Are you sure you want to share the notebook at '{file_path}'? "
                "This will upload the notebook to the public internet and cannot be undone."
            ),
        )

    async def handle_tool_call(
        self, request: ChatRequest, response: ChatResponse, tool_context: dict, tool_args: dict
    ) -> dict:
        """Handle the tool call to share a notebook.

        Args:
            request (ChatRequest): The chat request object.
            response (ChatResponse): The chat response object.
            tool_context (dict): The context for the tool invocation.
            tool_args (dict): The arguments for the tool invocation.

        Return:
            dict: The result of the tool invocation.
        """
        file_path = tool_args.get("notebook_file_path")
        assert isinstance(file_path, str)
        file_name = path.basename(file_path)

        if not path.isfile(file_path):
            return {
                "error": f"Notebook file '{file_path}' was not found.",
                "confirmationTitle": "File not found",
            }

        try:
            share_url = nbss_upload.upload_notebook(file_path, False, False, "https://notebooksharing.space")
            response.stream(AnchorData(share_url, f"Click here to view the shared notebook '{file_name}'"))
            return {"result": f"Notebook '{file_name}' has been shared."}
        except Exception as e:
            err_msg = f"Failed to share notebook '{file_name}': {str(e)}"
            response.stream(err_msg)
            return {"error": err_msg}
