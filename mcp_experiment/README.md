JUPYTER_TOKEN=$(python -c 'import secrets;print(secrets.token_urlsafe(24))')

echo $JUPYTER_TOKEN   
MbeZwzyiprj29m6aAHUB2xWS3o0DpFWM


poetry run jupyter lab \
  --IdentityProvider.token="$JUPYTER_TOKEN" \
  --ServerApp.allow_origin='*'

poetry run jupyter lab --port 8888 --IdentityProvider.token JUPYTER_TOKEN --ip 0.0.0.0





JUPYTER_TOKEN=<....>
echo $JUPYTER_TOKEN                  
QLg8vUr4S5LmPGtP3N8S1sMKxmCoR48O

poetry run jupyter-mcp-server start \
  --transport streamable-http \
  --port 4040 \
  --runtime-url http://127.0.0.1:8888 \
  --room-url    http://127.0.0.1:8888 \
  --runtime-token "$JUPYTER_TOKEN" \
  --room-token    "$JUPYTER_TOKEN" \
  --start-new-runtime true




Download claude desktop:
https://claude.ai/download


Change Claude config:
Open Claude Desktop → Settings → Developer → Edit Config. This opens the config at:
macOS: ~/Library/Application Support/Claude/claude_desktop_config.

json
{
  "mcpServers": {
    "jupyter": {
      "command": "npx",
      "args": ["mcp-remote", "http://127.0.0.1:4040/mcp"]
    }
  }
}


poetry run python -m ipykernel install --user --name senselab



to enable Jupyter’s collaboration API:
# pin the version MCP 0.6 expects
poetry add --group mcp jupyter-ydoc==3.0.5

# add the rest; let Poetry solve around ydoc 3.0.5
poetry add --group mcp jupyter-server-fileid ypy-websocket "jupyter-collaboration<4.1"








4) Quick Makefile (optional, so it’s one command next time)

Create Makefile in your repo:

.PHONY: jupyterlab start-streamable-http

jupyterlab:
	@echo "Starting JupyterLab…"
	poetry run jupyter lab --IdentityProvider.token="$(JUPYTER_TOKEN)" --ServerApp.allow_origin='*'

start-streamable-http:
	@echo "Starting Jupyter MCP Server (streamable-http)…"
	@if [ -z "$$JUPYTER_TOKEN" ]; then echo "Set JUPYTER_TOKEN first"; exit 1; fi
	RUNTIME_TOKEN="$$JUPYTER_TOKEN" ROOM_TOKEN="$$JUPYTER_TOKEN" \
	poetry run jupyter-mcp-server start \
	  --transport streamable-http \
	  --port 4040 \
	  --runtime-url http://127.0.0.1:8888 \
	  --room-url    http://127.0.0.1:8888 \
	  --start-new-runtime true


Then:

export JUPYTER_TOKEN=QLg8vUr4S5LmPGtP3N8S1sMKxmCoR48O
make jupyterlab
# new terminal
make start-streamable-http







WITH COLAB:

COLAB + senselab + Jupyter Notebook

# it was critical running the jupyter notebook with 
/Users/fabiocat/miniconda3/envs/colab/bin/jupyter server \
  --ServerApp.allow_origin='https://colab.research.google.com' \
  --port=8888 --ServerApp.port_retries=0

from the same environment as senselab and also make sure the kernel used is the one with senselab installed!!!
xx
TEST IT AGAIN UPLOADING THE TUTORIAL NOTEBOOK AS A REFERENCE!!!!!!!!

