# Agentic Interface for senselab

<button class="tutorial-button" onclick="window.location.href='https://github.com/sensein/senselab/blob/main/tutorials/senselab-ai/senselab_ai_intro.ipynb'">Getting started</button>

The `agentic_interface` submodule provides a flexible, user-friendly bridge between natural language and code, enabling users to interact with senselab's AI agent for scientific coding, data analysis, and reproducible research—even without strong technical skills.

![alt text](<https://github.com/sensein/senselab/blob/main/tutorials/senselab-ai/resources/Screenshot 2025-09-02 at 8.52.07 PM.png>)

---

### Motivation

The goal is to **lower the entry barrier for non-programmers** while keeping workflows transparent and reproducible for technical users. Many potential users of senseLab (e.g., clinicians, clinical researchers, students from non-technical domains) could benefit from audio data analysis but lack the Python expertise required today.

`agentic_interface` aims to:

* Enable **interactive exploration and analysis** of audio datasets through natural-language interaction.
* **Generate reproducible code** alongside the analysis, so that results can be shared, rerun, and extended.
* Integrate seamlessly with existing senselab functionality without adding setup complexity.

In short: make audio analysis in senseLab **as simple as asking questions**, while still producing transparent, reproducible workflows.

---

### Related work & alternative tools we evaluated

We reviewed several existing tools that inspired this direction but also highlighted limitations that `agentic_interface` aims to overcome:

#### 1. **Jupyter AI**

* Reference: [Jupyter AI](https://jupyter-ai.readthedocs.io/en/latest/index.html)
* Strengths: Side-chat within Jupyter notebooks for natural-language queries.
* Limitation: Does **not yet support MCP server integration**, which is key to exposing senseLab features as tools and enabling autonomous analysis.

#### 2. **PretzelAI**

* Reference: [PretzelAI GitHub](https://github.com/pretzelai/pretzelai?tab=readme-ov-file)
* Strengths: Chat interface, inline prompting, polished UI.
* Limitation: Also lacks **MCP server integration**, making full tool-driven workflows harder.

#### 3. **Toolchain Integration via MCP**

* Workflow idea: Run [jupyter-mcp-server](https://jupyter-mcp-server.datalayer.tech/jupyter/) to control a notebook, connect MCP clients like Claude Desktop, Cline, or [LM Studio](https://lmstudio.ai/), and request analyses.
* Tested: Works in principle.
* Limitations:

  * Requires installing & launching multiple tools (high setup overhead for non-technical users).
  * Without a dedicated MCP server exposing senselab tools/docs, the LLM often gets confused about available functions.
  * Relies on external LLM usage credits, which can be restrictive.

#### 4. **Notebook Intelligence**

* Reference: [notebook-intelligence](https://github.com/notebook-intelligence/notebook-intelligence)
* Strengths: Easy install, integrates chat and inline commands into JupyterLab, supports many LLMs (including local models like Ollama), and supports MCP servers (remote only for now).
* Limitations:
  * Cannot enforce ReAct behaviors like consulting senseLab docs first, executing cells step-by-step, or test-driven coding ([issue #63](https://github.com/notebook-intelligence/notebook-intelligence/issues/63#issuecomment-3245071828)).
* This is the one we currently use under the hood for our implementation.
