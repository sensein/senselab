# ruff: noqa: D100 - the module docstring is generated
from senselab.reference._code_pages import datasets_api_section
from senselab.reference._render import datasets_page, or_unavailable

__doc__ = or_unavailable(lambda: f"{datasets_page()}\n{datasets_api_section()}")
