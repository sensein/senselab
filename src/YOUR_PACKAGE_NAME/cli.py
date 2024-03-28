"""This module defines CLI commands for the PipePal application."""

import click

from .app import hello_world as hw_function  # Renamed to avoid conflict


@click.group()
def main() -> None:
    """Define the main CLI group."""
    pass


@main.command()
def hello_world() -> None:
    """Execute the hello_world command from the app module."""
    hw_function()
