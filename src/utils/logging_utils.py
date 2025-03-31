"""
Logging and status display utilities.
"""

import logging
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text
from typing import Optional

# Initialize the rich console
console = Console()

def setup_logging(log_level=logging.INFO):
    """
    Set up logging with rich formatting.
    
    Args:
        log_level: The log level to use (default: INFO)
    
    Returns:
        logger: The configured logger instance
    """
    # Configure the rich handler
    rich_handler = RichHandler(
        rich_tracebacks=True,
        console=console,
        show_time=False
    )
    
    # Configure the root logger
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[rich_handler]
    )
    
    # Get the logger
    logger = logging.getLogger("model_evaluator")
    logger.setLevel(log_level)
    
    return logger

def print_status(status_type, message, color=None):
    """
    Print a formatted status message.
    
    Args:
        status_type (str): Type of status (info, success, error, warning)
        message (str): The message to display
        color (str, optional): Color to use for the message
    """
    # Map status types to symbols and default colors
    status_map = {
        "info": ("ℹ️", "blue"),
        "success": ("✅", "green"),
        "error": ("❌", "red"),
        "warning": ("⚠️", "yellow"),
        "progress": ("⏳", "cyan"),
        "debug": ("🔍", "magenta")
    }
    
    # Get the symbol and default color
    symbol, default_color = status_map.get(status_type.lower(), ("➔", "white"))
    
    # Use the provided color or the default one
    text_color = color or default_color
    
    # Format and print the message
    formatted_text = Text(f"{symbol} {message}")
    formatted_text.stylize(text_color)
    
    console.print(formatted_text)

def print_section_header(title, color="blue"):
    """
    Print a formatted section header.
    
    Args:
        title (str): The section title
        color (str): Color to use for the header
    """
    # Create a panel with the title
    panel = Panel(
        Text(title, style=f"bold {color}"),
        border_style=color,
        expand=False
    )
    
    # Print the panel with some spacing
    console.print()
    console.print(panel)
    console.print()

def format_time(seconds):
    """
    Format time in seconds to a human-readable string.
    
    Args:
        seconds (float): Time in seconds
    
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.2f}s"
    else:
        hours = int(seconds // 3600)
        remaining = seconds % 3600
        minutes = int(remaining // 60)
        remaining_seconds = remaining % 60
        return f"{hours}h {minutes}m {remaining_seconds:.2f}s" 