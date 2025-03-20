import logging

from rich.logging import RichHandler

__all__ = ["logger"]

# Create a logger
logger = logging.getLogger("starccato_jax")
logger.setLevel(
    logging.DEBUG
)  # Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

# Remove default handlers (if any)
if logger.hasHandlers():
    logger.handlers.clear()

# Setup RichHandler for pretty logs
rich_handler = RichHandler(
    rich_tracebacks=True, show_time=True, show_level=True
)
formatter = logging.Formatter(
    "%(message)s"
)  # Simple formatting (Rich handles colors)
rich_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(rich_handler)
