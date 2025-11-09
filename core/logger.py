"""
Logger setup module for the adaptive trading bot.

This module provides functionality to set up logging with both console and file output.
"""
import logging
import logging.handlers
from pathlib import Path
from typing import Union


def setup_logger(log_dir: Union[str, Path], level: str = "INFO") -> logging.Logger:
    """
    Set up a logger with both console and file handlers.
    
    Args:
        log_dir: Directory where log files will be stored. Will be created if it doesn't exist.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Defaults to "INFO".
    
    Returns:
        Configured logger instance.
    
    Raises:
        ValueError: If the log level is invalid.
        OSError: If the log directory cannot be created.
    """
    # Convert to Path object and create directory if it doesn't exist
    log_path = Path(log_dir)
    try:
        log_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Cannot create log directory {log_path.absolute()}: {e}")
    
    # Validate log level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    # Get or create logger
    logger = logging.getLogger("bot")
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Set the logging level
    logger.setLevel(numeric_level)
    
    # Create formatter
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = log_path / "bot.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False
    
    return logger