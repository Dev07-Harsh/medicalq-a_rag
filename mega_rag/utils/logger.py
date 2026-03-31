"""
MEGA-RAG Logger

Centralized logging for all components. Writes to both console and file.
Log files are saved in logs/ directory with timestamps.

Usage:
    from mega_rag.utils.logger import get_logger
    logger = get_logger("llm")
    logger.info("Groq call: 150 tokens")
    logger.warning("Rate limit hit, waiting 5s")
"""
import logging
import os
from datetime import datetime
from pathlib import Path


_LOGGERS = {}
_LOG_DIR = None
_LOG_FILE = None


def setup_log_dir(base_dir: str = None) -> Path:
    """Create logs directory and return path."""
    global _LOG_DIR, _LOG_FILE
    if _LOG_DIR:
        return _LOG_DIR

    base = Path(base_dir) if base_dir else Path(os.getcwd())
    _LOG_DIR = base / "logs"
    _LOG_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _LOG_FILE = _LOG_DIR / f"mega_rag_{timestamp}.log"
    return _LOG_DIR


def get_logger(name: str, level: str = None) -> logging.Logger:
    """Get or create a named logger that writes to console + file.

    Args:
        name: Logger name (e.g., "llm", "retrieval", "eval")
        level: Override log level. Default uses LOG_LEVEL env var or INFO.
    """
    global _LOGGERS

    full_name = f"mega_rag.{name}"
    if full_name in _LOGGERS:
        return _LOGGERS[full_name]

    # Determine level
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, level, logging.INFO)

    logger = logging.getLogger(full_name)
    logger.setLevel(log_level)
    logger.propagate = False

    # Don't add handlers if they already exist
    if logger.handlers:
        _LOGGERS[full_name] = logger
        return logger

    # Console handler (concise)
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console_fmt = logging.Formatter("%(message)s")
    console.setFormatter(console_fmt)
    logger.addHandler(console)

    # File handler (detailed)
    setup_log_dir()
    if _LOG_FILE:
        file_handler = logging.FileHandler(_LOG_FILE, mode="a")
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_fmt = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

    _LOGGERS[full_name] = logger
    return logger
