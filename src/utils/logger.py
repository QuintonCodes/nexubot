import logging
import os
import sys


class UnicodeStreamHandler(logging.StreamHandler):
    """
    Custom StreamHandler to handle Unicode characters on Windows consoles
    """

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream

            if hasattr(stream, "buffer"):
                stream.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
                stream.buffer.flush()
            else:
                stream.write(msg + "\n")
                stream.flush()
        except Exception:
            self.handleError(record)


def setup_logging():
    """
    Initialize the logging configuration.
    Logs to file and console with specific formatting.
    """
    # Create a root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%d %b %H:%M:%S")

    # File Handler
    file_handler = logging.FileHandler("nexubot.log", encoding="utf-8", mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler (Custom)
    console_handler = UnicodeStreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("aiohttp").setLevel(logging.ERROR)
    logging.getLogger("asyncio").setLevel(logging.ERROR)
    logging.getLogger("sqlalchemy").setLevel(logging.ERROR)
    logging.getLogger("geventwebsocket.handler").setLevel(logging.ERROR)

    return logger


def read_recent_logs(limit=20):
    """Reads the last N lines from the log file."""
    log_file = "nexubot.log"
    if not os.path.exists(log_file):
        return []

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            # simple filter to remove clutter
            filtered = [l.strip() for l in lines if "geventwebsocket" not in l and "aiohttp" not in l]
            return filtered[-limit:]
    except Exception:
        return []


# Global accessor
logger = logging.getLogger(__name__)
