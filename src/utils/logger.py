import logging
import sys

class UnicodeStreamHandler(logging.StreamHandler):
    """
    Custom StreamHandler to handle Unicode characters on Windows consoles
    """
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream

            # Write message with explicit newline
            # Use 'replace' to handle potential encoding errors gracefully
            if hasattr(stream, 'buffer'):
                stream.buffer.write((msg + '\n').encode('utf-8', errors='replace'))
                stream.buffer.flush()
            else:
                stream.write(msg + '\n')
                stream.flush()
        except Exception:
            self.handleError(record)

def setup_logging():
    """
    Initialize the logging configuration for the application
    """
    # Create a root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File Handler
    file_handler = logging.FileHandler('nexubot.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler (Custom)
    console_handler = UnicodeStreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Reduce noise from third-party libraries
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)

def get_logger(name):
    return logging.getLogger(name)