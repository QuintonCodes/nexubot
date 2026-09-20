"""
Structured Logging Configuration.
Outputs logs in JSON format for easy filtering and searching in Railway.
"""

import logging
import sys
import structlog


def setup_logger():
    # Disable default logging to avoid duplicate output
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger("nexubot")


# Export singleton logger
logger = setup_logger()
