import asyncio
import threading
import logging

logger = logging.getLogger(__name__)

_background_loop = None
_loop_thread = None


def _start_background_loop(loop):
    """Runs the asyncio loop forever in a separate thread."""
    asyncio.set_event_loop(loop)
    loop.run_forever()


def get_persistent_loop():
    """Ensures a background loop is running and returns it."""
    global _background_loop, _loop_thread
    if _background_loop is None:
        _background_loop = asyncio.new_event_loop()
        _loop_thread = threading.Thread(target=_start_background_loop, args=(_background_loop,), daemon=True)
        _loop_thread.start()
        logger.info("⚡ Background AsyncIO Loop Started")
    return _background_loop


def safe_get_result(future, timeout=3.0):
    try:
        return future.result(timeout=timeout)
    except (asyncio.TimeoutError, TimeoutError):
        return None
    except Exception as e:
        logger.error(f"Async Task Error: {e}")
        return None
