import logging
import os
import multiprocessing as mp
from logging.handlers import TimedRotatingFileHandler, QueueHandler, QueueListener

# Standard log format for all processes
LOG_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Global log queue for multiprocessing (created in main process)
_log_queue = None
_queue_listener = None


def setup_process_logging(logger_name, log_queue=None, log_level=logging.INFO):
    """
    Setup logging for a process (CameraAgent, ResultWorker, etc.)
    Uses QueueHandler to send logs to parent process to avoid file locking issues.
    
    Args:
        logger_name: Logger identifier (e.g., "Agent-CAM01", "ResultWorker")
        log_queue: Queue to send logs to (shared with main process)
        log_level: Logging level (default INFO)
    
    Returns:
        logger: Configured logger instance
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Use QueueHandler to send logs to main process
    if log_queue is None:
        log_queue = _log_queue

    if log_queue is None:
        # Fallback to a direct StreamHandler if queue isn't available
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(ch)
    else:
        queue_handler = QueueHandler(log_queue)
        # Don't format here—let QueueListener handle formatting
        logger.addHandler(queue_handler)
    
    return logger


def setup_root_logging(log_dir="logs", log_level=logging.INFO):
    """
    Setup root logging with QueueListener to handle logs from child processes.
    This should be called ONCE in the main process.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level (default INFO)
    
    Returns:
        listener: QueueListener instance that must be started
    """
    global _queue_listener, _log_queue

    os.makedirs(log_dir, exist_ok=True)

    # Ensure a multiprocessing Queue is created in the main process
    if _log_queue is None:
        _log_queue = mp.Queue()

    # File handler with daily rotation
    log_path = os.path.join(log_dir, "system.log")
    file_handler = TimedRotatingFileHandler(
        log_path,
        when="midnight",           # Rotate at midnight
        interval=1,               # Every 1 day
        backupCount=30,           # Keep 30 days of logs
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    # Create QueueListener to process logs from child processes
    _queue_listener = QueueListener(
        _log_queue,
        file_handler,
        console_handler,
        respect_handler_level=True
    )

    # Also setup root logger for main process logs
    logging.basicConfig(
        level=log_level,
        handlers=[file_handler, console_handler]
    )

    return _queue_listener, _log_queue

