import logging
import os
from logging.handlers import TimedRotatingFileHandler

# Standard log format for all processes
LOG_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def setup_process_logging(logger_name, log_dir="logs", log_level=logging.INFO):
    """
    Setup logging for a process (CameraAgent, ResultWorker, etc.)
    
    Args:
        logger_name: Logger identifier (e.g., "Agent-CAM01", "ResultWorker")
        log_dir: Directory to store log files
        log_level: Logging level (default INFO)
    
    Returns:
        logger: Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
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
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def setup_root_logging(log_dir="logs", log_level=logging.INFO):
    """
    Setup root logging (for main supervisor process)
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level (default INFO)
    """
    os.makedirs(log_dir, exist_ok=True)
    
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
    
    logging.basicConfig(
        level=log_level,
        handlers=[file_handler, console_handler]
    )
