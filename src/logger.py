"""Logging configuration for the Korean Q&A evaluation system."""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
import sys
from typing import Optional

class LoggerSetup:
    """Setup and configure logging for the application."""
    
    def __init__(self, 
                 log_dir: str = "logs",
                 log_level: str = "INFO",
                 max_bytes: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5):
        self.log_dir = Path(log_dir)
        self.log_level = getattr(logging, log_level.upper())
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        
        # Create logs directory if it doesn't exist
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Configure logging with file and console handlers."""
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # File handler for all logs
        all_logs_file = self.log_dir / f"evaluation_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            all_logs_file,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
        
        # Error file handler
        error_logs_file = self.log_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_logs_file,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)
        
        # Log the setup completion
        logging.info(f"Logging initialized - Level: {logging.getLevelName(self.log_level)}")
        logging.info(f"Log files location: {self.log_dir.absolute()}")
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance for a specific module."""
        return logging.getLogger(name)
    
    def log_evaluation_start(self, dataset_path: str, model_name: str) -> None:
        """Log evaluation session start."""
        logger = logging.getLogger("evaluation")
        logger.info("=" * 80)
        logger.info("EVALUATION SESSION STARTED")
        logger.info(f"Dataset: {dataset_path}")
        logger.info(f"Model: {model_name}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info("=" * 80)
    
    def log_evaluation_end(self, results: dict) -> None:
        """Log evaluation session end with results."""
        logger = logging.getLogger("evaluation")
        logger.info("=" * 80)
        logger.info("EVALUATION SESSION COMPLETED")
        logger.info(f"Total test cases: {results.get('total_cases', 'N/A')}")
        logger.info(f"Pass rate: {results.get('pass_rate', 'N/A')}%")
        logger.info(f"Average score: {results.get('average_score', 'N/A')}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info("=" * 80)

def setup_logging(log_level: str = "INFO") -> LoggerSetup:
    """Setup logging and return logger setup instance."""
    return LoggerSetup(log_level=log_level)