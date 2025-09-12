import logging
import os
import yaml
from datetime import datetime
import sys
from typing import List

class CustomLoggerTracker:
    _instance = None
    _initialized = False
    
    def __new__(cls, config_path='logging_config.yaml'):
        """Singleton pattern to ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super(CustomLoggerTracker, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_path='logging_config.yaml'):
        """Initialize the custom logger with configuration."""
        if self._initialized:
            return
            
        self.config = self._load_config(config_path)
        self.loggers = {}
        self.base_log_dir = self.config.get('base_log_dir', 'logs')
        self._setup_base_directory()
        self._initialized = True
        
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            # Default configuration if file not found
            return {
                'base_log_dir': 'logs',
                'default_level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'console_output': True,
                'modules': {
                    'main': {'level': 'INFO'},
                    'utils': {'level': 'INFO'},
                    'old_docs': {'level': 'INFO'},
                    'rag': {'level': 'INFO'},
                    'query_utils': {'level': 'INFO'},
                    'prompt_temp': {'level': 'INFO'}
                }
            }

    def _setup_base_directory(self):
        """Setup the base directory structure for logs."""
        if not os.path.exists(self.base_log_dir):
            os.makedirs(self.base_log_dir)

    def _get_log_path(self, module_name):
        """Generate the hierarchical path for log files."""
        now = datetime.now()
        year_dir = os.path.join(self.base_log_dir, str(now.year))
        month_dir = os.path.join(year_dir, f"{now.month:02d}")
        day_dir = os.path.join(month_dir, f"{now.day:02d}")
        os.makedirs(day_dir, exist_ok=True)
        return os.path.join(day_dir, f"{module_name}.log")

    def get_logger(self, module_name):
        """Get or create a logger for a specific module."""
        if module_name in self.loggers:
            return self.loggers[module_name]

        # Create new logger & Models Specific Config
        logger = logging.getLogger(module_name)
        module_config = self.config['modules'].get(module_name, {})
        level = getattr(logging, module_config.get('level', self.config['default_level']))
        logger.setLevel(level)

        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(self.config.get('format'))

        # Create file handler with the hierarchical path
        log_path = self._get_log_path(module_name)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Optionally add console handler
        if self.config.get('console_output', True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # Prevent propagation to avoid duplicate messages
        logger.propagate = False
        
        self.loggers[module_name] = logger
        return logger

    def update_config(self, new_config):
        """Update logger configuration."""
        self.config.update(new_config)
        # Reset all loggers to apply new configuration
        for module_name in self.loggers:
            logger = self.loggers[module_name]
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
        self.loggers = {}

    def log_message(self, process_log: List[str], message: str, level: str = "info", module: str = "default") -> None:
        """
        Append to process_log AND send to the central logger.
        
        Args:
            process_log: List to append the message to
            message: The message to log
            level: Log level ('info', 'warning', 'error')
            module: Module name for the logger (optional, defaults to 'default')
        """
        process_log.append(message)
        
        # Get the logger for the specified module
        logger = self.get_logger(module)
        
        # Log the message at the appropriate level
        if level.lower() == "error":
            logger.error(message)
        elif level.lower() == "warning":
            logger.warning(message)
        else:
            logger.info(message)

    def log_info(self, message: str, module: str = "default") -> None:
        """Log an info message."""
        logger = self.get_logger(module)
        logger.info(message)

    def log_warning(self, message: str, module: str = "default") -> None:
        """Log a warning message."""
        logger = self.get_logger(module)
        logger.warning(message)

    def log_error(self, message: str, module: str = "default") -> None:
        """Log an error message."""
        logger = self.get_logger(module)
        logger.error(message)

    # Alternative method names that match your original _log function pattern
    def _log(self, process_log: List[str], message: str, level: str = "info", module: str = "default") -> None:
        """Alias for log_message to match your original function name."""
        self.log_message(process_log, message, level, module)


# Create a default instance for easy importing
default_logger = CustomLoggerTracker()

# Expose the methods at module level for easy importing
log_message = default_logger.log_message
log_info = default_logger.log_info
log_warning = default_logger.log_warning
log_error = default_logger.log_error
_log = default_logger._log


# Example usage
if __name__ == "__main__":
    # Method 1: Create your own instance
    logger_tracker = CustomLoggerTracker()
    process_log = []
    
    logger_tracker.log_message(process_log, "This is a test info message", "info", "registration")
    logger_tracker.log_message(process_log, "This is a warning message", "warning", "registration")
    logger_tracker.log_message(process_log, "This is an error message", "error", "registration")
    
    # Method 2: Use the default instance functions
    process_log2 = []
    log_message(process_log2, "Using default logger", "info", "detection")
    _log(process_log2, "Using _log alias", "warning", "detection")
    
    # Method 3: Direct logging without process_log
    log_info("Direct info message", "main")
    log_warning("Direct warning message", "main")
    log_error("Direct error message", "main")
    
    print("Process log 1 contents:")
    for log_entry in process_log:
        print(f"  {log_entry}")
    
    print("Process log 2 contents:")
    for log_entry in process_log2:
        print(f"  {log_entry}")