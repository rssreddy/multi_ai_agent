import logging
from typing import Any, Optional

class Logger:
    """
    Centralized logging utility for consistent logging across the application.
    Implements the Singleton pattern to ensure a single logger instance.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._setup_logger()
        return cls._instance
    
    def _setup_logger(self):
        """Configure the logger with appropriate formatting and level."""
        self.logger = logging.getLogger("workflow")
        self.logger.setLevel(logging.INFO)
        
        # Create console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(handler)
    
    def node_transition(self, current_node: str, next_node: Optional[str] = None):
        """
        Log a node transition in the workflow.
        
        Args:
            current_node: The current node name
            next_node: The next node name (if applicable)
        """
        if next_node:
            self.logger.info(f"Current Node: {current_node} -> Goto: {next_node}")
        else:
            self.logger.info(f"Current Node: {current_node}")
    
    def debug(self, message: str, data: Any = None):
        """
        Log a debug message, optionally with data.
        
        Args:
            message: The debug message
            data: Optional data to include in the log
        """
        if data:
            self.logger.debug(f"{message}: {data}")
        else:
            self.logger.debug(message)
    
    def info(self, message: str):
        """
        Log an info message.
        
        Args:
            message: The info message
        """
        self.logger.info(message)
    
    def warning(self, message: str):
        """
        Log a warning message.
        
        Args:
            message: The warning message
        """
        self.logger.warning(message)
    
    def error(self, message: str, exc_info: bool = False):
        """
        Log an error message.
        
        Args:
            message: The error message
            exc_info: Whether to include exception info
        """
        self.logger.error(message, exc_info=exc_info)

# Create a singleton instance
logger = Logger()