import os
import yaml
from pathlib import Path
from typing import Dict, Any
import logging
from .utils import load_environment_variables, validate_api_keys

logger = logging.getLogger(__name__)

class Config:    
    def __init__(self, config_path: str = "config.yaml"):
        load_environment_variables()
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
        
        # Validate API keys
        api_validation = validate_api_keys()
        if not api_validation['valid']:
            raise ValueError(f"Missing required API keys: {api_validation['missing_required']}")
    
    
    def _load_config(self) -> Dict[str, Any]:
        try:
            if not self.config_path.exists():
                logger.error(f"Configuration file not found: {self.config_path}")
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            return config
        
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise

        except Exception as e:
            logger.error(f"Unexpected error loading configuration: {e}")
            raise



    def _validate_config(self) -> None:
        required_keys = ['gemini_model']
        for key in required_keys:
            if key not in self.config:
                logger.error(f"Missing required configuration key: {key}")
                raise ValueError(f"Missing required configuration key: {key}")
        logger.info("Configuration validation passed")
    
    
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        value = self.config.get(key, default)
        logger.debug(f"Retrieved config value for '{key}': {value}")
        return value
    

    
    def get_env_var(self, key: str, required: bool = True) -> str:
        value = os.getenv(key)
        if required and not value:
            logger.error(f"Required environment variable not found: {key}")
            raise ValueError(f"Required environment variable not found: {key}")
        if value:
            logger.info(f"Environment variable '{key}' loaded successfully")
        else:
            logger.warning(f"Optional environment variable '{key}' not found")
        return value
    
    
    
    @property
    def gemini_model(self) -> str:
        """Get Gemini model name."""
        return self.get('gemini_model')
    
    @property
    def google_api_key(self) -> str:
        """Get Google API key from environment."""
        return self.get_env_var('GOOGLE_API_KEY')