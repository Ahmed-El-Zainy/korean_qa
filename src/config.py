import os
import yaml
from pathlib import Path
from typing import Dict, Any
import logging
import sys 

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .utilites import load_environment_variables, validate_api_keys


# Import logger here to avoid circular imports
try:
    from logger.custom_logger import CustomLoggerTracker
    custom_log = CustomLoggerTracker()
    logger = custom_log.get_logger("config")
except ImportError:
    # Fallback to standard logging if custom logger not available
    logger = logging.getLogger("config")

class Config:    
    def __init__(self, config_path: str = "config.yaml"):
        logger.info("Start Loading data from configs")
        load_environment_variables()
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
        
        # Validate API keys
        api_validation = validate_api_keys()
        if not api_validation['valid']:
            logger.warning(f"Some API keys missing: {api_validation['missing_required']}")
            # Don't raise error for missing optional keys, just warn
    
    
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
        """Validate configuration based on the actual YAML structure."""
        
        # Check if we have either the old structure (gemini_model) or new structure (models + rag_system)
        has_gemini = 'gemini_model' in self.config
        has_models_section = 'models' in self.config
        has_rag_section = 'rag_system' in self.config
        
        if not has_gemini and not has_models_section:
            logger.error("Missing required configuration: either 'gemini_model' or 'models' section must be configured")
            raise ValueError("Missing required configuration: either 'gemini_model' or 'models' section must be configured")
        
        # Validate models section if present
        if has_models_section:
            models_config = self.config['models']
            required_models = ['embedding_model', 'llm_model']
            for key in required_models:
                if key not in models_config:
                    logger.error(f"Missing required model configuration: models.{key}")
                    raise ValueError(f"Missing required model configuration: models.{key}")
        
        # Validate rag_system section if present (optional validation)
        if has_rag_section:
            rag_config = self.config['rag_system']
            # These are optional but log if missing
            optional_rag_keys = ['chunk_size', 'chunk_overlap', 'max_context_chunks']
            for key in optional_rag_keys:
                if key not in rag_config:
                    logger.debug(f"Optional RAG configuration key not found: rag_system.{key}")
        
        # Validate vector store section if present
        if 'vector_store' in self.config:
            vector_config = self.config['vector_store']
            if 'provider' in vector_config and vector_config['provider'] == 'qdrant':
                # Check for qdrant specific config
                if 'collection_name' not in vector_config:
                    logger.warning("Qdrant collection_name not specified, will use default")
        
        logger.info("Configuration validation passed")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key, supporting nested keys with dot notation."""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            logger.debug(f"Retrieved config value for '{key}': {value}")
            return value
        except (KeyError, TypeError):
            logger.debug(f"Config key '{key}' not found, returning default: {default}")
            return default
    
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
        """Get Gemini model name (optional for RAG system)."""
        return self.get('gemini_model', 'models/gemini-2.5-flash')
    
    @property
    def google_api_key(self) -> str:
        """Get Google API key from environment."""
        try:
            return self.get_env_var('GOOGLE_API_KEY')
        except ValueError:
            logger.warning("Google API key not found, this is optional for RAG-only usage")
            return ""
    
    # RAG System Properties
    @property
    def rag_config(self) -> Dict[str, Any]:
        """Get RAG system configuration, combining rag_system and models sections."""
        rag_config = self.get('rag_system', {}).copy()
        
        # Add models to rag config if they exist
        models_config = self.get('models', {})
        if models_config:
            rag_config.update(models_config)
        
        # Add performance settings
        performance_config = self.get('performance', {})
        if performance_config:
            rag_config.update(performance_config)
            
        return rag_config
    
    @property
    def groq_api_key(self) -> str:
        """Get Groq API key from environment."""
        return self.get_env_var('GROQ_API_KEY', required=False) or ""
    
    @property
    def siliconflow_api_key(self) -> str:
        """Get Silicon Flow API key from environment."""
        return self.get_env_var('SILICONFLOW_API_KEY', required=False) or ""
    
    @property
    def qdrant_url(self) -> str:
        """Get Qdrant URL from environment or config."""
        env_url = self.get_env_var('QDRANT_URL', required=False)
        if env_url:
            return env_url
        return self.get('vector_store.qdrant_url', 'http://localhost:6333')
    
    @property
    def qdrant_api_key(self) -> str:
        """Get Qdrant API key from environment."""
        return self.get_env_var('QDRANT_API_KEY', required=False) or ""
    
    @property
    def document_processing_config(self) -> Dict[str, Any]:
        """Get document processing configuration."""
        return self.get('document_processing', {})
    
    @property
    def storage_config(self) -> Dict[str, Any]:
        """Get storage configuration."""
        # Combine multiple storage-related sections
        storage_config = {}
        
        # Vector store config
        vector_store = self.get('vector_store', {})
        if vector_store:
            storage_config.update(vector_store)
        
        # Cache config  
        cache_config = self.get('cache', {})
        if cache_config:
            storage_config.update(cache_config)
            
        # Add any storage-specific settings
        if 'storage' in self.config:
            storage_config.update(self.config['storage'])
            
        return storage_config

# Test the configuration loading
if __name__ == "__main__":
    try:
        config = Config()
        print("✅ Configuration loaded successfully!")
        print(f"RAG Config keys: {list(config.rag_config.keys())}")
        print(f"Has Groq API key: {'Yes' if config.groq_api_key else 'No'}")
        print(f"Has SiliconFlow API key: {'Yes' if config.siliconflow_api_key else 'No'}")
        print(f"Qdrant URL: {config.qdrant_url}")
    except Exception as e:
        print(f"❌ Configuration failed: {e}")