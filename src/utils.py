
import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def load_environment_variables(env_file: str = None) -> None:
    if env_file is None:
        # Try multiple locations
        possible_paths = [
            Path("src/.env"),
            Path(".env"),
            Path(__file__).parent / ".env"
        ]
        
        for env_path in possible_paths:
            if env_path.exists():
                load_dotenv(env_path)
                logger.info(f"Environment variables loaded from {env_path}")
                return
        
        logger.warning("No .env file found in any of the expected locations")
    else:
        env_path = Path(env_file)
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Environment variables loaded from {env_path}")
        else:
            logger.warning(f"Environment file not found: {env_path}")

def validate_api_keys() -> dict:
    """
    Validate that required API keys are available.
    
    Returns:
        Dict with validation results
    """
    required_keys = {
        'GOOGLE_API_KEY': 'Google AI API key for Gemini model'
    }
    
    optional_keys = {
        'OPENAI_API_KEY': 'OpenAI API key',
        'HF_TOKEN': 'Hugging Face token',
        'nvidia_api_key': 'NVIDIA API key',
        'GROQ_API_KEY': 'Groq API key for RAG system',
        'SILICON_FLOW_API_KEY': 'Silicon Flow API key for embeddings and reranking',
        'QDRANT_URL': 'Qdrant vector database URL',
        'QDRANT_API_KEY': 'Qdrant API key'
    }
    
    validation_results = {
        'valid': True,
        'missing_required': [],
        'missing_optional': [],
        'available_keys': []
    }
    
    # Check required keys
    for key, description in required_keys.items():
        if os.getenv(key):
            validation_results['available_keys'].append(key)
            logger.info(f"✓ {key} is available")
        else:
            validation_results['missing_required'].append(key)
            validation_results['valid'] = False
            logger.error(f"✗ Missing required {key}: {description}")
    
    # Check optional keys
    for key, description in optional_keys.items():
        if os.getenv(key):
            validation_results['available_keys'].append(key)
            logger.info(f"✓ {key} is available (optional)")
        else:
            validation_results['missing_optional'].append(key)
            logger.debug(f"- {key} not found (optional): {description}")
    
    return validation_results

def ensure_directory_exists(directory: str) -> Path:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        directory: Directory path to create
        
    Returns:
        Path object of the directory
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Directory ensured: {dir_path}")
    return dir_path

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"



def load_config_yaml(config_path: str):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
