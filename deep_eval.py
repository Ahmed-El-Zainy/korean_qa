import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.logger import setup_logging
from src.config import Config
from src.dataset_loader import DatasetLoader
from src.evaluator import KoreanQAEvaluator

def run_legacy_evaluation():
    """Run evaluation using the legacy approach but with new logging."""
    # Setup logging
    logger_setup = setup_logging(log_level="INFO")
    logger = logger_setup.get_logger(__name__)
    
    logger.warning("Using legacy evaluation script. Consider migrating to main.py")
    
    try:
        # Load configuration
        script_dir = Path(__file__).parent
        config_path = script_dir / "src" / "config.yaml"
        config = Config(str(config_path))
        
        # Log evaluation start
        dataset_path = script_dir / "assets" / "bench_korean.csv"
        logger_setup.log_evaluation_start(str(dataset_path), config.gemini_model)
        
        # Load dataset
        dataset_loader = DatasetLoader()
        dataset = dataset_loader.load_from_csv(str(dataset_path))
        
        # Initialize evaluator
        evaluator = KoreanQAEvaluator(
            model_name=config.gemini_model,
            api_key=config.google_api_key,
            threshold=0.8,
            verbose_mode=True
        )
        
        # Run evaluation
        results = evaluator.evaluate_dataset(dataset)
        
        # Save results
        output_path = evaluator.save_results(results)
        
        # Log evaluation end
        logger_setup.log_evaluation_end(results)
        
        logger.info(f"Legacy evaluation completed. Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Legacy evaluation failed: {e}")
        raise

if __name__ == "__main__":
    run_legacy_evaluation()
