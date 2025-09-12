"""Main entry point for the Korean Q&A evaluation system."""

import argparse
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.logger import setup_logging
from src.config import Config
from src.dataset_loader import DatasetLoader
from src.evaluator import KoreanQAEvaluator

def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description="Korean Q&A Evaluation System")
    parser.add_argument("--config", default=None, help="src/config.yaml")
    parser.add_argument("--dataset", default="/Users/ahmedmostafa/Downloads/eval_Korean_qa/assets/bench_korean.csv", help="Path to dataset CSV file")
    parser.add_argument("--output", help="Output path for results (optional)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       help="Logging level")
    parser.add_argument("--threshold", type=float, default=0.8, help="Evaluation threshold")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose evaluation mode")
    
    args = parser.parse_args()
    
    # Setup logging
    logger_setup = setup_logging(log_level=args.log_level)
    logger = logger_setup.get_logger(__name__)
    
    try:
        logger.info("Starting Korean Q&A Evaluation System")
        
        # Load configuration
        logger.info("Loading configuration...")
        if args.config is None:
            # Try to find config file in multiple locations
            script_dir = Path(__file__).parent
            possible_configs = [script_dir / "src" / "config.yaml",
                script_dir / "config.yaml"]


            config_path = None
            for path in possible_configs:
                if path.exists():
                    config_path = str(path)
                    break
            
            if config_path is None:
                raise FileNotFoundError("No config.yaml found in expected locations")
        else:
            config_path = args.config
        config = Config(config_path)
        
        # Log evaluation start
        logger_setup.log_evaluation_start(args.dataset, config.gemini_model)
        
        # Load dataset
        logger.info("Loading dataset...")
        dataset_loader = DatasetLoader()
        dataset = dataset_loader.load_from_csv(args.dataset)
        
        # Get dataset statistics
        stats = dataset_loader.get_dataset_stats()
        logger.info(f"Dataset loaded: {stats}")
        
        # Initialize evaluator
        logger.info("Initializing evaluator...")
        evaluator = KoreanQAEvaluator(
            model_name=config.gemini_model,
            api_key=config.google_api_key,
            threshold=args.threshold,
            verbose_mode=args.verbose)
        
        
        # Run evaluation
        logger.info("Running evaluation...")
        results = evaluator.evaluate_dataset(dataset)
        
        # Save results
        if args.output:
            output_path = evaluator.save_results(results, args.output)
        else:
            output_path = evaluator.save_results(results)
        
        
        # Log evaluation end
        logger_setup.log_evaluation_end(results)
        logger.info(f"Evaluation completed successfully. Results saved to: {output_path}")
        # Print summary to console
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total test cases: {results['total_cases']}")
        print(f"Passed cases: {results['passed_cases']}")
        print(f"Failed cases: {results['failed_cases']}")
        print(f"Pass rate: {results['pass_rate']}%")
        print(f"Average score: {results['average_score']}")
        print(f"Threshold: {results['threshold']}")
        print(f"Model: {results['model_name']}")
        print(f"Results saved to: {output_path}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()