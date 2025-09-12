#!/usr/bin/env python3
"""Quick test to verify the evaluator fix works."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.logger import setup_logging
from src.config import Config
from src.evaluator import KoreanQAEvaluator

def test_single_evaluation():
    """Test single case evaluation to verify score extraction."""
    
    # Setup logging
    logger_setup = setup_logging(log_level="INFO")
    logger = logger_setup.get_logger(__name__)
    
    try:
        # Load config
        config_path = Path(__file__).parent / "config.yaml"
        config = Config(str(config_path))
        
        # Initialize evaluator
        evaluator = KoreanQAEvaluator(
            model_name=config.gemini_model,
            api_key=config.google_api_key,
            threshold=0.8,
            verbose_mode=True
        )
        
        # Test case
        input_text = "이번 달 우리 회사 전체 매출은 얼마야?"
        actual_output = "2025년 1월 삼광 Global 전체 매출은 335.4억원입니다."
        
        # Run evaluation
        logger.info("Testing single case evaluation...")
        results = evaluator.evaluate_single_case(input_text, actual_output)
        
        # Check if we got real scores
        detailed_results = results.get('detailed_results', [])
        if detailed_results:
            first_case = detailed_results[0]
            metrics = first_case.get('metrics', {})
            
            logger.info("Evaluation results:")
            for metric_name, metric_data in metrics.items():
                score = metric_data.get('score')
                passed = metric_data.get('passed')
                reason = metric_data.get('reason', '')
                
                logger.info(f"  {metric_name}: {score:.4f} ({'PASS' if passed else 'FAIL'})")
                if reason and not reason.startswith('Mock') and not reason.startswith('Fallback'):
                    logger.info("  ✓ Real DeepEval score extracted successfully!")
                else:
                    logger.warning("  ⚠ Still using fallback/mock scores")
        
        return results
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return None

if __name__ == "__main__":
    test_single_evaluation()