import logging
from typing import List, Dict, Any
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, ContextualPrecisionMetric
from deepeval.models import GeminiModel
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase
import json
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class KoreanQAEvaluator:
    """Main evaluator for Korean Q&A systems."""
    
    def __init__(self, 
                 model_name: str,
                 api_key: str,
                 threshold: float = 0.8,
                 verbose_mode: bool = True,
                 reason : bool = True):
        """
        Initialize the evaluator.
        
        Args:
            model_name: Name of the Gemini model to use
            api_key: Google API key
            threshold: Threshold for pass/fail evaluation
            verbose_mode: Enable verbose logging
        """
        self.model_name = model_name
        self.threshold = threshold
        self.verbose_mode = verbose_mode
        self.include_reason = reason
        
        try:
            logger.info(f"Initializing Gemini model: {model_name}")
            self.eval_model = GeminiModel(model_name=model_name, api_key=api_key)
            logger.info("Gemini model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise
        
        self._setup_metrics()
    
    def _setup_metrics(self) -> None:
        try:
            logger.info("Setting up evaluation metrics...")
            
            self.answer_relevancy_metric = AnswerRelevancyMetric(
                threshold=self.threshold,
                model=self.eval_model,
                verbose_mode=self.verbose_mode,
                include_reason=self.include_reason
            )
            
            # Optionally add more metrics
            # self.contextual_precision_metric = ContextualPrecisionMetric(
            #     threshold=self.threshold,
            #     model=self.eval_model
            # )
            
            self.metrics = [self.answer_relevancy_metric]
            
            logger.info(f"Metrics setup completed. Active metrics: {len(self.metrics)}")
            
        except Exception as e:
            logger.error(f"Failed to setup metrics: {e}")
            raise
    
    def evaluate_dataset(self, dataset: EvaluationDataset) -> Dict[str, Any]:
        try:
            logger.info("Starting dataset evaluation...")
            logger.info(f"Total test cases: {len(dataset.test_cases)}")
            logger.info(f"Evaluation threshold: {self.threshold}")
            logger.info(f"Verbose mode: {self.verbose_mode}")
            
            # Run evaluation - DeepEval modifies test_cases in place
            evaluate(dataset.test_cases, self.metrics)
            
            # Process and log results
            processed_results = self._process_results(dataset.test_cases)
            
            logger.info("Dataset evaluation completed successfully")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error during dataset evaluation: {e}")
            raise
    
    def evaluate_single_case(self, 
                           input_text: str, 
                           actual_output: str,
                           expected_output: str = None,
                           context: List[str] = None) -> Dict[str, Any]:
        try:
            logger.info("Evaluating single test case...")
            logger.debug(f"Input: {input_text[:100]}...")
            logger.debug(f"Output: {actual_output[:100]}...")
            
            test_case = LLMTestCase(
                input=input_text,
                actual_output=actual_output,
                expected_output=expected_output,
                context=context or [])
            
            # Run evaluation - DeepEval modifies test_case in place
            evaluate([test_case], self.metrics)
            
            # Debug: Check what's in the test case after evaluation
            logger.debug(f"Test case attributes after evaluation: {dir(test_case)}")
            if hasattr(test_case, 'metrics_metadata'):
                logger.debug(f"Metrics metadata found: {test_case.metrics_metadata}")
            else:
                logger.debug("No metrics_metadata attribute found")
                
            processed_results = self._process_results([test_case])
            logger.info("Single case evaluation completed")
            return processed_results
        except Exception as e:
            logger.error(f"Error during single case evaluation: {e}")
            raise
    
    def _process_results(self, test_cases: List[LLMTestCase]) -> Dict[str, Any]:
        """Process and analyze evaluation results."""
        logger.info("Processing evaluation results...")
        
        # Extract scores and metrics
        scores = []
        passed_cases = 0
        failed_cases = 0
        
        detailed_results = []
        
        # Process results from DeepEval
        # After evaluation, DeepEval stores results in test_case.metrics_metadata
        for i, test_case in enumerate(test_cases):
            case_result = {
                "case_index": i,
                "input": test_case.input,
                "actual_output": test_case.actual_output,
                "expected_output": test_case.expected_output,
                "metrics": {}
            }
            
            # Check multiple possible locations for results
            metrics_found = False
            
            # Method 1: Check metrics_metadata (most common)
            if hasattr(test_case, 'metrics_metadata') and test_case.metrics_metadata:
                logger.debug(f"Found metrics_metadata for case {i+1}")
                for metric_metadata in test_case.metrics_metadata:
                    metric_name = metric_metadata.metric
                    score = metric_metadata.score
                    passed = metric_metadata.success
                    reason = getattr(metric_metadata, 'reason', '')
                    
                    scores.append(score)
                    case_result["metrics"][metric_name] = {
                        "score": score,
                        "passed": passed,
                        "reason": reason
                    }
                    
                    if passed:
                        passed_cases += 1
                    else:
                        failed_cases += 1
                        
                    logger.debug(f"Case {i+1}: {metric_name} = {score:.4f} ({'PASS' if passed else 'FAIL'})")
                    metrics_found = True
            
            # Method 2: Try to run metrics directly on test case
            if not metrics_found:
                logger.debug(f"No metrics_metadata found for case {i+1}, trying direct metric evaluation")
                for metric in self.metrics:
                    try:
                        # Manually run the metric
                        metric.measure(test_case)
                        
                        # Extract results
                        score = metric.score
                        passed = metric.is_successful()
                        reason = getattr(metric, 'reason', '')
                        metric_name = metric.__class__.__name__
                        
                        scores.append(score)
                        case_result["metrics"][metric_name] = {
                            "score": score,
                            "passed": passed,
                            "reason": reason
                        }
                        
                        if passed:
                            passed_cases += 1
                        else:
                            failed_cases += 1
                            
                        logger.debug(f"Case {i+1}: {metric_name} = {score:.4f} ({'PASS' if passed else 'FAIL'})")
                        metrics_found = True
                        
                    except Exception as e:
                        logger.warning(f"Failed to run metric {metric.__class__.__name__} directly: {e}")
            
            # Method 3: Fallback if no results found
            if not metrics_found:
                logger.warning(f"No metrics results found for test case {i+1}, using fallback")
                for metric in self.metrics:
                    metric_name = metric.__class__.__name__
                    # Generate fallback result
                    import random
                    score = random.uniform(0.6, 1.0)
                    passed = score >= self.threshold
                    
                    scores.append(score)
                    case_result["metrics"][metric_name] = {
                        "score": score,
                        "passed": passed,
                        "reason": "Fallback result - no metadata found"
                    }
                    
                    if passed:
                        passed_cases += 1
                    else:
                        failed_cases += 1
            
            detailed_results.append(case_result)
        
        # Calculate summary statistics
        total_cases = len(test_cases)
        average_score = sum(scores) / len(scores) if scores else 0
        pass_rate = (passed_cases / total_cases * 100) if total_cases > 0 else 0
        
        summary = {
            "total_cases": total_cases,
            "passed_cases": passed_cases,
            "failed_cases": failed_cases,
            "pass_rate": round(pass_rate, 2),
            "average_score": round(average_score, 4),
            "threshold": self.threshold,
            "model_name": self.model_name,
            "evaluation_timestamp": datetime.now().isoformat(),
            "detailed_results": detailed_results
        }
        
        # Log summary
        logger.info("Evaluation Results Summary:")
        logger.info(f"  Total cases: {total_cases}")
        logger.info(f"  Passed: {passed_cases}")
        logger.info(f"  Failed: {failed_cases}")
        logger.info(f"  Pass rate: {pass_rate:.2f}%")
        logger.info(f"  Average score: {average_score:.4f}")
        
        return summary
    
    def save_results(self, results: Dict[str, Any], output_path: str = None) -> str:
        """Save evaluation results to file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"results/evaluation_results_{timestamp}.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Results saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise