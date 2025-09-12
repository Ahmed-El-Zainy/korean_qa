"""Gradio demo interface for the Korean Q&A evaluation system."""

import gradio as gr
import json
import sys
from pathlib import Path
import logging
import pandas as pd
from typing import Dict, Any, Tuple

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.logger import setup_logging
from src.config import Config
from src.dataset_loader import DatasetLoader
from src.evaluator import KoreanQAEvaluator
from src.visualization import EvaluationVisualizer

# Setup logging
logger_setup = setup_logging(log_level="INFO")
logger = logger_setup.get_logger(__name__)

class GradioDemo:
    """Gradio demo interface for Korean Q&A evaluation."""
    
    def __init__(self):
        self.config = None
        self.evaluator = None
        self.visualizer = EvaluationVisualizer()
        self.current_results = None
        
        # Try to load config
        try:
            script_dir = Path(__file__).parent
            config_path = script_dir / "src" / "config.yaml"
            if config_path.exists():
                self.config = Config(str(config_path))
                logger.info("Configuration loaded successfully")
            else:
                logger.warning("Configuration file not found")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
    
    def evaluate_single_question(self, 
                                input_text: str, 
                                actual_output: str,
                                api_key: str = None) -> Tuple[str, str, str]:
        """Evaluate a single question-answer pair."""
        try:
            if not input_text.strip() or not actual_output.strip():
                return "❌ Error: Please provide both input and output text", "", ""
            
            # Initialize evaluator if needed
            if self.evaluator is None or api_key:
                if not api_key and self.config:
                    api_key = self.config.google_api_key
                
                if not api_key:
                    return "❌ Error: Please provide Google API key", "", ""
                
                model_name = self.config.gemini_model if self.config else "gemini-2.0-flash"
                self.evaluator = KoreanQAEvaluator(
                    model_name=model_name,
                    api_key=api_key,
                    threshold=0.8,
                    verbose_mode=True
                )
            
            # Run evaluation
            results = self.evaluator.evaluate_single_case(
                input_text=input_text,
                actual_output=actual_output
            )
            
            # Format results
            summary = self._format_single_result(results)
            
            # Create visualizations
            score_hist = self.visualizer.create_score_histogram(results)
            pie_chart = self.visualizer.create_pass_fail_pie_chart(results)
            
            return summary, score_hist, pie_chart
            
        except Exception as e:
            logger.error(f"Error in single evaluation: {e}")
            return f"❌ Error: {str(e)}", None, None
    
    def evaluate_dataset(self, 
                        dataset_file,
                        api_key: str = None,
                        threshold: float = 0.8) -> Tuple[str, str, str, str, str]:
        """Evaluate an entire dataset."""
        try:
            if dataset_file is None:
                return "❌ Error: Please upload a dataset file", "", "", "", ""
            
            # Initialize evaluator
            if not api_key and self.config:
                api_key = self.config.google_api_key
            
            if not api_key:
                return "❌ Error: Please provide Google API key", "", "", "", ""
            
            model_name = self.config.gemini_model if self.config else "gemini-2.0-flash"
            self.evaluator = KoreanQAEvaluator(
                model_name=model_name,
                api_key=api_key,
                threshold=threshold,
                verbose_mode=True
            )
            
            # Load dataset
            dataset_loader = DatasetLoader()
            dataset = dataset_loader.load_from_csv(dataset_file.name)
            
            # Run evaluation
            results = self.evaluator.evaluate_dataset(dataset)
            self.current_results = results
            
            # Format summary
            summary = self._format_dataset_results(results)
            
            # Create visualizations
            score_hist = self.visualizer.create_score_histogram(results)
            pie_chart = self.visualizer.create_pass_fail_pie_chart(results)
            metrics_comp = self.visualizer.create_metrics_comparison(results)
            scatter_plot = self.visualizer.create_score_vs_length_scatter(results)
            
            return summary, score_hist, pie_chart, metrics_comp, scatter_plot
            
        except Exception as e:
            logger.error(f"Error in dataset evaluation: {e}")
            return f"❌ Error: {str(e)}", None, None, None, None
    
    def download_results(self) -> str:
        """Prepare results for download."""
        if self.current_results is None:
            return None
        
        try:
            # Save results to temporary file
            output_path = "temp_results.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.current_results, f, ensure_ascii=False, indent=2)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error preparing download: {e}")
            return None
    
    def _format_single_result(self, results: Dict[str, Any]) -> str:
        """Format single evaluation result."""
        summary = "## 📊 Single Evaluation Results\n\n"
        
        if results.get('detailed_results'):
            result = results['detailed_results'][0]
            
            summary += f"**Input:** {result.get('input', 'N/A')[:200]}...\n\n"
            summary += f"**Output:** {result.get('actual_output', 'N/A')[:200]}...\n\n"
            
            summary += "### Metrics:\n"
            for metric_name, metric_data in result.get('metrics', {}).items():
                score = metric_data.get('score', 0)
                passed = metric_data.get('passed', False)
                status = "✅ PASS" if passed else "❌ FAIL"
                summary += f"- **{metric_name}**: {score:.4f} {status}\n"
        
        summary += f"\n**Threshold:** {results.get('threshold', 0.8)}\n"
        summary += f"**Model:** {results.get('model_name', 'N/A')}\n"
        
        return summary
    
    def _format_dataset_results(self, results: Dict[str, Any]) -> str:
        """Format dataset evaluation results."""
        summary = "## 📊 Dataset Evaluation Results\n\n"
        
        summary += f"**Total Cases:** {results.get('total_cases', 0)}\n"
        summary += f"**Passed Cases:** {results.get('passed_cases', 0)}\n"
        summary += f"**Failed Cases:** {results.get('failed_cases', 0)}\n"
        summary += f"**Pass Rate:** {results.get('pass_rate', 0):.2f}%\n"
        summary += f"**Average Score:** {results.get('average_score', 0):.4f}\n"
        summary += f"**Threshold:** {results.get('threshold', 0.8)}\n"
        summary += f"**Model:** {results.get('model_name', 'N/A')}\n\n"
        
        # Add some sample results
        if results.get('detailed_results'):
            summary += "### Sample Results:\n"
            for i, result in enumerate(results['detailed_results'][:3]):
                summary += f"\n**Case {i+1}:**\n"
                summary += f"- Input: {result.get('input', 'N/A')[:100]}...\n"
                
                for metric_name, metric_data in result.get('metrics', {}).items():
                    score = metric_data.get('score', 0)
                    passed = metric_data.get('passed', False)
                    status = "✅" if passed else "❌"
                    summary += f"- {metric_name}: {score:.3f} {status}\n"
        
        return summary
    
    def create_interface(self):
        """Create the Gradio interface."""
        with gr.Blocks(title="Korean Q&A Evaluation System", theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
            # 🇰🇷 Korean Q&A Evaluation System
            
            Evaluate Korean language question-answering systems using Google's Gemini model.
            """)
            
            with gr.Tabs():
                # Single Evaluation Tab
                with gr.TabItem("Single Evaluation"):
                    gr.Markdown("### Evaluate a single question-answer pair")
                    
                    with gr.Row():
                        with gr.Column():
                            single_input = gr.Textbox(
                                label="Input Question (Korean)",
                                placeholder="이번 달 우리 회사 전체 매출은 얼마야?",
                                lines=3
                            )
                            single_output = gr.Textbox(
                                label="Actual Output (Korean)",
                                placeholder="2025년 1월 삼광 Global 전체 매출은 335.4억원입니다...",
                                lines=5
                            )
                            single_api_key = gr.Textbox(
                                label="Google API Key (optional if configured)",
                                type="password",
                                placeholder="Enter your Google API key"
                            )
                            single_eval_btn = gr.Button("🔍 Evaluate", variant="primary")
                        
                        with gr.Column():
                            single_results = gr.Markdown(label="Results")
                    
                    with gr.Row():
                        single_score_plot = gr.Plot(label="Score Distribution")
                        single_pie_plot = gr.Plot(label="Pass/Fail")
                    
                    single_eval_btn.click(
                        fn=self.evaluate_single_question,
                        inputs=[single_input, single_output, single_api_key],
                        outputs=[single_results, single_score_plot, single_pie_plot]
                    )
                
                # Dataset Evaluation Tab
                with gr.TabItem("Dataset Evaluation"):
                    gr.Markdown("### Evaluate an entire dataset from CSV file")
                    
                    with gr.Row():
                        with gr.Column():
                            dataset_file = gr.File(
                                label="Upload Dataset CSV",
                                file_types=[".csv"],
                                type="filepath"
                            )
                            dataset_api_key = gr.Textbox(
                                label="Google API Key (optional if configured)",
                                type="password",
                                placeholder="Enter your Google API key"
                            )
                            dataset_threshold = gr.Slider(
                                label="Evaluation Threshold",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.8,
                                step=0.1
                            )
                            dataset_eval_btn = gr.Button("📊 Evaluate Dataset", variant="primary")
                        
                        with gr.Column():
                            dataset_results = gr.Markdown(label="Results Summary")
                            download_btn = gr.File(label="Download Results JSON")
                    
                    with gr.Row():
                        dataset_score_plot = gr.Plot(label="Score Distribution")
                        dataset_pie_plot = gr.Plot(label="Pass/Fail Distribution")
                    
                    with gr.Row():
                        metrics_comparison_plot = gr.Plot(label="Metrics Comparison")
                        scatter_plot = gr.Plot(label="Score vs Length Analysis")
                    
                    dataset_eval_btn.click(
                        fn=self.evaluate_dataset,
                        inputs=[dataset_file, dataset_api_key, dataset_threshold],
                        outputs=[dataset_results, dataset_score_plot, dataset_pie_plot, 
                                metrics_comparison_plot, scatter_plot]
                    )
                    
                    # Download functionality
                    download_results_btn = gr.Button("📥 Prepare Download")
                    download_results_btn.click(
                        fn=self.download_results,
                        outputs=download_btn
                    )
                
                # About Tab
                with gr.TabItem("About"):
                    gr.Markdown("""
                    ## About Korean Q&A Evaluation System
                    
                    This system evaluates Korean language question-answering models using:
                    
                    - **DeepEval Framework**: Advanced evaluation metrics
                    - **Google Gemini Model**: State-of-the-art language model for evaluation
                    - **Answer Relevancy Metric**: Measures how well answers address questions
                    
                    ### Features:
                    - ✅ Single question evaluation
                    - ✅ Batch dataset evaluation
                    - ✅ Interactive visualizations
                    - ✅ Detailed metrics analysis
                    - ✅ Results export
                    
                    ### Supported Metrics:
                    - **Answer Relevancy**: How relevant is the answer to the question?
                    - **Contextual Precision**: How precise is the answer given the context?
                    
                    ### CSV Format:
                    Your dataset should have columns: `input`, `expected_output`
                    
                    ```csv
                    input,expected_output
                    "이번 달 매출은?","2025년 1월 매출은 335억원입니다."
                    ```
                    """)
        
        return demo

def main():
    """Launch the Gradio demo."""
    demo_app = GradioDemo()
    demo = demo_app.create_interface()
    
    # Launch with public link for sharing
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Creates public link
        show_error=True,
        debug=True
    )

if __name__ == "__main__":
    main()