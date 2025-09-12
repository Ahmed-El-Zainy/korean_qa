"""Visualization utilities for the Korean Q&A evaluation system."""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class EvaluationVisualizer:
    """Create visualizations for evaluation results."""
    
    def __init__(self):
        # Set style for matplotlib
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def create_score_histogram(self, results: Dict[str, Any], metric_name: str = "Answer Relevancy") -> go.Figure:
        """
        Create histogram of evaluation scores.
        
        Args:
            results: Evaluation results dictionary
            metric_name: Name of the metric to visualize
            
        Returns:
            Plotly figure object
        """
        try:
            # Extract scores from detailed results
            scores = []
            for result in results.get('detailed_results', []):
                metrics = result.get('metrics', {})
                for metric, data in metrics.items():
                    # Handle both display names and class names
                    if (metric_name.lower() in metric.lower() or 
                        metric_name.replace(" ", "").lower() in metric.lower() or
                        "answerrelevancy" in metric.lower()):
                        scores.append(data.get('score', 0))
            
            if not scores:
                logger.warning(f"No scores found for metric: {metric_name}")
                return self._create_empty_figure("No data available")
            
            # Create histogram
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=scores,
                nbinsx=20,
                name=metric_name,
                marker_color='skyblue',
                opacity=0.7,
                hovertemplate='Score Range: %{x}<br>Count: %{y}<extra></extra>'
            ))
            
            # Add threshold line
            threshold = results.get('threshold', 0.8)
            fig.add_vline(
                x=threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold: {threshold}",
                annotation_position="top right"
            )
            
            # Update layout
            fig.update_layout(
                title=f'{metric_name} Score Distribution',
                xaxis_title='Score',
                yaxis_title='Frequency',
                showlegend=False,
                template='plotly_white',
                height=400)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating score histogram: {e}")
            return self._create_empty_figure("Error creating histogram")
    
    def create_pass_fail_pie_chart(self, results: Dict[str, Any]) -> go.Figure:
        """Create pie chart showing pass/fail distribution."""
        try:
            passed = results.get('passed_cases', 0)
            failed = results.get('failed_cases', 0)
            
            if passed == 0 and failed == 0:
                return self._create_empty_figure("No evaluation data available")
            
            fig = go.Figure(data=[go.Pie(
                labels=['Passed', 'Failed'],
                values=[passed, failed],
                hole=0.3,
                marker_colors=['#2E8B57', '#DC143C'],
                hovertemplate='%{label}: %{value} cases<br>%{percent}<extra></extra>'
            )])
            
            fig.update_layout(
                title='Pass/Fail Distribution',
                template='plotly_white',
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating pie chart: {e}")
            return self._create_empty_figure("Error creating pie chart")
    
    def create_metrics_comparison(self, results: Dict[str, Any]) -> go.Figure:
        """Create comparison chart for different metrics."""
        try:
            # Extract metrics data
            metrics_data = {}
            
            for result in results.get('detailed_results', []):
                metrics = result.get('metrics', {})
                for metric_name, data in metrics.items():
                    if metric_name not in metrics_data:
                        metrics_data[metric_name] = []
                    metrics_data[metric_name].append(data.get('score', 0))
            
            if not metrics_data:
                return self._create_empty_figure("No metrics data available")
            
            # Create subplots
            fig = make_subplots(
                rows=len(metrics_data),
                cols=1,
                subplot_titles=list(metrics_data.keys()),
                vertical_spacing=0.1
            )
            
            colors = px.colors.qualitative.Set3
            
            for i, (metric_name, scores) in enumerate(metrics_data.items()):
                fig.add_trace(
                    go.Histogram(
                        x=scores,
                        name=metric_name,
                        marker_color=colors[i % len(colors)],
                        opacity=0.7,
                        nbinsx=15
                    ),
                    row=i+1, col=1
                )
            
            fig.update_layout(
                title='Metrics Comparison',
                template='plotly_white',
                height=300 * len(metrics_data),
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating metrics comparison: {e}")
            return self._create_empty_figure("Error creating metrics comparison")
    
    
    
    
    def create_score_vs_length_scatter(self, results: Dict[str, Any]) -> go.Figure:
        try:
            scores = []
            input_lengths = []
            output_lengths = []
            for result in results.get('detailed_results', []):
                input_text = result.get('input', '')
                output_text = result.get('actual_output', '')
                input_lengths.append(len(input_text))
                output_lengths.append(len(output_text))
                # Get the first available score
                metrics = result.get('metrics', {})
                score = 0
                for metric_data in metrics.values():
                    score = metric_data.get('score', 0)
                    break
                scores.append(score)
            if not scores:
                return self._create_empty_figure("No data available for scatter plot")
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=['Score vs Input Length', 'Score vs Output Length']
            )
            
            # Input length scatter
            fig.add_trace(
                go.Scatter(
                    x=input_lengths,
                    y=scores,
                    mode='markers',
                    name='Input Length',
                    marker=dict(color='blue', opacity=0.6),
                    hovertemplate='Input Length: %{x}<br>Score: %{y:.3f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Output length scatter
            fig.add_trace(
                go.Scatter(
                    x=output_lengths,
                    y=scores,
                    mode='markers',
                    name='Output Length',
                    marker=dict(color='red', opacity=0.6),
                    hovertemplate='Output Length: %{x}<br>Score: %{y:.3f}<extra></extra>'
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title='Score vs Text Length Analysis',
                template='plotly_white',
                height=400,
                showlegend=False
            )
            
            fig.update_xaxes(title_text="Input Length (characters)", row=1, col=1)
            fig.update_xaxes(title_text="Output Length (characters)", row=1, col=2)
            fig.update_yaxes(title_text="Score", row=1, col=1)
            fig.update_yaxes(title_text="Score", row=1, col=2)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating scatter plot: {e}")
            return self._create_empty_figure("Error creating scatter plot")
    
    def create_summary_stats_table(self, results: Dict[str, Any]) -> go.Figure:
        """Create summary statistics table."""
        try:
            stats = [
                ['Total Cases', results.get('total_cases', 0)],
                ['Passed Cases', results.get('passed_cases', 0)],
                ['Failed Cases', results.get('failed_cases', 0)],
                ['Pass Rate', f"{results.get('pass_rate', 0):.2f}%"],
                ['Average Score', f"{results.get('average_score', 0):.4f}"],
                ['Threshold', results.get('threshold', 0.8)],
                ['Model', results.get('model_name', 'N/A')],
                ['Evaluation Time', results.get('evaluation_timestamp', 'N/A')]
            ]
            
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=['Metric', 'Value'],
                    fill_color='lightblue',
                    align='left',
                    font=dict(size=14, color='black')
                ),
                cells=dict(
                    values=list(zip(*stats)),
                    fill_color='white',
                    align='left',
                    font=dict(size=12)
                )
            )])
            
            fig.update_layout(
                title='Evaluation Summary',
                height=400,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating summary table: {e}")
            return self._create_empty_figure("Error creating summary table")
    
    def _create_empty_figure(self, message: str) -> go.Figure:
        """Create an empty figure with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            template='plotly_white',
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
    
    def save_matplotlib_plots(self, results: Dict[str, Any], output_dir: str = "plots") -> List[str]:
        """Save matplotlib plots to files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        
        try:
            # Extract scores
            scores = []
            for result in results.get('detailed_results', []):
                metrics = result.get('metrics', {})
                for metric_data in metrics.values():
                    scores.append(metric_data.get('score', 0))
                    break
            
            if scores:
                # Score histogram
                plt.figure(figsize=(10, 6))
                plt.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                plt.axvline(results.get('threshold', 0.8), color='red', linestyle='--', 
                           label=f"Threshold: {results.get('threshold', 0.8)}")
                plt.xlabel('Score')
                plt.ylabel('Frequency')
                plt.title('Score Distribution')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                hist_file = os.path.join(output_dir, 'score_histogram.png')
                plt.savefig(hist_file, dpi=300, bbox_inches='tight')
                plt.close()
                saved_files.append(hist_file)
                
                # Box plot
                plt.figure(figsize=(8, 6))
                plt.boxplot(scores, labels=['Scores'])
                plt.ylabel('Score')
                plt.title('Score Distribution (Box Plot)')
                plt.grid(True, alpha=0.3)
                
                box_file = os.path.join(output_dir, 'score_boxplot.png')
                plt.savefig(box_file, dpi=300, bbox_inches='tight')
                plt.close()
                saved_files.append(box_file)
            
            logger.info(f"Saved {len(saved_files)} matplotlib plots to {output_dir}")
            return saved_files
            
        except Exception as e:
            logger.error(f"Error saving matplotlib plots: {e}")
            return []