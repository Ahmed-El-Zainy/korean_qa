"""Streamlit demo interface for the Korean Q&A evaluation system."""

import streamlit as st
import json
import sys
from pathlib import Path
import logging
import pandas as pd
from typing import Dict, Any
import plotly.graph_objects as go

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.logger import setup_logging
from src.config import Config
from src.dataset_loader import DatasetLoader
from src.evaluator import KoreanQAEvaluator
from src.visualization import EvaluationVisualizer

# Page config
st.set_page_config(
    page_title="Korean Q&A Evaluation System",
    page_icon="🇰🇷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .error-metric {
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitDemo:
    """Streamlit demo interface for Korean Q&A evaluation."""
    
    def __init__(self):
        self.visualizer = EvaluationVisualizer()
        
        # Initialize session state
        if 'config' not in st.session_state:
            st.session_state.config = self._load_config()
        if 'evaluator' not in st.session_state:
            st.session_state.evaluator = None
        if 'current_results' not in st.session_state:
            st.session_state.current_results = None
    
    def _load_config(self):
        """Load configuration."""
        try:
            script_dir = Path(__file__).parent
            config_path = script_dir / "src" / "config.yaml"
            if config_path.exists():
                return Config(str(config_path))
        except Exception as e:
            st.error(f"Failed to load configuration: {e}")
        return None
    
    def _initialize_evaluator(self, api_key: str, threshold: float = 0.8):
        """Initialize the evaluator."""
        try:
            if not api_key and st.session_state.config:
                api_key = st.session_state.config.google_api_key
            
            if not api_key:
                st.error("Please provide Google API key")
                return False
            
            model_name = st.session_state.config.gemini_model if st.session_state.config else "gemini-2.0-flash"
            st.session_state.evaluator = KoreanQAEvaluator(
                model_name=model_name,
                api_key=api_key,
                threshold=threshold,
                verbose_mode=True
            )
            return True
            
        except Exception as e:
            st.error(f"Failed to initialize evaluator: {e}")
            return False
    
    def render_header(self):
        """Render the main header."""
        st.markdown('<h1 class="main-header">🇰🇷 Korean Q&A Evaluation System</h1>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Info section
        with st.expander("ℹ️ About this system"):
            st.markdown("""
            This system evaluates Korean language question-answering models using:
            
            - **DeepEval Framework**: Advanced evaluation metrics
            - **Google Gemini Model**: State-of-the-art language model for evaluation
            - **Interactive Visualizations**: Real-time charts and analysis
            
            **Supported Metrics:**
            - Answer Relevancy: How relevant is the answer to the question?
            - Contextual Precision: How precise is the answer given the context?
            """)
    
    def render_sidebar(self):
        """Render the sidebar with configuration."""
        st.sidebar.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.sidebar.text_input(
            "Google API Key",
            type="password",
            help="Enter your Google API key for Gemini model access"
        )
        
        # Threshold slider
        threshold = st.sidebar.slider(
            "Evaluation Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.1,
            help="Minimum score required to pass evaluation"
        )
        
        # Model info
        if st.session_state.config:
            st.sidebar.info(f"Model: {st.session_state.config.gemini_model}")
        
        return api_key, threshold
    
    def render_single_evaluation(self, api_key: str, threshold: float):
        """Render single evaluation interface."""
        st.header("🔍 Single Question Evaluation")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            input_text = st.text_area(
                "Input Question (Korean)",
                placeholder="이번 달 우리 회사 전체 매출은 얼마야?",
                height=100
            )
            
            actual_output = st.text_area(
                "Actual Output (Korean)",
                placeholder="2025년 1월 삼광 Global 전체 매출은 335.4억원입니다...",
                height=150
            )
            
            if st.button("🔍 Evaluate Single Question", type="primary"):
                if not input_text.strip() or not actual_output.strip():
                    st.error("Please provide both input and output text")
                    return
                
                if not self._initialize_evaluator(api_key, threshold):
                    return
                
                with st.spinner("Evaluating..."):
                    try:
                        results = st.session_state.evaluator.evaluate_single_case(
                            input_text=input_text,
                            actual_output=actual_output
                        )
                        
                        # Display results
                        self._display_single_results(results)
                        
                    except Exception as e:
                        st.error(f"Evaluation failed: {e}")
        
        with col2:
            st.info("💡 **Tips:**\n\n- Enter Korean text for best results\n- Longer, more detailed answers typically score higher\n- The system evaluates relevance, not correctness")
    
    def render_dataset_evaluation(self, api_key: str, threshold: float):
        """Render dataset evaluation interface."""
        st.header("📊 Dataset Evaluation")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Dataset CSV",
            type=['csv'],
            help="CSV file should have 'input' and 'expected_output' columns"
        )
        
        if uploaded_file is not None:
            # Show preview
            try:
                df = pd.read_csv(uploaded_file)
                st.subheader("📋 Dataset Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    st.metric("Total Rows", len(df))
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    st.write("**Columns:**", ", ".join(df.columns.tolist()))
                
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                return
            
            # Evaluation button
            if st.button("📊 Evaluate Dataset", type="primary"):
                if not self._initialize_evaluator(api_key, threshold):
                    return
                
                with st.spinner("Evaluating dataset... This may take a while."):
                    try:
                        # Save uploaded file temporarily
                        temp_path = "temp_dataset.csv"
                        df.to_csv(temp_path, index=False)
                        
                        # Load and evaluate
                        dataset_loader = DatasetLoader()
                        dataset = dataset_loader.load_from_csv(temp_path)
                        
                        results = st.session_state.evaluator.evaluate_dataset(dataset)
                        st.session_state.current_results = results
                        
                        # Display results
                        self._display_dataset_results(results)
                        
                        # Clean up
                        Path(temp_path).unlink(missing_ok=True)
                        
                    except Exception as e:
                        st.error(f"Dataset evaluation failed: {e}")
    
    def _display_single_results(self, results: Dict[str, Any]):
        """Display single evaluation results."""
        st.subheader("📈 Evaluation Results")
        
        if results.get('detailed_results'):
            result = results['detailed_results'][0]
            
            # Metrics display
            metrics = result.get('metrics', {})
            if metrics:
                cols = st.columns(len(metrics))
                for i, (metric_name, metric_data) in enumerate(metrics.items()):
                    with cols[i]:
                        score = metric_data.get('score', 0)
                        passed = metric_data.get('passed', False)
                        
                        # Color based on pass/fail
                        if passed:
                            st.markdown(f'<div class="metric-card success-metric">', unsafe_allow_html=True)
                            st.metric(metric_name, f"{score:.4f}", "✅ PASS")
                        else:
                            st.markdown(f'<div class="metric-card error-metric">', unsafe_allow_html=True)
                            st.metric(metric_name, f"{score:.4f}", "❌ FAIL")
                        st.markdown('</div>', unsafe_allow_html=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            with col1:
                fig = self.visualizer.create_score_histogram(results)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = self.visualizer.create_pass_fail_pie_chart(results)
                st.plotly_chart(fig, use_container_width=True)
    
    def _display_dataset_results(self, results: Dict[str, Any]):
        """Display dataset evaluation results."""
        st.subheader("📊 Dataset Evaluation Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Cases", results.get('total_cases', 0))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            passed = results.get('passed_cases', 0)
            st.markdown('<div class="metric-card success-metric">', unsafe_allow_html=True)
            st.metric("Passed", passed)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            failed = results.get('failed_cases', 0)
            st.markdown('<div class="metric-card error-metric">', unsafe_allow_html=True)
            st.metric("Failed", failed)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            pass_rate = results.get('pass_rate', 0)
            color_class = "success-metric" if pass_rate >= 80 else "warning-metric" if pass_rate >= 60 else "error-metric"
            st.markdown(f'<div class="metric-card {color_class}">', unsafe_allow_html=True)
            st.metric("Pass Rate", f"{pass_rate:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Score", f"{results.get('average_score', 0):.4f}")
        with col2:
            st.metric("Threshold", results.get('threshold', 0.8))
        with col3:
            st.metric("Model", results.get('model_name', 'N/A'))
        
        # Visualizations
        st.subheader("📈 Detailed Analysis")
        
        # First row of charts
        col1, col2 = st.columns(2)
        with col1:
            fig = self.visualizer.create_score_histogram(results)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = self.visualizer.create_pass_fail_pie_chart(results)
            st.plotly_chart(fig, use_container_width=True)
        
        # Second row of charts
        fig = self.visualizer.create_metrics_comparison(results)
        st.plotly_chart(fig, use_container_width=True)
        
        # Third row
        fig = self.visualizer.create_score_vs_length_scatter(results)
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary table
        fig = self.visualizer.create_summary_stats_table(results)
        st.plotly_chart(fig, use_container_width=True)
        
        # Download results
        if st.button("📥 Download Results JSON"):
            json_str = json.dumps(results, ensure_ascii=False, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="evaluation_results.json",
                mime="application/json"
            )
    
    def render_sample_data_tab(self):
        """Render sample data information."""
        st.header("📋 Sample Data Format")
        
        st.markdown("""
        ### CSV Format Requirements
        
        Your dataset CSV file should have the following columns:
        - `input`: The question or input text (Korean)
        - `expected_output`: The expected answer or output text (Korean)
        """)
        
        # Sample data
        sample_data = {
            'input': [
                '이번 달 우리 회사 전체 매출은 얼마야?',
                '사업부별 매출 비중이 어떻게 되나요?',
                '최근 수율이 낮은 공정이 있나요?'
            ],
            'expected_output': [
                '2025년 1월 삼광 Global 전체 매출은 335.4억원입니다.',
                '한국 사업부: 213.0억원 (39.7%), 베트남 사업부: 38.6억원 (44.1%)',
                'R47 ENCLOSURE 사출: 59%, R47 ARM 사출: 80% 등이 90% 미만입니다.'
            ]
        }
        
        sample_df = pd.DataFrame(sample_data)
        st.subheader("Sample Data")
        st.dataframe(sample_df, use_container_width=True)
        
        # Download sample
        csv = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample CSV",
            data=csv,
            file_name="sample_korean_qa.csv",
            mime="text/csv"
        )
    
    def run(self):
        """Run the Streamlit app."""
        self.render_header()
        
        # Sidebar
        api_key, threshold = self.render_sidebar()
        
        # Main tabs
        tab1, tab2, tab3 = st.tabs(["🔍 Single Evaluation", "📊 Dataset Evaluation", "📋 Sample Data"])
        
        with tab1:
            self.render_single_evaluation(api_key, threshold)
        
        with tab2:
            self.render_dataset_evaluation(api_key, threshold)
        
        with tab3:
            self.render_sample_data_tab()

def main():
    """Main function to run the Streamlit app."""
    demo = StreamlitDemo()
    demo.run()

if __name__ == "__main__":
    main()