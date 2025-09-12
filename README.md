---
title: eval_sales_korean_agent
app_file: gradio_demo.py
sdk: gradio
sdk_version: 5.40.0
---
# Korean Q&A Evaluation System

A comprehensive evaluation framework for Korean language Q&A systems using DeepEval, designed to assess answer relevancy and quality for business intelligence queries.

## Overview

This project evaluates Korean language question-answering systems using Google's Gemini model through the DeepEval framework. It focuses on business-related queries covering financial metrics, production data, and operational insights.

## Features

### 🔍 Evaluation Capabilities
- **Answer Relevancy Evaluation**: Measures how well answers address the input questions
- **Korean Language Support**: Specialized for Korean business terminology and context
- **Comprehensive Metrics**: Detailed scoring with verbose logging and explanations
- **CSV Dataset Integration**: Easy data loading from structured CSV files
- **Gemini Model Integration**: Leverages Google's latest Gemini 2.0 Flash model

### 🌐 Interactive Demos
- **Gradio Interface**: User-friendly web interface with real-time evaluation
- **Streamlit Dashboard**: Professional analytics dashboard with advanced visualizations
- **Single Question Evaluation**: Test individual Q&A pairs instantly
- **Batch Dataset Processing**: Evaluate entire datasets with progress tracking
- **Public Sharing**: Generate shareable links for collaborative evaluation

### 📊 Advanced Visualizations
- **Score Distribution Histograms**: Understand score patterns across your dataset
- **Pass/Fail Analytics**: Visual breakdown of success rates
- **Metrics Comparison Charts**: Compare different evaluation metrics
- **Text Length Analysis**: Correlation between answer length and scores
- **Interactive Tables**: Sortable, filterable results with detailed breakdowns
- **Export Capabilities**: Download results in JSON format for further analysis

## Project Structure

```
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── logger.py          # Logging setup and utilities
│   ├── dataset_loader.py  # Dataset loading and processing
│   ├── evaluator.py       # Main evaluation engine
│   └── utils.py           # Utility functions
├── assets/
│   └── bench_korean.csv   # Korean Q&A benchmark dataset
├── logs/                  # Log files (auto-created)
├── results/               # Evaluation results (auto-created)
├── main.py               # Main entry point
├── run_evaluation.py     # Simple runner script
├── deep_eval.py          # Legacy script (deprecated)
├── config.yaml           # Configuration file
├── .env                  # Environment variables
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## Dataset

The benchmark dataset (`assets/bench_korean.csv`) contains Korean business Q&A pairs covering:

- **Financial Metrics**: Revenue, profit margins, cost analysis
- **Production Data**: Manufacturing yields, process efficiency
- **Operational Insights**: Inventory status, departmental performance
- **Quality Control**: Failure costs, process optimization

Sample questions include:
- "이번 달 우리 회사 전체 매출은 얼마야?" (What's our company's total revenue this month?)
- "사업부별 매출 비중이 어떻게 되나요?" (What's the revenue distribution by business unit?)
- "최근 수율이 낮은 공정이 있나요?" (Are there any processes with low yields recently?)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd eval_Korean_qa
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Google AI API key in `deep_eval.py`:
```python
GOOGLEAI_API_KEY = "your-api-key-here"
```

## Usage

### 🌐 Live Demo Interfaces

#### Gradio Demo (Recommended)
Interactive web interface with real-time evaluation and visualizations:

```bash
python launch_gradio.py
```

- **Local**: http://localhost:7860
- **Public**: Shareable link generated automatically
- **Features**: Single evaluation, batch processing, interactive charts

#### Streamlit Demo
Professional dashboard interface:

```bash
python launch_streamlit.py
```

- **Local**: http://localhost:8501
- **Features**: Advanced visualizations, detailed analytics, download results

### 🖥️ Command Line Interface

#### Quick Start

Run evaluation with default settings:

```bash
python run_evaluation.py
```

#### Advanced Usage

Run evaluation with custom parameters:

```bash
python main.py --dataset assets/bench_korean.csv --threshold 0.8 --verbose --log-level INFO
```

#### Command Line Options

```bash
python main.py --help
```

Available options:
- `--config`: Path to configuration file (default: src/config.yaml)
- `--dataset`: Path to dataset CSV file (default: assets/bench_korean.csv)
- `--output`: Output path for results (optional)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `--threshold`: Evaluation threshold (default: 0.8)
- `--verbose`: Enable verbose evaluation mode

### 📊 Visualization Features

Both demo interfaces include:

- **Score Distribution Histograms**: Visual distribution of evaluation scores
- **Pass/Fail Pie Charts**: Success rate visualization
- **Metrics Comparison**: Side-by-side metric analysis
- **Score vs Length Analysis**: Correlation between text length and scores
- **Interactive Tables**: Detailed results with sorting and filtering
- **Export Functionality**: Download results in JSON format

### 🔧 Programmatic Usage

```python
from src.config import Config
from src.dataset_loader import DatasetLoader
from src.evaluator import KoreanQAEvaluator
from src.logger import setup_logging
from src.visualization import EvaluationVisualizer

# Setup logging
logger_setup = setup_logging()

# Load configuration
config = Config("src/config.yaml")

# Load dataset
dataset_loader = DatasetLoader()
dataset = dataset_loader.load_from_csv("assets/bench_korean.csv")

# Run evaluation
evaluator = KoreanQAEvaluator(
    model_name=config.gemini_model,
    api_key=config.google_api_key
)
results = evaluator.evaluate_dataset(dataset)

# Create visualizations
visualizer = EvaluationVisualizer()
score_hist = visualizer.create_score_histogram(results)
pie_chart = visualizer.create_pass_fail_pie_chart(results)
```

## Evaluation Metrics

### Answer Relevancy
- **Threshold**: 0.8 (configurable)
- **Model**: Gemini 2.0 Flash
- **Scoring**: 0.0 to 1.0 scale
- **Verbose Mode**: Detailed statement-by-statement analysis

### Results Interpretation

- **Score ≥ 0.8**: Pass (relevant answer)
- **Score < 0.8**: Fail (needs improvement)
- **Overall Pass Rate**: Percentage of test cases meeting threshold

## Configuration

Key parameters in `deep_eval.py`:

```python
EVAL_MODEL = "gemini-2.0-flash"           # Evaluation model
threshold = 0.8                           # Pass/fail threshold
verbose_mode = True                       # Detailed logging
```

## Sample Results

Recent evaluation achieved:
- **Overall Pass Rate**: 91.67%
- **Perfect Scores**: 10/12 test cases
- **Average Score**: 0.94

Common failure patterns:
- Irrelevant topic mentions in responses
- Off-topic statements mixed with relevant content

## Logging

The system provides comprehensive logging with multiple levels and outputs:

### Log Files

- `logs/evaluation_YYYYMMDD.log`: All evaluation logs
- `logs/errors_YYYYMMDD.log`: Error logs only
- Console output: Real-time logging during execution

### Log Levels

- **DEBUG**: Detailed debugging information
- **INFO**: General information about execution
- **WARNING**: Warning messages
- **ERROR**: Error messages

### Log Features

- Automatic log rotation (10MB max file size)
- Timestamped entries
- Module and line number tracking
- Separate error log files
- Configurable log levels

## Requirements

- Python 3.7+
- DeepEval framework
- Google AI API access
- Pandas for data handling
- PyYAML for configuration
- python-dotenv for environment variables

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add test cases to the CSV dataset
4. Update evaluation metrics as needed
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For questions or issues:
- Check the DeepEval documentation
- Review the verbose evaluation logs
- Ensure proper API key configuration