# Technology Stack

## Core Framework
- **Python 3.7+**: Primary programming language
- **DeepEval**: Main evaluation framework for LLM testing
- **Google Gemini 2.0 Flash**: AI model for evaluation scoring

## Key Dependencies
- `deepeval>=0.21.0`: Core evaluation framework
- `google-generativeai>=0.3.0`: Google AI API integration
- `pandas>=1.5.0`: Data manipulation and CSV handling
- `pyyaml>=6.0`: Configuration file parsing
- `python-dotenv>=1.0.0`: Environment variable management

## UI Frameworks
- **Gradio 4.0+**: Interactive web interface for real-time evaluation
- **Streamlit 1.28+**: Professional dashboard interface
- **Plotly 5.15+**: Interactive visualizations and charts
- **Matplotlib/Seaborn**: Statistical plotting and analysis

## Configuration Management
- **YAML**: Primary configuration format (`config.yaml`)
- **Environment Variables**: API keys and sensitive data (`.env`)
- **Dotenv**: Automatic environment loading

## Common Commands

### Setup and Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env  # Edit with your API keys
```

### Running the System
```bash
# Main CLI evaluation
python main.py --dataset assets/bench_korean.csv --threshold 0.8 --verbose

# Interactive Gradio interface
python launch_gradio.py

# Streamlit dashboard
python launch_streamlit.py

# Simple evaluation runner
python run_evaluation.py
```

### Development Commands
```bash
# Run with debug logging
python main.py --log-level DEBUG

# Evaluate single dataset with custom output
python main.py --dataset path/to/data.csv --output results/custom_results.json

# Test configuration
python src/tests/test_setup.py
```

## API Requirements
- **Google AI API Key**: Required for Gemini model access
- Set `GOOGLE_API_KEY` environment variable or in `.env` file

## File Formats
- **Input**: CSV files with `input` and `expected_output` columns
- **Output**: JSON files with detailed evaluation results
- **Logs**: Timestamped log files in `logs/` directory