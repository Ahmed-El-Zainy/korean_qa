# Project Structure

## Directory Organization

```
├── src/                    # Core source code modules
│   ├── __init__.py        # Package initialization
│   ├── config.py          # Configuration management and validation
│   ├── config.yaml        # YAML configuration file
│   ├── dataset_loader.py  # CSV dataset loading and processing
│   ├── evaluator.py       # Main evaluation engine with DeepEval
│   ├── logger.py          # Logging setup and utilities
│   ├── utils.py           # Utility functions and helpers
│   ├── visualization.py   # Chart generation and data visualization
│   ├── .env               # Environment variables (API keys)
│   └── tests/             # Test modules
├── assets/                # Static assets and datasets
│   └── bench_korean.csv   # Korean Q&A benchmark dataset
├── logs/                  # Auto-generated log files
├── results/               # Auto-generated evaluation results
├── main.py               # Primary CLI entry point
├── run_evaluation.py     # Simple evaluation runner
├── launch_gradio.py      # Gradio web interface launcher
├── launch_streamlit.py   # Streamlit dashboard launcher
├── gradio_demo.py        # Gradio interface implementation
├── streamlit_demo.py     # Streamlit interface implementation
└── requirements.txt      # Python dependencies
```

## Code Organization Patterns

### Module Responsibilities
- **config.py**: Centralized configuration loading, validation, and environment variable management
- **dataset_loader.py**: CSV file processing, data cleaning, and DeepEval dataset creation
- **evaluator.py**: Core evaluation logic, metric setup, and result processing
- **logger.py**: Structured logging with file rotation and multiple output streams
- **visualization.py**: Chart generation for Gradio/Streamlit interfaces
- **utils.py**: Shared utility functions and API key validation

### Entry Points
- **main.py**: Full-featured CLI with argument parsing and comprehensive logging
- **run_evaluation.py**: Simplified runner for basic evaluation tasks
- **launch_*.py**: UI launcher scripts that import and run demo interfaces
- ***_demo.py**: Complete UI implementations with class-based organization

### File Naming Conventions
- **Snake_case**: All Python files and directories
- **Descriptive names**: Clear purpose indication (e.g., `dataset_loader.py`, `launch_gradio.py`)
- **Demo separation**: UI implementations separate from launchers
- **Test organization**: Tests in dedicated `src/tests/` subdirectory

### Configuration Architecture
- **Hierarchical config**: YAML files with environment variable overrides
- **Validation layer**: Required key checking and API key validation
- **Environment separation**: `.env` files for sensitive data, YAML for application config

### Logging Strategy
- **Centralized setup**: Single logger configuration in `src/logger.py`
- **Multiple outputs**: Console + file logging with rotation
- **Structured format**: Timestamps, module names, and log levels
- **Separate error logs**: Dedicated error log files for troubleshooting

### Data Flow
1. **Configuration loading**: YAML + environment variables
2. **Dataset processing**: CSV → pandas → DeepEval format
3. **Evaluation execution**: DeepEval metrics with Gemini model
4. **Result processing**: Score calculation and statistical analysis
5. **Output generation**: JSON results + visualizations