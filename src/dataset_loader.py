
import pandas as pd
from pathlib import Path
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class DatasetLoader:
    """Handle loading and processing of evaluation datasets."""
    
    def __init__(self):
        self.dataset = EvaluationDataset()
    
    def load_from_csv(self, 
                     file_path: str,
                     input_col: str = "input",
                     output_col: str = "expected_output",
                     context_col: Optional[str] = None) -> EvaluationDataset:
        """
        Load dataset from CSV file with comprehensive logging.
        
        Args:
            file_path: Path to the CSV file
            input_col: Column name for input questions
            output_col: Column name for expected outputs
            context_col: Optional column name for context
            
        Returns:
            EvaluationDataset: Loaded dataset
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.error(f"Dataset file not found: {file_path}")
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
            logger.info(f"Loading dataset from: {file_path}")
            
            # Read CSV file
            df = pd.read_csv(file_path)
            logger.info(f"CSV file loaded successfully. Shape: {df.shape}")
            
            # Validate required columns
            required_cols = [input_col, output_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                logger.error(f"Available columns: {list(df.columns)}")
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Log column information
            logger.info(f"Dataset columns: {list(df.columns)}")
            logger.info(f"Input column: {input_col}")
            logger.info(f"Output column: {output_col}")
            if context_col:
                logger.info(f"Context column: {context_col}")
            
            # Clean and validate data
            df = self._clean_data(df, input_col, output_col)
            
            # Load test cases
            self.dataset.add_test_cases_from_csv_file(
                file_path=str(file_path),
                input_col_name=input_col,
                actual_output_col_name=output_col,
            )
            
            logger.info(f"Successfully loaded {len(self.dataset.test_cases)} test cases")
            
            # Log sample data
            self._log_sample_data(df, input_col, output_col)
            
            return self.dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def _clean_data(self, df: pd.DataFrame, input_col: str, output_col: str) -> pd.DataFrame:
        """Clean and validate dataset."""
        logger.info("Cleaning dataset...")
        
        initial_count = len(df)
        
        # Remove rows with missing values in required columns
        df = df.dropna(subset=[input_col, output_col])
        
        # Remove empty strings
        df = df[df[input_col].str.strip() != '']
        df = df[df[output_col].str.strip() != '']
        
        final_count = len(df)
        removed_count = initial_count - final_count
        
        if removed_count > 0:
            logger.warning(f"Removed {removed_count} invalid rows during cleaning")
        
        logger.info(f"Dataset cleaned. Final count: {final_count} rows")
        
        return df
    
    def _log_sample_data(self, df: pd.DataFrame, input_col: str, output_col: str) -> None:
        """Log sample data for verification."""
        logger.info("Sample data from dataset:")
        
        for i, row in df.head(3).iterrows():
            logger.info(f"Sample {i+1}:")
            logger.info(f"  Input: {row[input_col][:100]}...")
            logger.info(f"  Output: {row[output_col][:100]}...")
    
    def get_dataset_stats(self) -> dict:
        """Get dataset statistics."""
        if not self.dataset.test_cases:
            return {"total_cases": 0}
        
        stats = {
            "total_cases": len(self.dataset.test_cases),
            "avg_input_length": sum(len(case.input) for case in self.dataset.test_cases) / len(self.dataset.test_cases),
            "avg_output_length": sum(len(case.actual_output or "") for case in self.dataset.test_cases) / len(self.dataset.test_cases)
        }
        
        logger.info(f"Dataset statistics: {stats}")
        return stats