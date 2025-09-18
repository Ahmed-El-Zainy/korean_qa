#!/usr/bin/env python3
"""
Complete CSV Question Evaluation Script for Manufacturing RAG Agent
"""

import pandas as pd
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('csv_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    from src.config import Config
    from src.rag_engine import RAGEngine
    from src.document_processor import DocumentProcessorFactory, DocumentType
    from src.pdf_processor import PDFProcessor
    from src.excel_processor import ExcelProcessor
    from src.image_processor import ImageProcessor
    
except ImportError as e:
    logger.error(f"Failed to import RAG components: {e}")
    print(f"❌ Import Error: {e}")
    print("Please ensure all src/ modules are properly structured and dependencies are installed")
    sys.exit(1)


class CSVEvaluator:
    """CSV-based question evaluation system."""
    
    def __init__(self, config_path: str = "src/config.yaml"):
        """Initialize the CSV evaluator."""
        self.config_path = config_path
        self.rag_engine = None
        self.system_initialized = False
        
    def initialize_system(self) -> bool:
        """Initialize the RAG system."""
        try:
            logger.info("Initializing RAG system...")
            
            # Load configuration
            if not os.path.exists(self.config_path):
                logger.error(f"Configuration file not found: {self.config_path}")
                return False
            
            config = Config(self.config_path)
            
            # Validate required API keys
            required_keys = {
                'GROQ_API_KEY': config.groq_api_key,
                'SILICONFLOW_API_KEY': config.siliconflow_api_key,
                'QDRANT_URL': config.qdrant_url
            }
            
            missing_keys = [k for k, v in required_keys.items() if not v]
            if missing_keys:
                logger.error(f"Missing required environment variables: {', '.join(missing_keys)}")
                return False
            
            # Create configuration dictionary
            rag_config = config.rag_config
            config_dict = {
                # API configuration
                'siliconflow_api_key': config.siliconflow_api_key,
                'groq_api_key': config.groq_api_key,
                'qdrant_url': config.qdrant_url,
                'qdrant_api_key': config.qdrant_api_key,
                'qdrant_collection': 'manufacturing_docs',
                
                # Model configuration
                'embedding_model': rag_config.get('embedding_model', 'Qwen/Qwen3-Embedding-8B'),
                'reranker_model': rag_config.get('reranker_model', 'Qwen/Qwen3-Reranker-8B'), 
                'llm_model': rag_config.get('llm_model', 'openai/gpt-oss-120b'),
                
                # RAG parameters
                'max_context_chunks': rag_config.get('max_context_chunks', 5),
                'similarity_threshold': rag_config.get('similarity_threshold', 0.7),
                'rerank_top_k': rag_config.get('rerank_top_k', 20),
                'final_top_k': rag_config.get('final_top_k', 5),
                'max_context_length': 4000,
                'vector_size': 1024,
                
                # Performance settings
                'max_retries': 3,
                'temperature': rag_config.get('temperature', 0.1),
                'max_tokens': rag_config.get('max_tokens', 1024)
            }
            
            # Register document processors
            DocumentProcessorFactory.register_processor(DocumentType.PDF, PDFProcessor)
            DocumentProcessorFactory.register_processor(DocumentType.EXCEL, ExcelProcessor)
            DocumentProcessorFactory.register_processor(DocumentType.IMAGE, ImageProcessor)
            
            # Initialize RAG engine
            self.rag_engine = RAGEngine(config_dict)
            
            # Verify system health
            health = self.rag_engine.health_check()
            if not health.get('vector_store', False):
                logger.warning("Vector store health check failed - this might affect performance")
            
            if not health.get('llm_system', False):
                logger.error("LLM system health check failed")
                return False
            
            self.system_initialized = True
            logger.info("✅ RAG system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            return False
    
    def load_questions_csv(self, csv_path: str, question_column: str = "question") -> pd.DataFrame:
        """Load questions from CSV file."""
        try:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} questions from {csv_path}")
            
            if question_column not in df.columns:
                raise ValueError(f"Question column '{question_column}' not found in CSV. Available columns: {df.columns.tolist()}")
            
            # Remove empty questions
            original_count = len(df)
            df = df[df[question_column].notna() & (df[question_column].str.strip() != "")]
            final_count = len(df)
            
            if original_count != final_count:
                logger.info(f"Filtered out {original_count - final_count} empty questions")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load questions CSV: {e}")
            raise
    
    def evaluate_questions(self, questions_df: pd.DataFrame, question_column: str = "question",
                          batch_size: int = 10, delay_between_batches: float = 1.0) -> pd.DataFrame:
        """Evaluate questions and return results DataFrame."""
        if not self.system_initialized:
            raise RuntimeError("System not initialized. Call initialize_system() first.")
        
        results = []
        total_questions = len(questions_df)
        
        logger.info(f"Starting evaluation of {total_questions} questions...")
        
        # Process questions in batches to avoid overwhelming the API
        for batch_start in range(0, total_questions, batch_size):
            batch_end = min(batch_start + batch_size, total_questions)
            batch_df = questions_df.iloc[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}/{(total_questions-1)//batch_size + 1} "
                       f"(questions {batch_start+1}-{batch_end})")
            
            # Process each question in the batch
            for idx, row in batch_df.iterrows():
                question = row[question_column]
                
                try:
                    logger.info(f"Processing question {idx+1}: {question[:50]}...")
                    
                    # Get answer from RAG system
                    start_time = time.time()
                    response = self.rag_engine.answer_question(question)
                    processing_time = time.time() - start_time
                    
                    # Extract result information
                    result = {
                        'question_id': idx,
                        'question': question,
                        'answer': response.answer if response.success else "Error: Could not generate answer",
                        'success': response.success,
                        'confidence_score': response.confidence_score if response.success else 0.0,
                        'processing_time': processing_time,
                        'retrieval_time': response.retrieval_time if response.success else 0.0,
                        'generation_time': response.generation_time if response.success else 0.0,
                        'sources_count': len(response.citations) if response.success else 0,
                        'chunks_retrieved': response.total_chunks_retrieved if response.success else 0,
                        'model_used': response.model_used if response.success else "N/A",
                        'error_message': response.error_message if not response.success else "",
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Add citations information
                    if response.success and response.citations:
                        citations_info = []
                        for i, citation in enumerate(response.citations):
                            citation_text = f"Source {i+1}: {citation.source_file}"
                            if citation.page_number:
                                citation_text += f" (Page {citation.page_number})"
                            if citation.worksheet_name:
                                citation_text += f" (Sheet: {citation.worksheet_name})"
                            citations_info.append(citation_text)
                        
                        result['citations'] = " | ".join(citations_info)
                        result['top_citation_confidence'] = max([c.confidence for c in response.citations])
                    else:
                        result['citations'] = ""
                        result['top_citation_confidence'] = 0.0
                    
                    # Copy additional columns from original CSV
                    for col in row.index:
                        if col != question_column and col not in result:
                            result[col] = row[col]
                    
                    results.append(result)
                    
                    # Log success
                    if response.success:
                        logger.info(f"✅ Question {idx+1} processed successfully "
                                  f"(confidence: {response.confidence_score:.2f}, "
                                  f"time: {processing_time:.2f}s)")
                    else:
                        logger.warning(f"⚠️ Question {idx+1} failed: {response.error_message}")
                    
                except Exception as e:
                    logger.error(f"❌ Error processing question {idx+1}: {e}")
                    
                    # Add error result
                    error_result = {
                        'question_id': idx,
                        'question': question,
                        'answer': f"Error: {str(e)}",
                        'success': False,
                        'confidence_score': 0.0,
                        'processing_time': 0.0,
                        'retrieval_time': 0.0,
                        'generation_time': 0.0,
                        'sources_count': 0,
                        'chunks_retrieved': 0,
                        'model_used': "N/A",
                        'error_message': str(e),
                        'citations': "",
                        'top_citation_confidence': 0.0,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Copy additional columns
                    for col in row.index:
                        if col != question_column and col not in error_result:
                            error_result[col] = row[col]
                    
                    results.append(error_result)
                
                # Small delay between questions
                time.sleep(0.5)
            
            # Delay between batches
            if batch_end < total_questions:
                logger.info(f"Waiting {delay_between_batches}s before next batch...")
                time.sleep(delay_between_batches)
        
        logger.info(f"Completed evaluation of {len(results)} questions")
        return pd.DataFrame(results)
    
    def save_results(self, results_df: pd.DataFrame, output_path: str, 
                    include_summary: bool = True) -> str:
        """Save results to CSV file and optionally create summary."""
        try:
            # Ensure output directory exists
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save main results
            results_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
            
            # Create summary if requested
            if include_summary:
                summary_path = output_file.with_suffix('.summary.txt')
                summary = self._generate_summary(results_df)
                
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(summary)
                
                logger.info(f"Summary saved to {summary_path}")
                return str(summary_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
    
    def _generate_summary(self, results_df: pd.DataFrame) -> str:
        """Generate evaluation summary."""
        total_questions = len(results_df)
        successful_questions = len(results_df[results_df['success'] == True])
        failed_questions = total_questions - successful_questions
        
        success_rate = (successful_questions / total_questions * 100) if total_questions > 0 else 0
        
        # Calculate statistics for successful questions
        successful_df = results_df[results_df['success'] == True]
        
        if len(successful_df) > 0:
            avg_confidence = successful_df['confidence_score'].mean()
            avg_processing_time = successful_df['processing_time'].mean()
            avg_sources = successful_df['sources_count'].mean()
            avg_chunks = successful_df['chunks_retrieved'].mean()
        else:
            avg_confidence = avg_processing_time = avg_sources = avg_chunks = 0
        
        # Generate summary text
        summary = f"""
=== Manufacturing RAG Agent - CSV Evaluation Summary ===
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 Overall Results:
• Total Questions: {total_questions}
• Successful Answers: {successful_questions}
• Failed Answers: {failed_questions}
• Success Rate: {success_rate:.1f}%

📈 Performance Metrics (Successful Questions):
• Average Confidence Score: {avg_confidence:.3f}
• Average Processing Time: {avg_processing_time:.2f}s
• Average Sources per Answer: {avg_sources:.1f}
• Average Chunks Retrieved: {avg_chunks:.1f}

📋 Detailed Breakdown:
"""
        
        # Add confidence distribution
        if len(successful_df) > 0:
            confidence_ranges = [
                (0.9, 1.0, "Very High (0.9-1.0)"),
                (0.7, 0.9, "High (0.7-0.9)"),
                (0.5, 0.7, "Medium (0.5-0.7)"),
                (0.0, 0.5, "Low (0.0-0.5)")
            ]
            
            summary += "\n🎯 Confidence Score Distribution:\n"
            for min_conf, max_conf, label in confidence_ranges:
                count = len(successful_df[
                    (successful_df['confidence_score'] >= min_conf) & 
                    (successful_df['confidence_score'] < max_conf)
                ])
                percentage = (count / len(successful_df) * 100) if len(successful_df) > 0 else 0
                summary += f"• {label}: {count} questions ({percentage:.1f}%)\n"
        
        # Add processing time distribution
        if len(successful_df) > 0:
            summary += "\n⏱️ Processing Time Distribution:\n"
            time_ranges = [
                (0, 1, "Very Fast (0-1s)"),
                (1, 3, "Fast (1-3s)"),
                (3, 5, "Medium (3-5s)"),
                (5, float('inf'), "Slow (5s+)")
            ]
            
            for min_time, max_time, label in time_ranges:
                if max_time == float('inf'):
                    count = len(successful_df[successful_df['processing_time'] >= min_time])
                else:
                    count = len(successful_df[
                        (successful_df['processing_time'] >= min_time) & 
                        (successful_df['processing_time'] < max_time)
                    ])
                percentage = (count / len(successful_df) * 100) if len(successful_df) > 0 else 0
                summary += f"• {label}: {count} questions ({percentage:.1f}%)\n"
        
        # Add error analysis
        if failed_questions > 0:
            summary += f"\n❌ Error Analysis:\n"
            error_counts = results_df[results_df['success'] == False]['error_message'].value_counts()
            for error, count in error_counts.head(5).items():
                summary += f"• {error}: {count} occurrences\n"
        
        # Add top performing questions
        if len(successful_df) > 0:
            summary += f"\n🏆 Top 5 Questions by Confidence:\n"
            top_questions = successful_df.nlargest(5, 'confidence_score')
            for idx, row in top_questions.iterrows():
                question_preview = row['question'][:60] + "..." if len(row['question']) > 60 else row['question']
                summary += f"• {question_preview} (Confidence: {row['confidence_score']:.3f})\n"
        
        return summary


def create_sample_csv(output_path: str = "sample_questions.csv"):
    """Create a sample CSV file with example questions."""
    sample_questions = [
        "What is the production yield mentioned in the documents?",
        "What are the main quality control processes?", 
        "What is the average processing time for manufacturing?",
        "What materials are used in the production process?",
        "What are the safety requirements mentioned?",
        "What is the capacity of the manufacturing line?",
        "What quality metrics are tracked?",
        "What is the maintenance schedule?",
        "What are the operating temperatures?",
        "What certifications are required?"
    ]
    
    df = pd.DataFrame({
        'id': range(1, len(sample_questions) + 1),
        'question': sample_questions,
        'category': ['production', 'quality', 'process', 'materials', 'safety', 
                    'capacity', 'metrics', 'maintenance', 'operations', 'compliance']
    })
    
    df.to_csv(output_path, index=False)
    print(f"📝 Sample CSV created: {output_path}")
    return output_path


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Evaluate questions from CSV using Manufacturing RAG Agent")
    
    parser.add_argument(
        "input_csv",
        nargs='?',
        help="Path to input CSV file containing questions"
    )
    
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create a sample CSV file with example questions"
    )
    
    parser.add_argument(
        "--output-csv",
        "-o",
        help="Path to output CSV file for results (default: input_file_results.csv)"
    )
    
    parser.add_argument(
        "--question-column",
        "-q",
        default="question",
        help="Column name containing questions (default: 'question')"
    )
    
    parser.add_argument(
        "--config",
        "-c",
        default="src/config.yaml",
        help="Path to configuration file (default: src/config.yaml)"
    )
    
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=10,
        help="Number of questions to process in each batch (default: 10)"
    )
    
    parser.add_argument(
        "--delay",
        "-d",
        type=float,
        default=1.0,
        help="Delay between batches in seconds (default: 1.0)"
    )
    
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip generating summary file"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Handle create sample option
    if args.create_sample:
        sample_path = args.input_csv if args.input_csv else "sample_questions.csv"
        create_sample_csv(sample_path)
        print("\n🚀 To run evaluation:")
        print(f"python {sys.argv[0]} {sample_path}")
        return
    
    # Validate input file
    if not args.input_csv:
        print("❌ Please provide an input CSV file or use --create-sample to create one")
        parser.print_help()
        sys.exit(1)
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Validate input file
        if not os.path.exists(args.input_csv):
            print(f"❌ Input CSV file not found: {args.input_csv}")
            sys.exit(1)
        
        # Generate output path if not provided
        if not args.output_csv:
            input_path = Path(args.input_csv)
            args.output_csv = str(input_path.parent / f"{input_path.stem}_results.csv")
        
        print(f"🏭 Manufacturing RAG Agent - CSV Evaluation")
        print(f"Input: {args.input_csv}")
        print(f"Output: {args.output_csv}")
        print(f"Question Column: {args.question_column}")
        print(f"Config: {args.config}")
        print("-" * 50)
        
        # Initialize evaluator
        print("🚀 Initializing RAG system...")
        evaluator = CSVEvaluator(args.config)
        
        if not evaluator.initialize_system():
            print("❌ Failed to initialize RAG system")
            sys.exit(1)
        
        print("✅ RAG system initialized successfully")
        
        # Load questions
        print(f"📄 Loading questions from {args.input_csv}...")
        questions_df = evaluator.load_questions_csv(args.input_csv, args.question_column)
        print(f"✅ Loaded {len(questions_df)} questions")
        
        # Evaluate questions
        print("🔍 Starting evaluation...")
        start_time = time.time()
        
        results_df = evaluator.evaluate_questions(
            questions_df,
            question_column=args.question_column,
            batch_size=args.batch_size,
            delay_between_batches=args.delay
        )
        
        total_time = time.time() - start_time
        
        # Save results
        print(f"💾 Saving results to {args.output_csv}...")
        summary_path = evaluator.save_results(
            results_df, 
            args.output_csv,
            include_summary=not args.no_summary
        )
        
        # Print final summary
        successful = len(results_df[results_df['success'] == True])
        success_rate = (successful / len(results_df) * 100) if len(results_df) > 0 else 0
        
        print("\n" + "=" * 50)
        print("🎉 Evaluation Complete!")
        print(f"📊 Results: {successful}/{len(results_df)} questions answered successfully ({success_rate:.1f}%)")
        print(f"⏱️ Total time: {total_time:.2f} seconds")
        print(f"💾 Results saved to: {args.output_csv}")
        
        if not args.no_summary:
            print(f"📋 Summary saved to: {summary_path}")
        
        print("\n🔍 Quick Preview of Results:")
        if len(results_df) > 0:
            preview_df = results_df[['question', 'answer', 'success', 'confidence_score']].head(3)
            for idx, row in preview_df.iterrows():
                status = "✅" if row['success'] else "❌"
                conf = f"({row['confidence_score']:.2f})" if row['success'] else ""
                question_preview = row['question'][:40] + "..." if len(row['question']) > 40 else row['question']
                answer_preview = str(row['answer'])[:60] + "..." if len(str(row['answer'])) > 60 else str(row['answer'])
                print(f"{status} Q: {question_preview}")
                print(f"   A: {answer_preview} {conf}")
                print()
        
    except KeyboardInterrupt:
        print("\n🛑 Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()