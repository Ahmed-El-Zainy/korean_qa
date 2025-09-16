import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import requests
import json
import os 
import os 
import sys 
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Setup logging
try:
    from logger.custom_logger import CustomLoggerTracker
    custom_log = CustomLoggerTracker()
    logger = custom_log.get_logger("groq_client")

except ImportError:
    # Fallback to standard logging if custom logger not available
    logger = logging.getLogger("groq_client")



@dataclass
class LLMResponse:
    """Response from LLM generation."""
    text: str
    model_name: str
    processing_time: float
    token_count: int
    success: bool
    error_message: Optional[str] = None
    finish_reason: Optional[str] = None


class GroqClient:
    """
    Groq API client for fast LLM inference.
    
    This client provides high-speed inference using Groq's LPU architecture
    with support for various models like Llama, Mixtral, and Gemma.
    """
    
    def __init__(self, api_key: str, base_url: str = "https://api.groq.com/openai/v1"):
        """
        Initialize the Groq client.
        
        Args:
            api_key: Groq API key
            base_url: Base URL for Groq API
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
        
        # Rate limiting
        self.max_requests_per_minute = 30
        self.request_timestamps = []
        
        logger.info(f"Groq client initialized with base URL: {base_url}")
    
    def generate_response(self, messages: List[Dict[str, str]], 
                         model: str = "openai/gpt-oss-120b",
                         max_tokens: int = 1024,
                         temperature: float = 0.1) -> LLMResponse:
        """
        Generate response using Groq LLM.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model name to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            LLMResponse with generated text and metadata
        """
        start_time = time.time()
        
        try:
            # Rate limiting check
            self._check_rate_limit()
            
            # Prepare request payload
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            # Make API request
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=60
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract response text
                choice = data.get('choices', [{}])[0]
                message = choice.get('message', {})
                generated_text = message.get('content', '')
                finish_reason = choice.get('finish_reason', 'unknown')
                
                # Get usage info
                usage = data.get('usage', {})
                token_count = usage.get('total_tokens', 0)
                
                logger.debug(f"Generated response in {processing_time:.2f}s, {token_count} tokens")
                
                return LLMResponse(
                    text=generated_text,
                    model_name=model,
                    processing_time=processing_time,
                    token_count=token_count,
                    success=True,
                    finish_reason=finish_reason
                )
            else:
                error_msg = f"API request failed with status {response.status_code}: {response.text}"
                logger.error(error_msg)
                
                return LLMResponse(
                    text="",
                    model_name=model,
                    processing_time=processing_time,
                    token_count=0,
                    success=False,
                    error_message=error_msg
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"LLM generation failed: {str(e)}"
            logger.error(error_msg)
            
            return LLMResponse(
                text="",
                model_name=model,
                processing_time=processing_time,
                token_count=0,
                success=False,
                error_message=error_msg
            )
    
    def answer_question(self, question: str, context: str, 
                       model: str = "openai/gpt-oss-120b") -> LLMResponse:
        """
        Answer a question based on provided context.
        
        Args:
            question: Question to answer
            context: Context information
            model: Model name to use
            
        Returns:
            LLMResponse with the answer
        """
        # Create system prompt for manufacturing Q&A
        system_prompt = """You are an expert manufacturing analyst assistant. Your task is to answer questions about manufacturing data, processes, and documentation based on the provided context.

Guidelines:
1. Answer questions accurately based only on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Include specific references to data points, measurements, or processes when available
4. Use technical manufacturing terminology appropriately
5. Provide concise but complete answers
6. If asked about trends or comparisons, use the numerical data from the context

Always cite your sources by mentioning the specific document, page, or section where you found the information."""
        
        # Create user prompt with context and question
        user_prompt = f"""Context:
{context}

Question: {question}

Please provide a detailed answer based on the context above. Include specific citations where possible."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.generate_response(messages, model, max_tokens=1024, temperature=0.1)
    
    def summarize_document(self, content: str, 
                          model: str = "openai/gpt-oss-120b") -> LLMResponse:
        system_prompt = """You are an expert at summarizing manufacturing documents. Create concise, informative summaries that capture the key information, data points, and insights from manufacturing documentation."""
        
        user_prompt = f"""Please provide a comprehensive summary of the following manufacturing document content:

{content}

Focus on:
- Key manufacturing processes described
- Important measurements, specifications, or data points
- Quality metrics or performance indicators
- Any issues, recommendations, or conclusions
- Critical dates, locations, or responsible parties

Keep the summary concise but comprehensive."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.generate_response(messages, model, max_tokens=512, temperature=0.1)
    
    def extract_key_information(self, content: str, 
                               model: str = "openai/gpt-oss-120b") -> LLMResponse:
        """
        Extract key information from document content.
        
        Args:
            content: Document content to analyze
            model: Model name to use
            
        Returns:
            LLMResponse with extracted key information
        """
        system_prompt = """You are an expert at extracting key information from manufacturing documents. Identify and extract the most important data points, specifications, processes, and insights."""
        
        user_prompt = f"""Extract the key information from the following manufacturing document content:

{content}

Please organize the extracted information into categories such as:
- Manufacturing Processes
- Quality Metrics
- Specifications/Parameters
- Performance Data
- Issues/Problems
- Recommendations
- Dates and Timelines

Present the information in a structured, easy-to-read format."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.generate_response(messages, model, max_tokens=768, temperature=0.1)
    
    def _check_rate_limit(self):
        """Check and enforce rate limiting."""
        current_time = time.time()
        
        # Remove timestamps older than 1 minute
        self.request_timestamps = [
            ts for ts in self.request_timestamps 
            if current_time - ts < 60
        ]
        
        # Check if we're at the rate limit
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            sleep_time = 60 - (current_time - self.request_timestamps[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        
        # Add current request timestamp
        self.request_timestamps.append(current_time)
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models.
        
        Returns:
            List of available model names
        """
        try:
            response = self.session.get(f"{self.base_url}/models")
            
            if response.status_code == 200:
                data = response.json()
                models = [model['id'] for model in data.get('data', [])]
                return models
            else:
                logger.error(f"Failed to get models: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []
    
    def health_check(self) -> bool:
        """
        Check if the Groq API is accessible.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/models", timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Groq health check failed: {e}")
            return False




class LLMSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = os.getenv('GROQ_API_KEY') or config.get('groq_api_key')
        self.default_model = config.get('llm_model', 'openai/gpt-oss-120b')
        self.max_retries = config.get('max_retries', 3)
        if not self.api_key:
            raise ValueError("Groq API key is required")
        self.client = GroqClient(self.api_key)
        logger.info(f"LLM system initialized with default model: {self.default_model}")
    
    def answer_question(self, question: str, context: str, model: Optional[str] = None) -> str:
        model = model or self.default_model
        for attempt in range(self.max_retries):
            try:
                response = self.client.answer_question(question, context, model)
                if response.success:
                    return response.text
                else:
                    logger.warning(f"LLM generation failed (attempt {attempt + 1}): {response.error_message}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                logger.warning(f"LLM generation error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        # Return fallback response if all attempts failed
        return "I apologize, but I'm unable to generate a response at this time due to technical difficulties. Please try again later."
    
    def summarize_content(self, content: str, model: Optional[str] = None) -> str:
        model = model or self.default_model
        for attempt in range(self.max_retries):
            try:
                response = self.client.summarize_document(content, model)
                if response.success:
                    return response.text
                else:
                    logger.warning(f"Summarization failed (attempt {attempt + 1}): {response.error_message}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)
            except Exception as e:
                logger.warning(f"Summarization error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        return "Unable to generate summary at this time."


if __name__=="__main__":
    logger.info(f"Groq client init ..")
    ## Test code (for demonstration purposes)
    config = {
        'groq_api_key': os.getenv('GROQ_API_KEY'),
        'llm_model': 'openai/gpt-oss-120b',
        'max_retries': 3
    }
    llm_system = LLMSystem(config)
    question = "What is the capital of France?"
    context = "France is a country in Western Europe."
    answer = llm_system.answer_question(question, context)
    logger.info(f"Answer: {answer}")