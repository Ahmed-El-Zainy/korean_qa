
import gradio as gr
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
import requests
import json
import time
import threading

# Load environment variables
load_dotenv()

try:
    from logger.custom_logger import CustomLoggerTracker
    custom_log = CustomLoggerTracker()
    logger = custom_log.get_logger("gradio_demo")

except ImportError:
    # Fallback to standard logging if custom logger not available
    logger = logging.getLogger("gradio_demo")



@dataclass
class Document:
    """Document structure for RAG system."""
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

@dataclass
class RAGResult:
    """RAG query result."""
    query: str
    answer: str
    relevant_documents: List[Document]
    processing_time: float

class SimpleVectorStore:
    """Simple in-memory vector store for demonstration."""
    
    def __init__(self):
        self.documents: List[Document] = []
        self.embeddings: List[List[float]] = []
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store."""
        self.documents.extend(documents)
        for doc in documents:
            if doc.embedding:
                self.embeddings.append(doc.embedding)
    
    def similarity_search(self, query_embedding: List[float], top_k: int = 5) -> List[Document]:
        """Find most similar documents using cosine similarity."""
        if not self.embeddings or not query_embedding:
            return []
        
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            if doc_embedding:
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                similarities.append((similarity, self.documents[i]))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in similarities[:top_k]]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)

class SiliconFlowClient:
    """SiliconFlow API client for embeddings and chat completion."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.siliconflow.cn/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def generate_embeddings(self, texts: List[str], 
                          model: str = "BAAI/bge-large-zh-v1.5") -> List[List[float]]:
        """Generate embeddings for texts."""
        try:
            payload = {
                "model": model,
                "input": texts,
                "encoding_format": "float"
            }
            
            response = requests.post(
                f"{self.base_url}/embeddings",
                json=payload,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return [item['embedding'] for item in data.get('data', [])]
            else:
                logger.error(f"Embedding API error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []
    
    def rerank_documents(self, query: str, documents: List[str], 
                        model: str = "BAAI/bge-reranker-large",
                        top_k: int = 5) -> List[Dict]:
        """Rerank documents based on query relevance."""
        try:
            payload = {
                "model": model,
                "query": query,
                "documents": documents,
                "top_k": top_k,
                "return_documents": True
            }
            
            response = requests.post(
                f"{self.base_url}/rerank",
                json=payload,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('results', [])
            else:
                logger.error(f"Rerank API error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return []
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       model: str = "Qwen/Qwen2.5-7B-Instruct") -> str:
        """Generate chat completion."""
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self.headers,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content']
            else:
                logger.error(f"Chat completion API error: {response.status_code} - {response.text}")
                return "죄송합니다. 응답을 생성할 수 없습니다."
                
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            return "죄송합니다. 응답을 생성할 수 없습니다."

class RAGSystem:
    """Complete RAG system using SiliconFlow."""
    
    def __init__(self, api_key: str):
        self.client = SiliconFlowClient(api_key)
        self.vector_store = SimpleVectorStore()
        self.initialized = False
        logger.info("RAG System initialized")
    
    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """Add documents to the RAG system."""
        if not metadatas:
            metadatas = [{"source": f"doc_{i}"} for i in range(len(texts))]
        
        logger.info(f"Adding {len(texts)} documents...")
        
        # Generate embeddings
        embeddings = self.client.generate_embeddings(texts)
        
        # Create document objects
        documents = []
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            embedding = embeddings[i] if i < len(embeddings) else None
            doc = Document(content=text, metadata=metadata, embedding=embedding)
            documents.append(doc)
        
        # Add to vector store
        self.vector_store.add_documents(documents)
        self.initialized = True
        logger.info(f"Successfully added {len(documents)} documents")
    
    def query(self, query: str, top_k: int = 5, use_reranking: bool = True) -> RAGResult:
        """Query the RAG system."""
        start_time = time.time()
        
        if not self.initialized:
            return RAGResult(
                query=query,
                answer="시스템이 초기화되지 않았습니다.",
                relevant_documents=[],
                processing_time=time.time() - start_time
            )
        
        # Generate query embedding
        query_embeddings = self.client.generate_embeddings([query])
        query_embedding = query_embeddings[0] if query_embeddings else []
        
        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return RAGResult(
                query=query,
                answer="죄송합니다. 쿼리를 처리할 수 없습니다.",
                relevant_documents=[],
                processing_time=time.time() - start_time
            )
        
        # Find similar documents
        similar_docs = self.vector_store.similarity_search(query_embedding, top_k * 2)
        
        if not similar_docs:
            return RAGResult(
                query=query,
                answer="관련 문서를 찾을 수 없습니다.",
                relevant_documents=[],
                processing_time=time.time() - start_time
            )
        
        # Optional reranking
        if use_reranking and len(similar_docs) > top_k:
            doc_texts = [doc.content for doc in similar_docs]
            rerank_results = self.client.rerank_documents(query, doc_texts, top_k=top_k)
            
            if rerank_results:
                # Reorder documents based on reranking
                reranked_docs = []
                for result in rerank_results:
                    doc_idx = result.get('index', 0)
                    if doc_idx < len(similar_docs):
                        reranked_docs.append(similar_docs[doc_idx])
                similar_docs = reranked_docs
        
        # Limit to top_k
        relevant_docs = similar_docs[:top_k]
        
        # Generate answer using context
        context = "\n\n".join([doc.content for doc in relevant_docs])
        answer = self._generate_answer(query, context)
        
        processing_time = time.time() - start_time
        
        return RAGResult(
            query=query,
            answer=answer,
            relevant_documents=relevant_docs,
            processing_time=processing_time
        )
    
    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using context and query."""
        system_prompt = """당신은 한국어로 답변하는 도움이 되는 어시스턴트입니다. 
주어진 컨텍스트를 바탕으로 질문에 정확하고 유용한 답변을 제공해주세요.
컨텍스트에 정보가 없으면 '주어진 정보로는 답변하기 어렵습니다'라고 말해주세요."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"컨텍스트:\n{context}\n\n질문: {query}"}
        ]
        
        return self.client.chat_completion(messages)

# Sample Korean manufacturing data
SAMPLE_DOCUMENTS = [
    "TAB S10 도장 공정의 수율은 현재 95.2%입니다. 목표 수율 94%를 상회하고 있으며, 지난달 대비 1.3% 향상되었습니다.",
    "도장 라인에서 불량률이 4.8% 발생하고 있습니다. 주요 불량 원인은 온도 편차(45%)와 습도 변화(30%)입니다.",
    "S10 모델의 전체 생산 수율은 89.5%로 목표치 88%를 상회하고 있습니다. 월간 생산량은 15,000대입니다.",
    "도장 라인의 온도는 22±2℃, 습도는 45±5%로 유지되어야 합니다. 현재 자동 제어 시스템으로 관리되고 있습니다.",
    "품질관리 부서에서는 매일 3회 샘플링 검사를 실시하고 있습니다. 검사 항목은 색상, 광택, 두께입니다.",
    "예방 보전 계획에 따라 도장 설비는 주 1회 정기 점검을 실시합니다. 다음 정기 보전은 다음 주 화요일입니다.",
    "신규 도장 재료 적용 후 접착력이 15% 향상되었습니다. 비용은 10% 증가했지만 품질 개선 효과가 큽니다.",
    "작업자 교육은 월 2회 실시되며, 안전교육과 품질교육을 포함합니다. 교육 참석률은 98.5%입니다."
]

# Global RAG system instance
rag_system = None

def initialize_rag_system(api_key: str, progress=gr.Progress()) -> Tuple[str, str]:
    """Initialize the RAG system with the provided API key."""
    global rag_system
    
    if not api_key:
        return "❌ API 키를 입력해주세요.", "Not initialized"
    
    try:
        progress(0.1, desc="RAG 시스템 초기화 중...")
        
        # Initialize RAG system
        rag_system = RAGSystem(api_key)
        
        progress(0.5, desc="문서 임베딩 생성 중...")
        
        # Add sample documents
        rag_system.add_documents(SAMPLE_DOCUMENTS)
        
        progress(1.0, desc="완료!")
        
        return "✅ RAG 시스템이 성공적으로 초기화되었습니다!", "Ready"
        
    except Exception as e:
        logger.error(f"RAG system initialization failed: {e}")
        return f"❌ 초기화 실패: {str(e)}", "Error"

def query_rag_system(query: str, top_k: int, use_reranking: bool, progress=gr.Progress()) -> Tuple[str, str, str]:
    """Query the RAG system and return results."""
    global rag_system
    
    if not rag_system or not rag_system.initialized:
        return "시스템이 초기화되지 않았습니다. 먼저 API 키를 입력하고 초기화해주세요.", "", ""
    
    if not query.strip():
        return "질문을 입력해주세요.", "", ""
    
    try:
        progress(0.2, desc="쿼리 임베딩 생성 중...")
        progress(0.5, desc="유사 문서 검색 중...")
        progress(0.7, desc="문서 재순위화 중...")
        progress(0.9, desc="답변 생성 중...")
        
        # Query the system
        result = rag_system.query(query, top_k=top_k, use_reranking=use_reranking)
        
        # Format relevant documents
        docs_text = ""
        for i, doc in enumerate(result.relevant_documents, 1):
            docs_text += f"**문서 {i}** ({doc.metadata.get('source', 'Unknown')})\n"
            docs_text += f"{doc.content}\n\n"
        
        # Format statistics
        stats_text = f"🕐 처리 시간: {result.processing_time:.2f}초\n"
        stats_text += f"📄 찾은 문서 수: {len(result.relevant_documents)}\n"
        stats_text += f"🔄 재순위화 사용: {'Yes' if use_reranking else 'No'}\n"
        stats_text += f"📊 Top-K: {top_k}"
        
        progress(1.0, desc="완료!")
        
        return result.answer, docs_text, stats_text
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return f"❌ 쿼리 처리 실패: {str(e)}", "", ""

def create_gradio_interface():
    """Create and configure the Gradio interface."""
    
    # Custom CSS for better styling
    css = """
    .container {
        max-width: 1200px;
        margin: 0 auto;
    }
    .header {
        text-align: center;
        margin-bottom: 20px;
    }
    .sample-questions {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(css=css, title="Korean RAG Demo", theme=gr.themes.Soft()) as demo:
        
        gr.HTML("""
            <div class="header">
                <h1>🤖 Korean RAG Demo with SiliconFlow</h1>
                <p><em>Retrieval-Augmented Generation for Korean Manufacturing Q&A</em></p>
            </div>
        """)
        
        with gr.Tab("🏠 Main"):
            with gr.Row():
                with gr.Column(scale=2):
                    # Configuration Section
                    gr.Markdown("## ⚙️ Configuration")
                    
                    api_key_input = gr.Textbox(
                        label="SiliconFlow API Key",
                        placeholder="Enter your SiliconFlow API key...",
                        type="password",
                        value=os.getenv("SILICONFLOW_API_KEY", "")
                    )
                    
                    with gr.Row():
                        init_button = gr.Button("🚀 Initialize RAG System", variant="primary")
                        status_output = gr.Textbox(label="Status", value="Not initialized", interactive=False)
                    
                    init_output = gr.Markdown()
                    
                    # Settings
                    with gr.Accordion("🔧 Advanced Settings", open=False):
                        top_k_slider = gr.Slider(
                            minimum=1, maximum=10, value=3, step=1,
                            label="Top-K Results"
                        )
                        reranking_checkbox = gr.Checkbox(
                            label="Use Reranking", value=True
                        )
                
                with gr.Column(scale=3):
                    # Query Section
                    gr.Markdown("## 💬 Ask Questions")
                    
                    # Sample questions
                    gr.HTML("""
                        <div class="sample-questions">
                            <h4>📝 Sample Questions:</h4>
                            <ul>
                                <li>TAB S10 도장 공정 수율이 어떻게 되나요?</li>
                                <li>도장 라인의 불량률과 주요 원인은?</li>
                                <li>품질 검사는 어떻게 진행되나요?</li>
                                <li>예방 보전 계획에 대해 알려주세요</li>
                            </ul>
                        </div>
                    """)
                    
                    query_input = gr.Textbox(
                        label="Your Question",
                        placeholder="예: TAB S10 수율이 어떻게 되나요?",
                        lines=2
                    )
                    
                    # Quick question buttons
                    with gr.Row():
                        q1_btn = gr.Button("수율 질문", size="sm")
                        q2_btn = gr.Button("불량률 질문", size="sm")
                        q3_btn = gr.Button("품질검사 질문", size="sm")
                        q4_btn = gr.Button("보전계획 질문", size="sm")
                    
                    query_button = gr.Button("🔍 Search & Answer", variant="primary", size="lg")
            
            # Results Section
            gr.Markdown("## 📋 Results")
            
            with gr.Row():
                with gr.Column(scale=2):
                    answer_output = gr.Markdown(label="Answer")
                
                with gr.Column(scale=1):
                    stats_output = gr.Markdown(label="Statistics")
            
            # Documents Section
            gr.Markdown("## 📄 Relevant Documents")
            documents_output = gr.Markdown()
        
        with gr.Tab("📚 Sample Documents"):
            gr.Markdown("## 📚 Sample Manufacturing Documents")
            gr.Markdown("The system includes these Korean manufacturing documents:")
            
            sample_docs_text = ""
            for i, doc in enumerate(SAMPLE_DOCUMENTS, 1):
                sample_docs_text += f"**{i}.** {doc}\n\n"
            
            gr.Markdown(sample_docs_text)
        
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
                ## 🤖 About Korean RAG Demo
                
                This demo showcases a **Retrieval-Augmented Generation (RAG)** system specifically designed for Korean manufacturing Q&A.
                
                ### 🔧 Features:
                - **Korean Language Support**: Optimized for Korean manufacturing terminology
                - **SiliconFlow Integration**: Uses BAAI embedding and reranking models
                - **Real-time Processing**: Fast query processing with similarity search
                - **Document Reranking**: Improves relevance with optional reranking
                - **Interactive Interface**: User-friendly Gradio interface
                
                ### 🚀 How it Works:
                1. **Document Embedding**: Sample documents are converted to vector embeddings
                2. **Query Processing**: Your question is also embedded using the same model
                3. **Similarity Search**: Find most relevant documents using cosine similarity
                4. **Reranking** (optional): Reorder results for better relevance
                5. **Answer Generation**: Generate contextual answers using retrieved documents
                
                ### 🔑 Requirements:
                - SiliconFlow API key with access to:
                  - `BAAI/bge-large-zh-v1.5` (embedding model)
                  - `BAAI/bge-reranker-large` (reranking model)
                  - `Qwen/Qwen2.5-7B-Instruct` (chat model)
                
                ### 📊 Sample Data:
                The demo includes 8 Korean manufacturing documents covering:
                - Production yield rates
                - Quality control processes
                - Defect analysis
                - Maintenance schedules
                - Training programs
            """)
        
        # Event handlers
        init_button.click(
            fn=initialize_rag_system,
            inputs=[api_key_input],
            outputs=[init_output, status_output]
        )
        
        query_button.click(
            fn=query_rag_system,
            inputs=[query_input, top_k_slider, reranking_checkbox],
            outputs=[answer_output, documents_output, stats_output]
        )
        
        # Sample question button handlers
        q1_btn.click(lambda: "TAB S10 도장 공정 수율이 어떻게 되나요?", outputs=query_input)
        q2_btn.click(lambda: "도장 라인의 불량률과 주요 원인은?", outputs=query_input)
        q3_btn.click(lambda: "품질 검사는 어떻게 진행되나요?", outputs=query_input)
        q4_btn.click(lambda: "예방 보전 계획에 대해 알려주세요", outputs=query_input)
        
        # Allow Enter key to trigger search
        query_input.submit(
            fn=query_rag_system,
            inputs=[query_input, top_k_slider, reranking_checkbox],
            outputs=[answer_output, documents_output, stats_output]
        )
    
    return demo

def main():
    """Main function to run the Gradio app."""
    demo = create_gradio_interface()
    
    # Launch the app
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True to create public link
        debug=True,             # Enable debug mode
        show_error=True         # Show detailed errors
    )

if __name__ == "__main__":
    main()