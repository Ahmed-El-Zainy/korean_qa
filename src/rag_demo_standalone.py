import streamlit as st
import os
import sys
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import requests
import json
import time

# Load environment variables
load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


try:
    from logger.custom_logger import CustomLoggerTracker
    custom_log = CustomLoggerTracker()
    logger = custom_log.get_logger("rag_demo_standalone")

except ImportError:
    # Fallback to standard logging if custom logger not available
    logger = logging.getLogger("rag_demo_standalone")


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

class EmbeddingSystem:
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
        self.client = EmbeddingSystem(api_key)
        self.vector_store = SimpleVectorStore()
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
        logger.info(f"Successfully added {len(documents)} documents")
    
    def query(self, query: str, top_k: int = 5, use_reranking: bool = True) -> RAGResult:
        """Query the RAG system."""
        start_time = time.time()
        
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

def run_streamlit_app():
    """Run the Streamlit web application."""
    st.set_page_config(page_title="RAG Demo - Korean QA", page_icon="🤖", layout="wide")
    
    st.title("🤖 Korean RAG Demo with SiliconFlow")
    st.markdown("*Retrieval-Augmented Generation for Korean Manufacturing Q&A*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input(
            "SiliconFlow API Key",
            value=os.getenv("SILICONFLOW_API_KEY", ""),
            type="password",
            help="Enter your SiliconFlow API key"
        )
        
        use_reranking = st.checkbox("Use Reranking", value=True, help="Use reranking for better results")
        top_k = st.slider("Top K Results", min_value=1, max_value=10, value=3)
        
        if st.button("Initialize RAG System"):
            if not api_key:
                st.error("Please provide SiliconFlow API key")
            else:
                with st.spinner("Initializing RAG system..."):
                    try:
                        rag_system = RAGSystem(api_key)
                        rag_system.add_documents(SAMPLE_DOCUMENTS)
                        st.session_state['rag_system'] = rag_system
                        st.success("RAG system initialized successfully!")
                    except Exception as e:
                        st.error(f"Failed to initialize RAG system: {e}")
    
    # Main interface
    if 'rag_system' in st.session_state:
        st.header("💬 Ask Questions")
        
        # Sample questions
        sample_questions = [
            "TAB S10 도장 공정 수율이 어떻게 되나요?",
            "도장 라인의 불량률과 주요 원인은?",
            "품질 검사는 어떻게 진행되나요?",
            "예방 보전 계획에 대해 알려주세요"
        ]
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input("질문을 입력하세요:", placeholder="예: TAB S10 수율이 어떻게 되나요?")
        
        with col2:
            st.markdown("**샘플 질문:**")
            for i, sample in enumerate(sample_questions):
                if st.button(f"Q{i+1}", key=f"sample_{i}", help=sample):
                    st.rerun()
        
        if query:
            with st.spinner("Searching and generating answer..."):
                try:
                    result = st.session_state['rag_system'].query(
                        query, 
                        top_k=top_k, 
                        use_reranking=use_reranking
                    )
                    
                    # Display results
                    st.header("📋 Answer")
                    st.write(result.answer)
                    
                    st.header("📄 Relevant Documents")
                    for i, doc in enumerate(result.relevant_documents):
                        with st.expander(f"Document {i+1} - {doc.metadata.get('source', 'Unknown')}"):
                            st.write(doc.content)
                    
                    # Stats
                    st.sidebar.metric("Processing Time", f"{result.processing_time:.2f}s")
                    st.sidebar.metric("Documents Found", len(result.relevant_documents))
                    
                except Exception as e:
                    st.error(f"Query failed: {e}")
    else:
        st.info("👈 Please initialize the RAG system using the sidebar")
        
        # Show sample documents
        st.header("📚 Sample Documents")
        st.markdown("The system includes these sample manufacturing documents:")
        for i, doc in enumerate(SAMPLE_DOCUMENTS, 1):
            st.markdown(f"**{i}.** {doc}")

def run_cli_demo():
    """Run command line interface demo."""
    print("🤖 Korean RAG Demo - CLI Mode")
    print("=" * 50)
    
    # Get API key
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        api_key = input("Enter your SiliconFlow API key: ")
    
    if not api_key:
        print("❌ API key is required")
        return
    
    try:
        # Initialize RAG system
        print("🔄 Initializing RAG system...")
        rag_system = RAGSystem(api_key)
        rag_system.add_documents(SAMPLE_DOCUMENTS)
        print("✅ RAG system ready!")
        
        # Interactive loop
        while True:
            print("\n" + "-" * 50)
            query = input("질문을 입력하세요 (종료하려면 'quit'): ")
            
            if query.lower() in ['quit', 'exit', '종료']:
                break
            
            if not query.strip():
                continue
            
            print("🔍 Searching...")
            result = rag_system.query(query, top_k=3, use_reranking=True)
            
            print(f"\n📋 답변:")
            print(result.answer)
            
            print(f"\n📄 관련 문서들:")
            for i, doc in enumerate(result.relevant_documents, 1):
                print(f"{i}. {doc.content}")
            
            print(f"\n⏱️  처리 시간: {result.processing_time:.2f}초")
    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Check if running in streamlit context
    try:
        # This will raise an exception if not in streamlit context
        st.session_state
        run_streamlit_app()
    except:
        # Run CLI version
        run_cli_demo()