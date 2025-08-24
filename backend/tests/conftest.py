"""Shared pytest fixtures for testing the RAG system"""
import pytest
import tempfile
import shutil
import os
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk


class MockConfig:
    """Mock configuration for testing"""
    def __init__(self, temp_dir):
        self.CHUNK_SIZE = 800
        self.CHUNK_OVERLAP = 100
        self.CHROMA_PATH = temp_dir
        self.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        self.MAX_RESULTS = 5
        self.ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', 'test-key')
        self.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
        self.MAX_HISTORY = 10


@pytest.fixture
def temp_dir():
    """Create and cleanup temporary directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture  
def mock_config(temp_dir):
    """Create mock configuration for testing"""
    return MockConfig(temp_dir)


@pytest.fixture
def mock_anthropic():
    """Mock Anthropic client for testing"""
    with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Default response for simple queries
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_content = Mock()
        mock_content.text = "Test response from Claude"
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response
        
        yield mock_client


@pytest.fixture
def rag_system(mock_config, mock_anthropic):
    """Create RAG system for testing"""
    return RAGSystem(mock_config)


@pytest.fixture
def sample_course():
    """Sample course data for testing"""
    return Course(
        title="Introduction to Machine Learning",
        instructor="Dr. Jane Smith",
        course_link="https://example.com/ml-course",
        lessons=[
            Lesson(1, "ML Basics", "https://example.com/ml-course/lesson-1"),
            Lesson(2, "Supervised Learning", "https://example.com/ml-course/lesson-2"),
            Lesson(3, "Unsupervised Learning", "https://example.com/ml-course/lesson-3")
        ]
    )


@pytest.fixture
def sample_course_chunks():
    """Sample course chunks for testing"""
    return [
        CourseChunk(
            content="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            course_title="Introduction to Machine Learning",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Supervised learning uses labeled training data to learn a mapping function from input variables to output variables.",
            course_title="Introduction to Machine Learning", 
            lesson_number=2,
            chunk_index=0
        ),
        CourseChunk(
            content="Unsupervised learning finds hidden patterns in data without using labeled examples.",
            course_title="Introduction to Machine Learning",
            lesson_number=3,
            chunk_index=0
        )
    ]


@pytest.fixture
def populated_rag_system(rag_system, sample_course, sample_course_chunks):
    """RAG system populated with test data"""
    rag_system.vector_store.add_course_metadata(sample_course)
    rag_system.vector_store.add_course_content(sample_course_chunks)
    return rag_system


def create_test_app():
    """Create a test FastAPI app without static file mounting"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    
    # Import app components but avoid static file mounting
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    from config import config
    from rag_system import RAGSystem
    
    # Define Pydantic models locally to avoid importing from app.py
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class SourceWithLink(BaseModel):
        name: str
        link: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[SourceWithLink]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    class SessionClearResponse(BaseModel):
        success: bool
        message: str
    
    # Create test app
    app = FastAPI(title="Test RAG System")
    
    # Add middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Create test RAG system
    temp_dir = tempfile.mkdtemp()
    test_config = MockConfig(temp_dir)
    
    with patch('ai_generator.anthropic.Anthropic'):
        rag_system = RAGSystem(test_config)
    
    # Define API endpoints inline to avoid import issues
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()
            
            answer, sources = rag_system.query(request.query, session_id)
            
            formatted_sources = []
            for source in sources:
                if isinstance(source, dict):
                    formatted_sources.append(SourceWithLink(
                        name=source.get('name', ''),
                        link=source.get('link')
                    ))
                else:
                    formatted_sources.append(SourceWithLink(name=str(source)))
            
            return QueryResponse(
                answer=answer,
                sources=formatted_sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.delete("/api/sessions/{session_id}/clear", response_model=SessionClearResponse)
    async def clear_session(session_id: str):
        try:
            rag_system.session_manager.clear_session(session_id)
            return SessionClearResponse(
                success=True,
                message=f"Session {session_id} cleared successfully"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def root():
        return {"message": "RAG System API"}
    
    return app, temp_dir


@pytest.fixture
def test_client():
    """Create FastAPI test client"""
    app, temp_dir = create_test_app()
    client = TestClient(app)
    yield client
    # Cleanup temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_tool_response():
    """Mock tool response from Anthropic"""
    mock_tool_content = Mock()
    mock_tool_content.type = "tool_use"
    mock_tool_content.name = "search_course_content"
    mock_tool_content.input = {"query": "machine learning"}
    mock_tool_content.id = "tool-123"
    
    return mock_tool_content


@pytest.fixture
def mock_anthropic_tool_flow():
    """Mock complete tool calling flow for Anthropic"""
    with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # First response: tool use
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "machine learning"}
        mock_tool_content.id = "tool-123"
        
        initial_response = Mock()
        initial_response.stop_reason = "tool_use"
        initial_response.content = [mock_tool_content]
        
        # Second response: final answer
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_content = Mock()
        final_content.text = "Based on the course content, machine learning is a subset of AI that focuses on learning from data."
        final_response.content = [final_content]
        
        mock_client.messages.create.side_effect = [initial_response, final_response]
        
        yield mock_client


@pytest.fixture(autouse=True)
def suppress_warnings():
    """Suppress warnings during tests"""
    import warnings
    warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)