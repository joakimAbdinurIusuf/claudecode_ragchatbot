"""Test configuration and fixtures"""
import os
import sys
import pytest
import tempfile
import shutil
from typing import Dict, Any, List

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import config
from models import Course, Lesson, CourseChunk
from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, ToolManager

class TestConfig:
    """Test configuration class"""
    def __init__(self):
        self.CHROMA_PATH = tempfile.mkdtemp()  # Temporary directory for test DB
        self.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        self.MAX_RESULTS = 5
        self.ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', 'test-key')
        self.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

@pytest.fixture
def test_config():
    """Provide test configuration"""
    return TestConfig()

@pytest.fixture
def mock_vector_store(test_config):
    """Create a VectorStore for testing"""
    store = VectorStore(test_config.CHROMA_PATH, test_config.EMBEDDING_MODEL, test_config.MAX_RESULTS)
    yield store
    # Cleanup
    shutil.rmtree(test_config.CHROMA_PATH, ignore_errors=True)

@pytest.fixture
def populated_vector_store(test_config):
    """Create a VectorStore with test data"""
    store = VectorStore(test_config.CHROMA_PATH, test_config.EMBEDDING_MODEL, test_config.MAX_RESULTS)
    
    # Add test course data
    test_course = Course(
        title="Introduction to Machine Learning",
        instructor="Dr. Smith",
        course_link="https://example.com/ml-course",
        lessons=[
            Lesson(
                lesson_number=1,
                title="Introduction to ML",
                lesson_link="https://example.com/ml-course/lesson-1"
            ),
            Lesson(
                lesson_number=2,
                title="Supervised Learning",
                lesson_link="https://example.com/ml-course/lesson-2"
            )
        ]
    )
    
    # Add course metadata
    store.add_course_metadata(test_course)
    
    # Add some test content
    test_chunks = [
        CourseChunk(
            content="Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions.",
            course_title="Introduction to Machine Learning",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Supervised learning involves training models on labeled data to make predictions.",
            course_title="Introduction to Machine Learning", 
            lesson_number=2,
            chunk_index=1
        )
    ]
    
    store.add_course_content(test_chunks)
    
    yield store
    # Cleanup
    shutil.rmtree(test_config.CHROMA_PATH, ignore_errors=True)

class MockVectorStore:
    """Mock VectorStore for testing error conditions"""
    def __init__(self, should_error: bool = False, return_empty: bool = False):
        self.should_error = should_error
        self.return_empty = return_empty
    
    def search(self, query: str, course_name=None, lesson_number=None, limit=None) -> SearchResults:
        if self.should_error:
            return SearchResults.empty("Mock search error")
        
        if self.return_empty:
            return SearchResults(documents=[], metadata=[], distances=[])
        
        # Return mock successful results
        return SearchResults(
            documents=["Mock search result for: " + query],
            metadata=[{"course_title": "Mock Course", "lesson_number": 1}],
            distances=[0.1]
        )
    
    def get_lesson_link(self, course_title: str, lesson_number: int) -> str:
        return "https://example.com/mock-lesson"

@pytest.fixture
def mock_error_vector_store():
    """Vector store that returns errors"""
    return MockVectorStore(should_error=True)

@pytest.fixture
def mock_empty_vector_store():
    """Vector store that returns empty results"""
    return MockVectorStore(return_empty=True)

@pytest.fixture  
def mock_success_vector_store():
    """Vector store that returns successful results"""
    return MockVectorStore()