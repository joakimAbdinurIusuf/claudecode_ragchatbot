"""Integration tests for RAG system end-to-end functionality"""
import pytest
import sys
import os
import tempfile
import shutil
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_system import RAGSystem
from config import config
from test_config import TestConfig, MockVectorStore
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

class TestRAGSystemIntegration:
    """Test RAG system integration and end-to-end functionality"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = MockConfig(self.temp_dir)
    
    def teardown_method(self):
        """Cleanup after each test method"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_rag_system_initialization(self):
        """Test that RAG system initializes properly"""
        rag = RAGSystem(self.test_config)
        
        # Check that all components are initialized
        assert rag.document_processor is not None
        assert rag.vector_store is not None
        assert rag.ai_generator is not None
        assert rag.session_manager is not None
        assert rag.tool_manager is not None
        assert rag.search_tool is not None
        assert rag.outline_tool is not None
        
        # Check tools are registered
        tool_definitions = rag.tool_manager.get_tool_definitions()
        tool_names = [tool["name"] for tool in tool_definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names
        
        print("✓ RAG system initialization test passed")
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_query_without_course_content(self, mock_anthropic):
        """Test query when no course content is available"""
        # Setup mock AI response
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_content = Mock()
        mock_content.text = "I don't have information about that course topic."
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response
        
        # Test
        rag = RAGSystem(self.test_config)
        response, sources = rag.query("What is machine learning?")
        
        assert "don't have information" in response
        assert len(sources) == 0
        print(f"✓ Query without content test passed: {response}")
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_query_with_general_knowledge(self, mock_anthropic):
        """Test query that should use general knowledge instead of search"""
        # Setup mock AI response
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_content = Mock()
        mock_content.text = "Paris is the capital of France."
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response
        
        # Test
        rag = RAGSystem(self.test_config)
        response, sources = rag.query("What is the capital of France?")
        
        assert "Paris" in response
        assert len(sources) == 0  # No search should have been performed
        print(f"✓ General knowledge query test passed: {response}")
    
    @patch('ai_generator.anthropic.Anthropic')  
    def test_query_with_course_content(self, mock_anthropic):
        """Test query when course content is available and search is used"""
        # Setup mock AI responses for tool usage flow
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
        final_content.text = "Based on the course content, machine learning is a subset of AI..."
        final_response.content = [final_content]
        
        mock_client.messages.create.side_effect = [initial_response, final_response]
        
        # Add test course content
        rag = RAGSystem(self.test_config)
        
        # Add test course
        test_course = Course(
            title="Introduction to Machine Learning",
            instructor="Dr. Smith", 
            course_link="https://example.com/ml",
            lessons=[
                Lesson(1, "ML Basics", "https://example.com/ml/1")
            ]
        )
        rag.vector_store.add_course_metadata(test_course)
        
        # Add test content
        test_chunks = [
            CourseChunk(
                content="Machine learning is a subset of artificial intelligence.",
                course_title="Introduction to Machine Learning",
                lesson_number=1,
                chunk_index=0
            )
        ]
        rag.vector_store.add_course_content(test_chunks)
        
        # Test
        response, sources = rag.query("What is machine learning?")
        
        assert "machine learning is a subset of AI" in response.lower()
        print(f"✓ Course content query test passed: {response[:100]}...")
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_session_management(self, mock_anthropic):
        """Test that session management works correctly"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_content = Mock()
        mock_content.text = "Response with session"
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response
        
        # Test
        rag = RAGSystem(self.test_config)
        
        # First query with session
        response1, _ = rag.query("First question", session_id="test-session")
        
        # Second query with same session
        response2, _ = rag.query("Follow up", session_id="test-session")
        
        # Check that history was included in second call
        call_args = mock_client.messages.create.call_args_list[-1]
        system_content = call_args[1]['system']
        
        assert "Previous conversation:" in system_content
        assert "First question" in system_content
        print("✓ Session management test passed")
    
    def test_course_analytics(self):
        """Test getting course analytics"""
        rag = RAGSystem(self.test_config)
        
        # Initially should be empty
        analytics = rag.get_course_analytics()
        assert analytics["total_courses"] == 0
        assert len(analytics["course_titles"]) == 0
        
        # Add a course
        test_course = Course(
            title="Test Course",
            instructor="Test Instructor",
            course_link="https://example.com",
            lessons=[]
        )
        rag.vector_store.add_course_metadata(test_course)
        
        # Check analytics updated
        analytics = rag.get_course_analytics()
        assert analytics["total_courses"] == 1
        assert "Test Course" in analytics["course_titles"]
        
        print(f"✓ Course analytics test passed: {analytics}")

class TestErrorConditions:
    """Test error conditions and edge cases"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = MockConfig(self.temp_dir)
    
    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_anthropic_api_error(self, mock_anthropic):
        """Test handling of Anthropic API errors"""
        # Setup mock to raise exception
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API Error")
        
        # Test
        rag = RAGSystem(self.test_config)
        
        # Should handle exception gracefully
        try:
            response, sources = rag.query("test query")
            # If no exception, check we got some kind of error response
            assert "error" in response.lower() or response == ""
            print("✓ API error handling test passed")
        except Exception as e:
            # Exception propagated - this identifies the issue
            print(f"✗ API error not handled properly: {e}")
            raise
    
    @patch('vector_store.chromadb')
    def test_vector_store_error(self, mock_chromadb):
        """Test handling of vector store errors"""
        # Setup mock to raise exception
        mock_chromadb.PersistentClient.side_effect = Exception("ChromaDB Error")
        
        # Test
        try:
            rag = RAGSystem(self.test_config)
            print("✓ Vector store error handling during initialization")
        except Exception as e:
            print(f"✗ Vector store error not handled: {e}")
            raise

def run_integration_tests():
    """Run all integration tests manually"""
    print("Running RAG System Integration Tests...")
    print("=" * 60)
    
    # Test RAG system
    print("\nTesting RAG System...")
    rag_test = TestRAGSystemIntegration()
    
    print("\n1. Testing RAG system initialization...")
    rag_test.setup_method()
    try:
        rag_test.test_rag_system_initialization()
    finally:
        rag_test.teardown_method()
    
    print("\n2. Testing query without course content...")
    rag_test.setup_method()
    try:
        rag_test.test_query_without_course_content()
    finally:
        rag_test.teardown_method()
    
    print("\n3. Testing general knowledge query...")
    rag_test.setup_method()
    try:
        rag_test.test_query_with_general_knowledge()
    finally:
        rag_test.teardown_method()
    
    print("\n4. Testing query with course content...")
    rag_test.setup_method()
    try:
        rag_test.test_query_with_course_content()
    finally:
        rag_test.teardown_method()
    
    print("\n5. Testing session management...")
    rag_test.setup_method()
    try:
        rag_test.test_session_management()
    finally:
        rag_test.teardown_method()
    
    print("\n6. Testing course analytics...")
    rag_test.setup_method()
    try:
        rag_test.test_course_analytics()
    finally:
        rag_test.teardown_method()
    
    # Test error conditions
    print("\nTesting Error Conditions...")
    error_test = TestErrorConditions()
    
    print("\n7. Testing Anthropic API error handling...")
    error_test.setup_method()
    try:
        error_test.test_anthropic_api_error()
    finally:
        error_test.teardown_method()
    
    print("\n" + "=" * 60)
    print("Integration tests completed!")

if __name__ == "__main__":
    run_integration_tests()