"""Tests for CourseSearchTool and related functionality"""
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from test_config import (
    mock_vector_store, populated_vector_store, mock_error_vector_store,
    mock_empty_vector_store, mock_success_vector_store
)

class TestCourseSearchTool:
    """Test CourseSearchTool execute method and functionality"""
    
    def test_tool_definition_structure(self, mock_success_vector_store):
        """Test that tool definition has correct structure"""
        tool = CourseSearchTool(mock_success_vector_store)
        definition = tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["query"]
    
    def test_execute_basic_query_success(self, mock_success_vector_store):
        """Test basic query execution with successful results"""
        tool = CourseSearchTool(mock_success_vector_store)
        result = tool.execute(query="What is machine learning?")
        
        # Should return formatted results, not an error
        assert "Mock search result for: What is machine learning?" in result
        assert "[Mock Course - Lesson 1]" in result
        print(f"✓ Basic query success test passed: {result[:100]}...")
    
    def test_execute_with_course_filter(self, mock_success_vector_store):
        """Test query execution with course name filter"""
        tool = CourseSearchTool(mock_success_vector_store)
        result = tool.execute(query="What is ML?", course_name="Machine Learning")
        
        # Should include the search results
        assert "Mock search result for: What is ML?" in result
        print(f"✓ Course filter test passed: {result[:100]}...")
    
    def test_execute_with_lesson_filter(self, mock_success_vector_store):
        """Test query execution with lesson number filter"""
        tool = CourseSearchTool(mock_success_vector_store)
        result = tool.execute(query="What is ML?", lesson_number=1)
        
        assert "Mock search result for: What is ML?" in result
        print(f"✓ Lesson filter test passed: {result[:100]}...")
    
    def test_execute_empty_results(self, mock_empty_vector_store):
        """Test handling of empty search results"""
        tool = CourseSearchTool(mock_empty_vector_store)
        result = tool.execute(query="nonexistent topic")
        
        assert "No relevant content found" in result
        print(f"✓ Empty results test passed: {result}")
    
    def test_execute_search_error(self, mock_error_vector_store):
        """Test handling of search errors"""
        tool = CourseSearchTool(mock_error_vector_store)
        result = tool.execute(query="test query")
        
        assert "Mock search error" in result
        print(f"✓ Search error test passed: {result}")
    
    def test_execute_with_real_vector_store(self, populated_vector_store):
        """Test with real vector store and data"""
        tool = CourseSearchTool(populated_vector_store)
        result = tool.execute(query="machine learning")
        
        # Should find the test content we added
        assert "Introduction to Machine Learning" in result
        print(f"✓ Real vector store test passed: {result[:200]}...")
    
    def test_last_sources_tracking(self, mock_success_vector_store):
        """Test that sources are properly tracked"""
        tool = CourseSearchTool(mock_success_vector_store)
        tool.execute(query="test query")
        
        # Check that sources were tracked
        assert len(tool.last_sources) > 0
        assert tool.last_sources[0]['name'] == "Mock Course - Lesson 1"
        print(f"✓ Sources tracking test passed: {tool.last_sources}")

class TestCourseOutlineTool:
    """Test CourseOutlineTool functionality"""
    
    def test_outline_tool_definition(self, mock_success_vector_store):
        """Test outline tool definition structure"""
        tool = CourseOutlineTool(mock_success_vector_store)
        definition = tool.get_tool_definition()
        
        assert definition["name"] == "get_course_outline"
        assert "course_name" in definition["input_schema"]["properties"]
        print("✓ Outline tool definition test passed")
    
    def test_outline_execute_with_real_data(self, populated_vector_store):
        """Test outline tool with real data"""
        tool = CourseOutlineTool(populated_vector_store)
        result = tool.execute(course_name="Machine Learning")
        
        assert "Introduction to Machine Learning" in result
        assert "Dr. Smith" in result
        print(f"✓ Outline execute test passed: {result[:200]}...")

class TestToolManager:
    """Test ToolManager functionality"""
    
    def test_tool_registration(self, mock_success_vector_store):
        """Test registering tools with ToolManager"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_success_vector_store)
        
        manager.register_tool(search_tool)
        
        # Check that tool is registered
        definitions = manager.get_tool_definitions()
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"
        print("✓ Tool registration test passed")
    
    def test_tool_execution(self, mock_success_vector_store):
        """Test executing tools through ToolManager"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_success_vector_store)
        manager.register_tool(search_tool)
        
        result = manager.execute_tool("search_course_content", query="test query")
        
        assert "Mock search result" in result
        print(f"✓ Tool execution test passed: {result[:100]}...")
    
    def test_last_sources_retrieval(self, mock_success_vector_store):
        """Test retrieving last sources from ToolManager"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_success_vector_store)
        manager.register_tool(search_tool)
        
        # Execute a search
        manager.execute_tool("search_course_content", query="test query")
        
        # Get sources
        sources = manager.get_last_sources()
        assert len(sources) > 0
        print(f"✓ Last sources retrieval test passed: {sources}")

def run_search_tools_tests():
    """Run all search tools tests manually"""
    print("Running CourseSearchTool Tests...")
    print("=" * 50)
    
    # Import fixtures manually for direct testing
    from test_config import MockVectorStore
    
    mock_success = MockVectorStore(should_error=False, return_empty=False)
    mock_error = MockVectorStore(should_error=True, return_empty=False)  
    mock_empty = MockVectorStore(should_error=False, return_empty=True)
    
    # Run individual tests
    test_class = TestCourseSearchTool()
    
    print("\n1. Testing tool definition structure...")
    test_class.test_tool_definition_structure(mock_success)
    
    print("\n2. Testing basic query execution...")
    test_class.test_execute_basic_query_success(mock_success)
    
    print("\n3. Testing course filter...")
    test_class.test_execute_with_course_filter(mock_success)
    
    print("\n4. Testing lesson filter...")
    test_class.test_execute_with_lesson_filter(mock_success)
    
    print("\n5. Testing empty results...")
    test_class.test_execute_empty_results(mock_empty)
    
    print("\n6. Testing search errors...")
    test_class.test_execute_search_error(mock_error)
    
    print("\n7. Testing sources tracking...")
    test_class.test_last_sources_tracking(mock_success)
    
    print("\nTesting ToolManager...")
    manager_test = TestToolManager()
    
    print("\n8. Testing tool registration...")
    manager_test.test_tool_registration(mock_success)
    
    print("\n9. Testing tool execution via manager...")
    manager_test.test_tool_execution(mock_success)
    
    print("\n10. Testing last sources retrieval...")
    manager_test.test_last_sources_retrieval(mock_success)
    
    print("\n" + "=" * 50)
    print("CourseSearchTool tests completed!")

if __name__ == "__main__":
    run_search_tools_tests()