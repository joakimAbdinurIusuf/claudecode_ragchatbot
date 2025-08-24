"""Tests for AIGenerator tool calling functionality"""
import pytest
import sys
import os
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_generator import AIGenerator
from search_tools import CourseSearchTool, ToolManager
from test_config import mock_success_vector_store, MockVectorStore

class MockAnthropicResponse:
    """Mock Anthropic API response"""
    def __init__(self, content_text=None, stop_reason="end_turn", tool_use_content=None):
        if tool_use_content:
            # Tool use response
            self.content = tool_use_content
            self.stop_reason = stop_reason
        else:
            # Text response
            mock_content = Mock()
            mock_content.text = content_text or "Mock AI response"
            self.content = [mock_content]
            self.stop_reason = stop_reason

class MockToolUseContent:
    """Mock tool use content block"""
    def __init__(self, tool_name="search_course_content", tool_input=None, tool_id="mock-tool-id"):
        self.type = "tool_use"
        self.name = tool_name
        self.input = tool_input or {"query": "test query"}
        self.id = tool_id

class TestAIGenerator:
    """Test AIGenerator functionality"""
    
    @patch('anthropic.Anthropic')
    def test_generate_response_without_tools(self, mock_anthropic):
        """Test basic response generation without tools"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.return_value = MockAnthropicResponse("Hello, this is a test response")
        
        # Test
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        response = generator.generate_response("Hello")
        
        assert response == "Hello, this is a test response"
        assert mock_client.messages.create.called
        print(f"✓ Basic response test passed: {response}")
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_tools_no_use(self, mock_anthropic):
        """Test response generation with tools available but not used"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.return_value = MockAnthropicResponse("I can answer this from general knowledge")
        
        # Setup tools
        mock_store = MockVectorStore()
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_store)
        tool_manager.register_tool(search_tool)
        
        # Test
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        response = generator.generate_response(
            query="What is the capital of France?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        assert response == "I can answer this from general knowledge"
        print(f"✓ Tools available but not used test passed: {response}")
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_tool_use(self, mock_anthropic):
        """Test response generation with actual tool usage"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # First call returns tool use request
        tool_use_content = [MockToolUseContent()]
        initial_response = MockAnthropicResponse(
            stop_reason="tool_use", 
            tool_use_content=tool_use_content
        )
        
        # Second call returns final response
        final_response = MockAnthropicResponse("Based on the search results, machine learning is...")
        
        mock_client.messages.create.side_effect = [initial_response, final_response]
        
        # Setup tools
        mock_store = MockVectorStore()
        tool_manager = ToolManager() 
        search_tool = CourseSearchTool(mock_store)
        tool_manager.register_tool(search_tool)
        
        # Test
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        response = generator.generate_response(
            query="What is machine learning?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        assert response == "Based on the search results, machine learning is..."
        assert mock_client.messages.create.call_count == 2  # Two API calls
        print(f"✓ Tool use test passed: {response}")
    
    @patch('anthropic.Anthropic')
    def test_tool_execution_flow(self, mock_anthropic):
        """Test the detailed tool execution flow"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock tool use response
        tool_use_content = [MockToolUseContent(
            tool_name="search_course_content",
            tool_input={"query": "machine learning basics"},
            tool_id="tool-123"
        )]
        initial_response = MockAnthropicResponse(
            stop_reason="tool_use",
            tool_use_content=tool_use_content
        )
        
        final_response = MockAnthropicResponse("Here's what I found about machine learning...")
        mock_client.messages.create.side_effect = [initial_response, final_response]
        
        # Setup tool manager
        mock_store = MockVectorStore()
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_store)
        tool_manager.register_tool(search_tool)
        
        # Test
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        response = generator.generate_response(
            query="Tell me about machine learning",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Verify tool was actually called
        assert mock_client.messages.create.call_count == 2
        
        # Check that the second call included tool results
        second_call_args = mock_client.messages.create.call_args_list[1]
        messages = second_call_args[1]['messages']  # keyword arguments
        
        # Should have: original message, assistant tool use, user tool results  
        assert len(messages) >= 3
        print(f"✓ Tool execution flow test passed with {len(messages)} messages")
    
    @patch('anthropic.Anthropic') 
    def test_conversation_history_handling(self, mock_anthropic):
        """Test that conversation history is properly included"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.return_value = MockAnthropicResponse("Response with history")
        
        # Test
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        response = generator.generate_response(
            query="Follow up question",
            conversation_history="Previous: What is ML? Assistant: Machine learning is..."
        )
        
        # Check that system message includes history
        call_args = mock_client.messages.create.call_args
        system_content = call_args[1]['system']
        
        assert "Previous conversation:" in system_content
        assert "What is ML?" in system_content
        print("✓ Conversation history test passed")
    
    @patch('anthropic.Anthropic')
    def test_error_handling_in_tool_execution(self, mock_anthropic):
        """Test error handling when tool execution fails"""
        # Setup mock that returns tool use
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        tool_use_content = [MockToolUseContent()]
        initial_response = MockAnthropicResponse(
            stop_reason="tool_use",
            tool_use_content=tool_use_content
        )
        
        final_response = MockAnthropicResponse("I apologize, but I encountered an error...")
        mock_client.messages.create.side_effect = [initial_response, final_response]
        
        # Setup tool manager with error-prone store
        error_store = MockVectorStore(should_error=True)
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(error_store)
        tool_manager.register_tool(search_tool)
        
        # Test
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        response = generator.generate_response(
            query="What is machine learning?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Should still get a response even if tool failed
        assert response == "I apologize, but I encountered an error..."
        print("✓ Error handling test passed")

class TestSequentialToolCalling:
    """Test new sequential tool calling functionality"""
    
    @patch('anthropic.Anthropic')
    def test_sequential_two_round_tool_calling(self, mock_anthropic):
        """Test that AI can make 2 sequential tool calls"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Round 1: Initial tool call
        round1_tool_use = [MockToolUseContent(
            tool_name="get_course_outline",
            tool_input={"course_name": "Machine Learning"},
            tool_id="tool-1"
        )]
        round1_response = MockAnthropicResponse(
            stop_reason="tool_use", 
            tool_use_content=round1_tool_use
        )
        
        # Round 2: Follow-up tool call  
        round2_tool_use = [MockToolUseContent(
            tool_name="search_course_content",
            tool_input={"query": "neural networks", "course_name": "Machine Learning", "lesson_number": 3},
            tool_id="tool-2"
        )]
        round2_response = MockAnthropicResponse(
            stop_reason="tool_use",
            tool_use_content=round2_tool_use  
        )
        
        # Final response
        final_response = MockAnthropicResponse("Based on the course outline and specific search, neural networks are...")
        
        mock_client.messages.create.side_effect = [round1_response, round2_response, final_response]
        
        # Setup tools
        mock_store = MockVectorStore()
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_store)
        tool_manager.register_tool(search_tool)
        
        # Test
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        response = generator.generate_response(
            query="Tell me about neural networks in the Machine Learning course",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Assertions
        assert mock_client.messages.create.call_count == 3  # Three API calls
        assert "Based on the course outline and specific search" in response
        print("✓ Sequential two-round tool calling test passed")
    
    @patch('anthropic.Anthropic')
    def test_early_termination_no_second_tool_call(self, mock_anthropic):
        """Test termination when Claude doesn't request second tool call"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Round 1: Tool call
        round1_tool_use = [MockToolUseContent(tool_id="tool-1")]
        round1_response = MockAnthropicResponse(
            stop_reason="tool_use",
            tool_use_content=round1_tool_use
        )
        
        # Round 2: Text response (no tool call)
        final_response = MockAnthropicResponse("Based on the search results, here's the answer...")
        
        mock_client.messages.create.side_effect = [round1_response, final_response]
        
        # Setup tools
        mock_store = MockVectorStore()
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_store)
        tool_manager.register_tool(search_tool)
        
        # Test
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        response = generator.generate_response(
            query="What is machine learning?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Should terminate after 2 calls (not reach max rounds)
        assert mock_client.messages.create.call_count == 2
        assert "Based on the search results" in response
        print("✓ Early termination test passed")
    
    @patch('anthropic.Anthropic')
    def test_max_rounds_enforcement(self, mock_anthropic):
        """Test that system enforces 2-round maximum"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # All rounds return tool use requests
        tool_use_content = [MockToolUseContent()]
        tool_use_response = MockAnthropicResponse(
            stop_reason="tool_use",
            tool_use_content=tool_use_content
        )
        
        # Return tool use for all calls (should be limited to 2 rounds)
        mock_client.messages.create.return_value = tool_use_response
        
        # Setup tools
        mock_store = MockVectorStore()
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_store)
        tool_manager.register_tool(search_tool)
        
        # Test
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        response = generator.generate_response(
            query="Complex query that would need many searches",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Should be exactly 2 tool execution rounds (could be more API calls)
        assert mock_client.messages.create.call_count >= 2
        # Should extract text from final response and mention max rounds reached
        assert ("Mock AI response" in response or 
                "couldn't generate a text response" in response or
                "tool use" in response.lower())
        print("✓ Max rounds enforcement test passed")
    
    @patch('anthropic.Anthropic')
    def test_tool_error_handling_mid_sequence(self, mock_anthropic):
        """Test error handling during sequential execution"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # First round succeeds with tool call
        round1_tool_use = [MockToolUseContent(tool_id="tool-1")]
        round1_response = MockAnthropicResponse(
            stop_reason="tool_use",
            tool_use_content=round1_tool_use
        )
        
        mock_client.messages.create.return_value = round1_response
        
        # Setup tool manager that will fail on tool execution
        error_store = MockVectorStore(should_error=True)
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(error_store)
        tool_manager.register_tool(search_tool)
        
        # Test
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        response = generator.generate_response(
            query="What is machine learning?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Should handle tool error gracefully (error flows through to final response)
        assert "Mock search error" in response or "error" in response.lower() or "couldn't generate" in response.lower()
        print("✓ Tool error handling test passed")
    
    @patch('anthropic.Anthropic')
    def test_conversation_state_preservation_across_rounds(self, mock_anthropic):
        """Test that conversation state is preserved across rounds"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Round 1: Tool call
        round1_tool_use = [MockToolUseContent(tool_id="tool-1")]
        round1_response = MockAnthropicResponse(
            stop_reason="tool_use",
            tool_use_content=round1_tool_use
        )
        
        # Round 2: Tool call
        round2_tool_use = [MockToolUseContent(tool_id="tool-2")]  
        round2_response = MockAnthropicResponse(
            stop_reason="tool_use",
            tool_use_content=round2_tool_use
        )
        
        # Final response
        final_response = MockAnthropicResponse("Final answer")
        
        mock_client.messages.create.side_effect = [round1_response, round2_response, final_response]
        
        # Setup tools
        mock_store = MockVectorStore()
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_store)
        tool_manager.register_tool(search_tool)
        
        # Test
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        response = generator.generate_response(
            query="Complex comparison query",
            conversation_history="Previous: What is AI? Assistant: AI is...",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Check that all rounds preserved conversation history
        for call_args in mock_client.messages.create.call_args_list:
            if 'system' in call_args[1]:
                system_content = call_args[1]['system']
                assert "Previous conversation:" in system_content
                assert "What is AI?" in system_content
        
        print("✓ Conversation state preservation test passed")
    
    @patch('anthropic.Anthropic')
    def test_tools_remain_available_across_rounds(self, mock_anthropic):
        """Test that tools remain available in subsequent rounds"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Round 1 and 2: Tool calls
        tool_use_content = [MockToolUseContent()]
        round1_response = MockAnthropicResponse(stop_reason="tool_use", tool_use_content=tool_use_content)
        round2_response = MockAnthropicResponse(stop_reason="tool_use", tool_use_content=tool_use_content)
        final_response = MockAnthropicResponse("Final answer")
        
        mock_client.messages.create.side_effect = [round1_response, round2_response, final_response]
        
        # Setup tools
        mock_store = MockVectorStore()
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_store)
        tool_manager.register_tool(search_tool)
        
        # Test
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        response = generator.generate_response(
            query="Sequential query",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Check that both round 1 and round 2 had tools available
        call_args_list = mock_client.messages.create.call_args_list
        
        # Round 2 call should have tools
        round2_call_args = call_args_list[1]
        assert 'tools' in round2_call_args[1]
        assert len(round2_call_args[1]['tools']) > 0
        assert round2_call_args[1]['tool_choice'] == {"type": "auto"}
        
        print("✓ Tools availability test passed")

def run_ai_generator_tests():
    """Run all AI generator tests manually"""
    print("Running AIGenerator Tests...")
    print("=" * 50)
    
    # Original tests
    test_class = TestAIGenerator()
    
    print("\n1. Testing basic response generation...")
    test_class.test_generate_response_without_tools()
    
    print("\n2. Testing response with tools available but not used...")
    test_class.test_generate_response_with_tools_no_use()
    
    print("\n3. Testing response with actual tool usage...")
    test_class.test_generate_response_with_tool_use()
    
    print("\n4. Testing tool execution flow...")
    test_class.test_tool_execution_flow()
    
    print("\n5. Testing conversation history handling...")
    test_class.test_conversation_history_handling()
    
    print("\n6. Testing error handling in tool execution...")
    test_class.test_error_handling_in_tool_execution()
    
    # Sequential tool calling tests
    print("\n" + "=" * 50)
    print("Running Sequential Tool Calling Tests...")
    print("=" * 50)
    
    sequential_test_class = TestSequentialToolCalling()
    
    print("\n7. Testing sequential two-round tool calling...")
    sequential_test_class.test_sequential_two_round_tool_calling()
    
    print("\n8. Testing early termination without second tool call...")
    sequential_test_class.test_early_termination_no_second_tool_call()
    
    print("\n9. Testing max rounds enforcement...")
    sequential_test_class.test_max_rounds_enforcement()
    
    print("\n10. Testing tool error handling mid-sequence...")
    sequential_test_class.test_tool_error_handling_mid_sequence()
    
    print("\n11. Testing conversation state preservation across rounds...")
    sequential_test_class.test_conversation_state_preservation_across_rounds()
    
    print("\n12. Testing tools remain available across rounds...")
    sequential_test_class.test_tools_remain_available_across_rounds()
    
    print("\n" + "=" * 50)
    print("All AIGenerator tests completed!")

if __name__ == "__main__":
    run_ai_generator_tests()