"""Simple diagnostic test to identify the actual query failure issue"""
import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_system import RAGSystem
from config import config

def test_actual_query_flow():
    """Test the actual query flow that users experience"""
    print("=" * 60)
    print("DIAGNOSTIC: Testing actual query flow")
    print("=" * 60)
    
    # Initialize the system
    print("1. Initializing RAG system...")
    rag = RAGSystem(config)
    
    # Test different types of queries
    test_queries = [
        "What is machine learning?",  # Should trigger search
        "Tell me about MCP course structure",  # Should trigger outline or search
        "Explain prompt compression",  # Should trigger search
        "What is the capital of France?",  # Should use general knowledge
        "List all courses available",  # Should trigger search or general response
    ]
    
    print(f"\n2. Testing {len(test_queries)} different query types...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test Query {i}: '{query}' ---")
        
        try:
            # This is the exact same call the API makes
            response, sources = rag.query(query, session_id="test-session")
            
            print(f"✓ Response received (length: {len(response)})")
            print(f"✓ Sources: {len(sources)} items")
            
            # Check for failure indicators
            if "error" in response.lower():
                print(f"⚠️  Response contains 'error': {response[:200]}...")
            elif "failed" in response.lower():
                print(f"⚠️  Response contains 'failed': {response[:200]}...")
            elif len(response.strip()) == 0:
                print("⚠️  Empty response received")
            else:
                print(f"✓ Response looks normal: {response[:150]}...")
            
            if sources:
                print(f"✓ Sources found: {[s.get('name', str(s)) for s in sources]}")
            
        except Exception as e:
            print(f"❌ Query failed with exception: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")

def test_tool_manager_directly():
    """Test the tool manager directly with sample queries"""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC: Testing tool manager directly")
    print("=" * 60)
    
    rag = RAGSystem(config)
    
    # Test direct tool execution
    print("1. Testing direct search tool execution...")
    try:
        result = rag.tool_manager.execute_tool(
            "search_course_content", 
            query="machine learning"
        )
        print(f"✓ Direct search result: {result[:200]}...")
        
        if "error" in result.lower() or "failed" in result.lower():
            print(f"⚠️  Direct search contains error terms")
        
    except Exception as e:
        print(f"❌ Direct search failed: {e}")
    
    print("\n2. Testing outline tool execution...")
    try:
        result = rag.tool_manager.execute_tool(
            "get_course_outline",
            course_name="MCP"
        )
        print(f"✓ Direct outline result: {result[:200]}...")
        
    except Exception as e:
        print(f"❌ Direct outline failed: {e}")

def test_ai_generator_with_tools():
    """Test AI generator with and without tools"""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC: Testing AI generator with tools")
    print("=" * 60)
    
    rag = RAGSystem(config)
    
    # Test 1: General knowledge (shouldn't use tools)
    print("1. Testing general knowledge query...")
    try:
        response = rag.ai_generator.generate_response(
            query="What is the capital of France?",
            tools=None,  # No tools
            tool_manager=None
        )
        print(f"✓ General knowledge response: {response[:150]}...")
        
    except Exception as e:
        print(f"❌ General knowledge query failed: {e}")
    
    # Test 2: Course content (should use tools)
    print("\n2. Testing course content query with tools...")
    try:
        response = rag.ai_generator.generate_response(
            query="What is machine learning in the course content?",
            tools=rag.tool_manager.get_tool_definitions(),
            tool_manager=rag.tool_manager
        )
        print(f"✓ Course content response: {response[:150]}...")
        
        if "query failed" in response.lower():
            print("⚠️  Found 'query failed' in AI response!")
        
    except Exception as e:
        print(f"❌ Course content query failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

def test_sources_tracking():
    """Test if sources are being tracked correctly"""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC: Testing sources tracking")
    print("=" * 60)
    
    rag = RAGSystem(config)
    
    # Execute a search that should produce sources
    print("1. Executing search tool directly...")
    result = rag.tool_manager.execute_tool(
        "search_course_content",
        query="machine learning"
    )
    
    # Check if sources were tracked
    sources = rag.tool_manager.get_last_sources()
    print(f"✓ Sources after direct tool execution: {sources}")
    
    # Reset and try through query
    rag.tool_manager.reset_sources()
    
    print("\n2. Executing through full query...")
    response, query_sources = rag.query("Tell me about machine learning")
    
    print(f"✓ Sources from query: {query_sources}")
    print(f"✓ Tool manager sources: {rag.tool_manager.get_last_sources()}")

if __name__ == "__main__":
    print("Starting simple diagnostic tests...")
    
    test_actual_query_flow()
    test_tool_manager_directly()
    test_ai_generator_with_tools()
    test_sources_tracking()
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)