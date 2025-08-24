"""Debug sources tracking issue"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import config
from rag_system import RAGSystem


def debug_sources_flow():
    """Debug exactly where sources are being lost"""
    print("=" * 60)
    print("DEBUGGING: Sources tracking flow")
    print("=" * 60)

    rag = RAGSystem(config)

    print("1. Initial state - checking for existing sources...")
    initial_sources = rag.tool_manager.get_last_sources()
    print(f"   Initial sources: {initial_sources}")

    print("\n2. Direct tool execution...")
    direct_result = rag.tool_manager.execute_tool(
        "search_course_content", query="prompt compression"
    )
    print(f"   Direct result: {direct_result[:100]}...")

    print("\n3. Sources after direct execution...")
    after_direct = rag.tool_manager.get_last_sources()
    print(f"   Sources count: {len(after_direct)}")
    for i, source in enumerate(after_direct):
        print(f"   Source {i}: {source}")

    print("\n4. Now testing full query flow...")
    print("   Calling rag.query() with same query...")

    # Reset sources first to simulate clean state
    rag.tool_manager.reset_sources()
    print("   Sources reset - count now:", len(rag.tool_manager.get_last_sources()))

    # Call the full query
    response, query_sources = rag.query("Tell me about prompt compression techniques")

    print("\n5. After full query...")
    print(f"   Response length: {len(response)}")
    print(f"   Query sources returned: {len(query_sources)}")
    print(f"   Tool manager sources: {len(rag.tool_manager.get_last_sources())}")

    if query_sources:
        print("   Query sources:")
        for i, source in enumerate(query_sources):
            print(f"     {i}: {source}")
    else:
        print("   No query sources returned!")

    final_tm_sources = rag.tool_manager.get_last_sources()
    if final_tm_sources:
        print("   Tool manager sources:")
        for i, source in enumerate(final_tm_sources):
            print(f"     {i}: {source}")
    else:
        print("   No tool manager sources!")


def debug_ai_generator_tool_calls():
    """Debug if AI generator is actually calling tools"""
    print("\n" + "=" * 60)
    print("DEBUGGING: AI Generator tool calls")
    print("=" * 60)

    rag = RAGSystem(config)

    # Monkey patch the tool manager to log calls
    original_execute = rag.tool_manager.execute_tool

    def logged_execute_tool(tool_name, **kwargs):
        print(f"   🔧 TOOL CALLED: {tool_name} with {kwargs}")
        result = original_execute(tool_name, **kwargs)
        print(f"   🔧 TOOL RESULT: {result[:100]}...")
        sources_after = rag.tool_manager.get_last_sources()
        print(f"   🔧 SOURCES AFTER TOOL: {len(sources_after)} items")
        return result

    rag.tool_manager.execute_tool = logged_execute_tool

    test_queries = [
        "What is prompt compression?",  # Should use search
        "Tell me about MCP course structure",  # Should use outline
        "Explain machine learning concepts in the courses",  # Should use search
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: '{query}' ---")
        rag.tool_manager.reset_sources()

        print("   Calling rag.query()...")
        response, sources = rag.query(query)

        print(f"   Response: {response[:100]}...")
        print(f"   Sources returned: {len(sources)}")

        if not sources:
            print("   ❌ No sources returned - checking why...")
            tm_sources = rag.tool_manager.get_last_sources()
            print(f"   Tool manager has {len(tm_sources)} sources")


def debug_system_prompt():
    """Check if system prompt is causing AI to avoid using tools"""
    print("\n" + "=" * 60)
    print("DEBUGGING: System prompt and tool usage")
    print("=" * 60)

    from ai_generator import AIGenerator

    print("Current system prompt:")
    print("-" * 40)
    print(AIGenerator.SYSTEM_PROMPT)
    print("-" * 40)

    print("\nAnalyzing prompt for tool usage guidance:")

    if "use tools" in AIGenerator.SYSTEM_PROMPT.lower():
        print("✓ Prompt mentions using tools")
    else:
        print("❌ Prompt doesn't clearly mention using tools")

    if "search" in AIGenerator.SYSTEM_PROMPT.lower():
        print("✓ Prompt mentions search")
    else:
        print("❌ Prompt doesn't mention search")

    if "general knowledge" in AIGenerator.SYSTEM_PROMPT.lower():
        print("✓ Prompt mentions general knowledge")
        print("  ⚠️  This might be causing AI to prefer general knowledge over search")
    else:
        print("✓ Prompt doesn't mention general knowledge preference")


if __name__ == "__main__":
    debug_sources_flow()
    debug_ai_generator_tool_calls()
    debug_system_prompt()

    print("\n" + "=" * 60)
    print("SOURCES DEBUG COMPLETE")
    print("=" * 60)
