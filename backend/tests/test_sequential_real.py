"""End-to-end test of sequential tool calling with real system"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import config
from rag_system import RAGSystem


def test_sequential_in_real_system():
    """Test sequential tool calling with the real RAG system"""
    print("=" * 60)
    print("TESTING SEQUENTIAL TOOL CALLING IN REAL SYSTEM")
    print("=" * 60)

    rag = RAGSystem(config)

    # Test 1: Query that should trigger sequential searches
    print("\n1. Testing comparison query (should use sequential searches)...")
    query = "Compare the topics covered in lesson 1 vs lesson 2 of the MCP course"
    print(f"Query: {query}")

    response, sources = rag.query(query)
    print(f"\nResponse length: {len(response)}")
    print(f"Sources found: {len(sources)}")
    print(f"Response preview: {response[:200]}...")

    if len(sources) > 0:
        print("\nSources:")
        for i, source in enumerate(sources):
            print(f"  {i+1}. {source.get('name', str(source))}")

    # Test 2: Query that should use outline then search
    print("\n" + "-" * 40)
    print("2. Testing outline-then-search query...")
    query2 = "What is the course structure of the MCP course and what are the key topics in lesson 3?"
    print(f"Query: {query2}")

    response2, sources2 = rag.query(query2)
    print(f"\nResponse length: {len(response2)}")
    print(f"Sources found: {len(sources2)}")
    print(f"Response preview: {response2[:200]}...")

    if len(sources2) > 0:
        print("\nSources:")
        for i, source in enumerate(sources2):
            print(f"  {i+1}. {source.get('name', str(source))}")

    # Test 3: Complex query that should benefit from multiple searches
    print("\n" + "-" * 40)
    print("3. Testing complex multi-search query...")
    query3 = "Find courses that discuss prompt compression and tell me about the specific techniques mentioned"
    print(f"Query: {query3}")

    response3, sources3 = rag.query(query3)
    print(f"\nResponse length: {len(response3)}")
    print(f"Sources found: {len(sources3)}")
    print(f"Response preview: {response3[:200]}...")

    if len(sources3) > 0:
        print("\nSources:")
        for i, source in enumerate(sources3):
            print(f"  {i+1}. {source.get('name', str(source))}")

    print("\n" + "=" * 60)
    print("SEQUENTIAL TOOL CALLING TEST COMPLETE")
    print("=" * 60)

    # Summary
    total_sources = len(sources) + len(sources2) + len(sources3)
    print("\nSummary:")
    print(f"- Test 1 sources: {len(sources)}")
    print(f"- Test 2 sources: {len(sources2)}")
    print(f"- Test 3 sources: {len(sources3)}")
    print(f"- Total sources across all tests: {total_sources}")

    if total_sources > 6:  # Expect more sources with sequential calling
        print("✅ Sequential tool calling appears to be working - more sources found!")
    else:
        print("⚠️  May not be using sequential calls effectively")


if __name__ == "__main__":
    test_sequential_in_real_system()
