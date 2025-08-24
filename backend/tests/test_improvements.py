"""Test the improvements made to the RAG system"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_system import RAGSystem
from config import config

def test_improved_system():
    """Test the improved RAG system"""
    print("=" * 60)
    print("TESTING IMPROVED RAG SYSTEM")
    print("=" * 60)
    
    rag = RAGSystem(config)
    
    # Test 1: Query that should now use search more often
    print("\n1. Testing machine learning query (should now use search)...")
    response, sources = rag.query("What is machine learning?")
    print(f"   Response length: {len(response)}")
    print(f"   Sources found: {len(sources)}")
    if sources:
        print("   ✓ Sources returned - improvement working!")
        for source in sources:
            print(f"     - {source.get('name', str(source))}")
    else:
        print("   ⚠️  Still using general knowledge")
    
    # Test 2: Course outline query (should now provide sources)
    print("\n2. Testing course outline query (should now provide sources)...")
    response, sources = rag.query("Tell me about the MCP course structure")
    print(f"   Response length: {len(response)}")
    print(f"   Sources found: {len(sources)}")
    if sources:
        print("   ✓ Outline sources now working!")
        for source in sources:
            print(f"     - {source.get('name', str(source))}")
    else:
        print("   ⚠️  Outline sources still not working")
    
    # Test 3: Error handling
    print("\n3. Testing error handling with malformed query...")
    try:
        response, sources = rag.query("")  # Empty query
        print(f"   Response: {response[:100]}...")
        print("   ✓ Error handled gracefully")
    except Exception as e:
        print(f"   ❌ Error not handled: {e}")

if __name__ == "__main__":
    test_improved_system()