"""Main test runner to diagnose RAG system issues"""

import os
import sys
import traceback

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def run_all_tests():
    """Run all test suites and capture results"""
    print("=" * 80)
    print("RAG SYSTEM DIAGNOSTIC TEST SUITE")
    print("=" * 80)
    print("This will help identify where 'query failed' issues are occurring.\n")

    test_results = {
        "search_tools": {"passed": 0, "failed": 0, "errors": []},
        "ai_generator": {"passed": 0, "failed": 0, "errors": []},
        "integration": {"passed": 0, "failed": 0, "errors": []},
        "real_system": {"passed": 0, "failed": 0, "errors": []},
    }

    # Test 1: CourseSearchTool Tests
    print("🔍 TESTING COURSE SEARCH TOOL")
    print("-" * 40)
    try:
        from test_search_tools import run_search_tools_tests

        run_search_tools_tests()
        test_results["search_tools"]["passed"] += 1
        print("✅ CourseSearchTool tests completed successfully")
    except Exception as e:
        test_results["search_tools"]["failed"] += 1
        test_results["search_tools"]["errors"].append(str(e))
        print(f"❌ CourseSearchTool tests failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")

    print("\n")

    # Test 2: AIGenerator Tests
    print("🤖 TESTING AI GENERATOR")
    print("-" * 40)
    try:
        from test_ai_generator import run_ai_generator_tests

        run_ai_generator_tests()
        test_results["ai_generator"]["passed"] += 1
        print("✅ AIGenerator tests completed successfully")
    except Exception as e:
        test_results["ai_generator"]["failed"] += 1
        test_results["ai_generator"]["errors"].append(str(e))
        print(f"❌ AIGenerator tests failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")

    print("\n")

    # Test 3: Integration Tests
    print("🔗 TESTING INTEGRATION")
    print("-" * 40)
    try:
        from test_integration import run_integration_tests

        run_integration_tests()
        test_results["integration"]["passed"] += 1
        print("✅ Integration tests completed successfully")
    except Exception as e:
        test_results["integration"]["failed"] += 1
        test_results["integration"]["errors"].append(str(e))
        print(f"❌ Integration tests failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")

    print("\n")

    # Test 4: Real System Tests
    print("🏗️ TESTING REAL SYSTEM")
    print("-" * 40)
    try:
        test_real_system()
        test_results["real_system"]["passed"] += 1
        print("✅ Real system tests completed successfully")
    except Exception as e:
        test_results["real_system"]["failed"] += 1
        test_results["real_system"]["errors"].append(str(e))
        print(f"❌ Real system tests failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")

    # Summary
    print("\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)

    total_passed = sum(r["passed"] for r in test_results.values())
    total_failed = sum(r["failed"] for r in test_results.values())

    for test_name, results in test_results.items():
        status = "✅ PASSED" if results["failed"] == 0 else "❌ FAILED"
        print(f"{test_name.upper():<20}: {status}")
        if results["errors"]:
            for error in results["errors"]:
                print(f"  - {error}")

    print(f"\nOVERALL: {total_passed} passed, {total_failed} failed")

    if total_failed > 0:
        print("\n🔧 DIAGNOSIS & RECOMMENDATIONS:")
        diagnose_issues(test_results)

    return test_results


def test_real_system():
    """Test the actual deployed system"""
    print("Testing real system components...")

    # Test 1: Can we import all modules?
    print("\n1. Testing module imports...")
    try:
        from config import config
        from rag_system import RAGSystem

        print("✓ Successfully imported RAG system and config")
    except Exception as e:
        print(f"✗ Failed to import modules: {e}")
        raise

    # Test 2: Can we initialize the system?
    print("\n2. Testing system initialization...")
    try:
        rag = RAGSystem(config)
        print("✓ Successfully initialized RAG system")
    except Exception as e:
        print(f"✗ Failed to initialize RAG system: {e}")
        raise

    # Test 3: Are tools registered?
    print("\n3. Testing tool registration...")
    try:
        tools = rag.tool_manager.get_tool_definitions()
        tool_names = [t["name"] for t in tools]
        print(f"✓ Tools registered: {tool_names}")

        if "search_course_content" not in tool_names:
            raise Exception("search_course_content tool not registered")
        if "get_course_outline" not in tool_names:
            raise Exception("get_course_outline tool not registered")

    except Exception as e:
        print(f"✗ Tool registration failed: {e}")
        raise

    # Test 4: Can we execute search tool directly?
    print("\n4. Testing direct search tool execution...")
    try:
        result = rag.tool_manager.execute_tool("search_course_content", query="test")
        print(f"✓ Direct search tool execution result: {result[:100]}...")

        if "error" in result.lower() or "failed" in result.lower():
            print(f"⚠️  Search tool returned error-like message: {result}")
    except Exception as e:
        print(f"✗ Direct search tool execution failed: {e}")
        raise

    # Test 5: Check vector store data
    print("\n5. Testing vector store data...")
    try:
        course_count = rag.vector_store.get_course_count()
        course_titles = rag.vector_store.get_existing_course_titles()
        print(f"✓ Vector store has {course_count} courses: {course_titles}")

        if course_count == 0:
            print("⚠️  No courses found in vector store - this could be the issue!")

    except Exception as e:
        print(f"✗ Vector store data check failed: {e}")
        raise

    # Test 6: Test search directly on vector store
    print("\n6. Testing vector store search directly...")
    try:
        search_result = rag.vector_store.search("machine learning")
        print(f"✓ Vector store search returned {len(search_result.documents)} results")

        if search_result.error:
            print(f"⚠️  Vector store search returned error: {search_result.error}")
        if search_result.is_empty():
            print("⚠️  Vector store search returned empty results")
        else:
            print(f"   Sample result: {search_result.documents[0][:100]}...")

    except Exception as e:
        print(f"✗ Vector store search failed: {e}")
        raise

    # Test 7: Check if API key is set
    print("\n7. Testing API configuration...")
    try:
        api_key = config.ANTHROPIC_API_KEY
        if not api_key or api_key == "test-key":
            print("⚠️  ANTHROPIC_API_KEY not properly set")
        else:
            print(f"✓ API key is set (length: {len(api_key)})")
    except Exception as e:
        print(f"✗ API configuration check failed: {e}")
        raise


def diagnose_issues(test_results):
    """Provide diagnosis and recommendations based on test results"""

    print("\nBased on the test results, here are potential issues and fixes:")
    print("-" * 60)

    if test_results["search_tools"]["failed"] > 0:
        print("🔧 SEARCH TOOLS ISSUES:")
        print("- CourseSearchTool.execute() method may be failing")
        print("- Vector store search functionality may be broken")
        print("- Tool definition or parameter handling issues")
        print("- Recommendation: Check vector_store.search() implementation")
        print()

    if test_results["ai_generator"]["failed"] > 0:
        print("🔧 AI GENERATOR ISSUES:")
        print("- Tool calling mechanism may be broken")
        print("- Anthropic API integration issues")
        print("- Tool response handling problems")
        print("- Recommendation: Check _handle_tool_execution() method")
        print()

    if test_results["integration"]["failed"] > 0:
        print("🔧 INTEGRATION ISSUES:")
        print("- End-to-end flow is broken")
        print("- Component communication problems")
        print("- Session management issues")
        print("- Recommendation: Check RAGSystem.query() method")
        print()

    if test_results["real_system"]["failed"] > 0:
        print("🔧 REAL SYSTEM ISSUES:")
        print("- Configuration problems")
        print("- Missing data in vector store")
        print("- Environment setup issues")
        print("- API key or model configuration problems")
        print("- Recommendation: Check system initialization and data loading")
        print()

    print("🎯 COMMON FIXES TO TRY:")
    print("1. Verify ANTHROPIC_API_KEY is set correctly")
    print("2. Check that course documents are loaded into vector store")
    print("3. Verify ChromaDB is working correctly")
    print("4. Check that tools are properly registered with ToolManager")
    print("5. Verify AI generator is calling tools correctly")
    print("6. Check error handling in search tool execution")


if __name__ == "__main__":
    run_all_tests()
