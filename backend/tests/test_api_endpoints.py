"""API endpoint tests for the RAG system"""
import pytest
from fastapi.testclient import TestClient
import json


@pytest.mark.api
class TestAPIEndpoints:
    """Test FastAPI endpoints for proper request/response handling"""
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint returns correct response"""
        response = test_client.get("/")
        
        assert response.status_code == 200
        assert response.json() == {"message": "RAG System API"}
    
    def test_query_endpoint_success(self, test_client):
        """Test successful query endpoint request"""
        query_data = {
            "query": "What is machine learning?",
            "session_id": None
        }
        
        response = test_client.post("/api/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "answer" in data
        assert "sources" in data  
        assert "session_id" in data
        
        # Check types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        
        # Check answer is not empty
        assert len(data["answer"]) > 0
        
        print(f"✓ Query endpoint test passed: {data['answer'][:50]}...")
    
    def test_query_endpoint_with_session(self, test_client):
        """Test query endpoint with existing session ID"""
        query_data = {
            "query": "Tell me about AI",
            "session_id": "test-session-123"
        }
        
        response = test_client.post("/api/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return the same session ID
        assert data["session_id"] == "test-session-123"
        assert "answer" in data
        assert len(data["answer"]) > 0
    
    def test_query_endpoint_invalid_request(self, test_client):
        """Test query endpoint with invalid request data"""
        # Missing required query field
        invalid_data = {"session_id": "test"}
        
        response = test_client.post("/api/query", json=invalid_data)
        
        assert response.status_code == 422  # Unprocessable Entity
        
        # Check error response structure
        error_data = response.json()
        assert "detail" in error_data
        assert isinstance(error_data["detail"], list)
    
    def test_query_endpoint_empty_query(self, test_client):
        """Test query endpoint with empty query string"""
        query_data = {
            "query": "",
            "session_id": None
        }
        
        response = test_client.post("/api/query", json=query_data)
        
        # Should still process the request, even if query is empty
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "session_id" in data
    
    def test_courses_endpoint_empty(self, test_client):
        """Test courses endpoint when no courses are loaded"""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "total_courses" in data
        assert "course_titles" in data
        
        # Should be empty initially
        assert data["total_courses"] == 0
        assert data["course_titles"] == []
        assert isinstance(data["course_titles"], list)
    
    def test_clear_session_endpoint(self, test_client):
        """Test session clearing endpoint"""
        session_id = "test-session-to-clear"
        
        response = test_client.delete(f"/api/sessions/{session_id}/clear")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "success" in data
        assert "message" in data
        
        # Check values
        assert data["success"] is True
        assert session_id in data["message"]
        assert isinstance(data["message"], str)
    
    def test_clear_session_special_characters(self, test_client):
        """Test session clearing with special characters in session ID"""
        session_id = "test-session-with-special-chars-123!@#"
        
        response = test_client.delete(f"/api/sessions/{session_id}/clear")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    def test_nonexistent_endpoint(self, test_client):
        """Test accessing non-existent endpoint returns 404"""
        response = test_client.get("/api/nonexistent")
        
        assert response.status_code == 404


@pytest.mark.api 
class TestAPIRequestResponseFormats:
    """Test API request/response format validation"""
    
    def test_query_request_validation(self, test_client):
        """Test query request validation with various invalid inputs"""
        
        # Test with non-string query
        invalid_requests = [
            {"query": 123, "session_id": None},  # Non-string query
            {"query": None, "session_id": None},  # None query
            {"query": [], "session_id": None},    # List query
        ]
        
        for invalid_request in invalid_requests:
            response = test_client.post("/api/query", json=invalid_request)
            assert response.status_code == 422
    
    def test_query_response_format(self, test_client):
        """Test query response follows expected format"""
        query_data = {"query": "test query"}
        
        response = test_client.post("/api/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields exist
        required_fields = ["answer", "sources", "session_id"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Check sources format
        for source in data["sources"]:
            assert "name" in source
            # link is optional
            if "link" in source:
                assert source["link"] is None or isinstance(source["link"], str)
    
    def test_courses_response_format(self, test_client):
        """Test courses endpoint response format"""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert "total_courses" in data
        assert "course_titles" in data
        
        # Check types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        
        # All course titles should be strings
        for title in data["course_titles"]:
            assert isinstance(title, str)
    
    def test_session_clear_response_format(self, test_client):
        """Test session clear response format"""
        response = test_client.delete("/api/sessions/test-session/clear")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert "success" in data
        assert "message" in data
        
        # Check types
        assert isinstance(data["success"], bool)
        assert isinstance(data["message"], str)


@pytest.mark.api
class TestAPIErrorHandling:
    """Test API error handling scenarios"""
    
    def test_malformed_json_request(self, test_client):
        """Test API handles malformed JSON gracefully"""
        # Send malformed JSON
        response = test_client.post(
            "/api/query",
            data="{'invalid': json}",  # Invalid JSON format
            headers={"content-type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_unsupported_http_method(self, test_client):
        """Test unsupported HTTP methods return correct errors"""
        
        # Test POST on GET-only endpoint
        response = test_client.post("/api/courses")
        assert response.status_code == 405  # Method Not Allowed
        
        # Test GET on POST-only endpoint  
        response = test_client.get("/api/query")
        assert response.status_code == 405
        
        # Test POST on DELETE-only endpoint
        response = test_client.post("/api/sessions/test/clear")
        assert response.status_code == 405
    
    def test_content_type_handling(self, test_client):
        """Test API handles different content types appropriately"""
        query_data = {"query": "test"}
        
        # Test with correct content type
        response = test_client.post("/api/query", json=query_data)
        assert response.status_code == 200
        
        # Test with form data instead of JSON
        response = test_client.post("/api/query", data=query_data)
        assert response.status_code == 422  # Should reject form data


@pytest.mark.api
@pytest.mark.integration
class TestAPIWithMockedRAGSystem:
    """Test API endpoints with various RAG system responses"""
    
    def test_query_with_sources(self, test_client, populated_rag_system, mock_anthropic_tool_flow):
        """Test query endpoint when RAG system returns sources"""
        # The conftest.py handles the mocking, so we can test the full flow
        query_data = {
            "query": "What is machine learning?",
            "session_id": None
        }
        
        response = test_client.post("/api/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert len(data["answer"]) > 0
        
        # Sources should be properly formatted
        for source in data["sources"]:
            assert "name" in source
            assert isinstance(source["name"], str)
    
    def test_multiple_sequential_queries(self, test_client):
        """Test multiple queries in sequence maintain session context"""
        session_id = None
        
        # First query
        query1_data = {"query": "What is AI?", "session_id": session_id}
        response1 = test_client.post("/api/query", json=query1_data)
        assert response1.status_code == 200
        session_id = response1.json()["session_id"]
        
        # Second query with same session
        query2_data = {"query": "Tell me more about it", "session_id": session_id}
        response2 = test_client.post("/api/query", json=query2_data)
        assert response2.status_code == 200
        
        # Should maintain same session ID
        assert response2.json()["session_id"] == session_id
    
    def test_concurrent_sessions(self, test_client):
        """Test that different sessions are handled independently"""
        # Create two different sessions
        query1_data = {"query": "Session 1 query", "session_id": "session-1"}
        query2_data = {"query": "Session 2 query", "session_id": "session-2"}
        
        response1 = test_client.post("/api/query", json=query1_data)
        response2 = test_client.post("/api/query", json=query2_data)
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Sessions should remain separate
        assert response1.json()["session_id"] == "session-1"
        assert response2.json()["session_id"] == "session-2"


def run_api_tests():
    """Run API tests manually for development/debugging"""
    print("Running API Endpoint Tests...")
    print("=" * 60)
    
    from conftest import create_test_app
    
    # Create test client
    app, temp_dir = create_test_app()
    client = TestClient(app)
    
    try:
        # Test basic endpoints
        print("\n1. Testing root endpoint...")
        test = TestAPIEndpoints()
        test.test_root_endpoint(client)
        print("✓ Root endpoint test passed")
        
        print("\n2. Testing query endpoint...")
        test.test_query_endpoint_success(client)
        
        print("\n3. Testing courses endpoint...")
        test.test_courses_endpoint_empty(client)
        print("✓ Courses endpoint test passed")
        
        print("\n4. Testing session clear endpoint...")
        test.test_clear_session_endpoint(client)
        print("✓ Session clear endpoint test passed")
        
        print("\n5. Testing error handling...")
        test.test_query_endpoint_invalid_request(client)
        test.test_nonexistent_endpoint(client)
        print("✓ Error handling tests passed")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("\n" + "=" * 60)
    print("API endpoint tests completed!")


if __name__ == "__main__":
    run_api_tests()