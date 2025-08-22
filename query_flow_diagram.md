# RAG System Query Flow Diagram

```mermaid
sequenceDiagram
    participant User
    participant Frontend as Frontend<br/>(script.js)
    participant FastAPI as FastAPI<br/>(app.py)
    participant RAGSystem as RAG System<br/>(rag_system.py)
    participant AIGen as AI Generator<br/>(ai_generator.py)
    participant Claude as Anthropic<br/>Claude API
    participant ToolMgr as Tool Manager<br/>(search_tools.py)
    participant VectorStore as Vector Store<br/>(vector_store.py)
    participant ChromaDB as ChromaDB<br/>(Database)

    User->>Frontend: Types query & clicks send
    
    note over Frontend: Disable input, show loading
    
    Frontend->>FastAPI: POST /api/query<br/>{"query": "...", "session_id": "..."}
    
    FastAPI->>RAGSystem: rag_system.query(query, session_id)
    
    note over RAGSystem: Get conversation history<br/>from session_manager
    
    RAGSystem->>AIGen: generate_response(query, history, tools, tool_manager)
    
    AIGen->>Claude: API call with system prompt,<br/>query, and tool definitions
    
    alt Claude decides to search
        Claude-->>AIGen: Tool use request:<br/>search_course_content
        
        AIGen->>ToolMgr: execute_tool("search_course_content", ...)
        
        ToolMgr->>VectorStore: search(query, course_name, lesson_number)
        
        VectorStore->>ChromaDB: Semantic search with<br/>sentence transformers
        
        ChromaDB-->>VectorStore: Relevant course chunks<br/>with metadata
        
        VectorStore-->>ToolMgr: Formatted search results<br/>with sources
        
        ToolMgr-->>AIGen: Search results as tool output
        
        AIGen->>Claude: Follow-up API call with<br/>tool results
    end
    
    Claude-->>AIGen: Final response text
    
    AIGen-->>RAGSystem: Generated response
    
    note over RAGSystem: Extract sources from tool_manager<br/>Update conversation history
    
    RAGSystem-->>FastAPI: (response, sources)
    
    FastAPI-->>Frontend: QueryResponse:<br/>{"answer": "...", "sources": [...], "session_id": "..."}
    
    note over Frontend: Remove loading message<br/>Display response with sources<br/>Re-enable input
    
    Frontend-->>User: Shows AI response<br/>with collapsible sources

    note over User,ChromaDB: Tool-based RAG: Claude intelligently decides when to search based on query content
```

## Key Components

### Frontend (script.js)
- **Input Handling**: Captures user input and manages UI state
- **API Communication**: Sends POST requests to `/api/query`
- **Response Display**: Renders markdown responses with sources

### Backend API (app.py)
- **Request Validation**: Uses Pydantic models for type safety
- **Session Management**: Creates/maintains conversation sessions
- **RAG Orchestration**: Delegates to RAG system for processing

### RAG System (rag_system.py)
- **Central Coordinator**: Manages all system components
- **History Management**: Maintains conversation context
- **Tool Integration**: Provides search capabilities to AI

### AI Generator (ai_generator.py)
- **Claude Integration**: Handles Anthropic API communication
- **Tool Execution**: Processes function calls from Claude
- **Response Synthesis**: Combines search results into coherent answers

### Search Tools (search_tools.py)
- **Smart Search**: Semantic search with course/lesson filtering
- **Source Tracking**: Maintains references for UI display
- **Result Formatting**: Structures search output for Claude

### Vector Store (vector_store.py)
- **Semantic Search**: ChromaDB with sentence transformers
- **Content Storage**: Course chunks with rich metadata
- **Flexible Queries**: Supports filtering by course and lesson

## Query Types

1. **General Knowledge**: Claude answers directly without searching
2. **Course-Specific**: Claude searches first, then synthesizes response
3. **Contextual**: Uses conversation history for better understanding

The system uses **intelligent tool selection** where Claude autonomously decides when to search based on the query content, making it more efficient than traditional always-retrieve RAG systems.