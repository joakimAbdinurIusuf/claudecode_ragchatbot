# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

**Quick Start:**
```bash
chmod +x run.sh
./run.sh
```

**Manual Start:**
```bash
cd backend
uv run uvicorn app:app --reload --port 8000
```

**Setup Requirements:**
- Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Install dependencies: `uv sync`
- Create `.env` file with: `ANTHROPIC_API_KEY=your_key_here`

**Important:** Always use `uv` commands instead of `pip` directly. This project uses uv as the package manager.

## Package Management

**Use `uv` for all dependency operations:**
- Add packages: `uv add package_name`
- Remove packages: `uv remove package_name`  
- Install/sync: `uv sync`
- Run commands: `uv run command`
- Never use `pip install` directly

**Access Points:**
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Architecture Overview

This is a **tool-based RAG (Retrieval-Augmented Generation) system** that intelligently decides when to search course content vs. answering directly from Claude's knowledge.

### Core Architecture Pattern

The system uses a **tool-calling approach** where:
1. User queries go to Claude with available search tools
2. Claude autonomously decides whether to search based on query content
3. If searching, Claude calls `search_course_content` tool with parameters
4. Tool performs semantic search in ChromaDB and returns formatted results
5. Claude synthesizes final response using search context

### Key Components

**RAG System (`rag_system.py`)**: Central orchestrator that coordinates all components. Manages document loading, query processing, and conversation history.

**AI Generator (`ai_generator.py`)**: Handles Anthropic Claude API integration with tool execution. Uses a sophisticated two-phase approach: initial request with tools, then follow-up with tool results.

**Tool System (`search_tools.py`)**: Implements extensible tool architecture. `CourseSearchTool` performs semantic search with smart course name matching and lesson filtering. `ToolManager` handles tool registration and execution.

**Vector Store (`vector_store.py`)**: ChromaDB integration with sentence transformers for embeddings. Stores both course metadata and content chunks. Supports filtering by course name and lesson number.

**Document Processor (`document_processor.py`)**: Extracts structured course information from text files. Performs intelligent text chunking with sentence boundaries and overlap. Parses course titles, lesson numbers, and content hierarchy.

**Session Manager (`session_manager.py`)**: Maintains conversation context across queries with configurable history limits.

### Data Flow Architecture

1. **Document Processing**: Course files → structured parsing → chunked content + metadata
2. **Vector Storage**: Chunks → sentence transformer embeddings → ChromaDB collections
3. **Query Processing**: User input → Claude analysis → optional tool search → response synthesis
4. **Session Management**: Conversation history maintained for context continuity

### Configuration System

All settings centralized in `config.py`:
- Anthropic model: `claude-sonnet-4-20250514`
- Embedding model: `all-MiniLM-L6-v2`
- Chunk size: 800 characters with 100 character overlap
- Search results: 5 maximum returned
- Conversation history: 2 exchanges preserved

### Frontend Integration

Simple HTML/CSS/JS frontend with:
- Real-time chat interface with markdown rendering
- Collapsible sources display
- Course statistics sidebar
- Session persistence across page refreshes

### Tool-Based vs Traditional RAG

Unlike traditional RAG systems that always retrieve, this implementation:
- Lets Claude decide when searching is needed
- Reduces unnecessary vector store queries
- Handles general knowledge questions without search
- Provides more natural conversation flow