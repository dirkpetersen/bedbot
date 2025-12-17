# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BedBot is an AI chat assistant powered by AWS Bedrock that provides intelligent conversations and document analysis capabilities. It's a Flask-based web application with advanced PDF processing, vector store integration, and dual storage modes (S3/local). It was designed to analyze hundreds of PDFs in a RAG (Retrieval-Augmented Generation) style application. Users can optionally activate a vector store with a checkbox for semantic search across uploaded documents. Each user session has access to a dedicated isolated slice of the vector store.

## Key Architecture

### Core Components
- **bedbot.py**: Main Flask application handling chat, file uploads, and AWS Bedrock integration (~200KB, ~4000 lines)
- **vector_store.py**: FAISS-based vector store with hierarchical chunking (child→parent→grandparent) for document embeddings and retrieval (optional)
- **smart_extractor.py**: Hybrid LLM→Regex system for comprehensive extraction that bypasses context window limits
- **templates/index.html**: Frontend web interface with responsive design, drag-and-drop file upload, and Server-Sent Events for real-time progress
- **requirements.txt**: Core Python dependencies (Flask, boto3, PyMuPDF, markdown)
- **requirements-vectorstore.txt**: Optional vector store dependencies (faiss-cpu, sentence-transformers)
- **requirements-vectorstore-gpu.txt**: Optional GPU-accelerated vector store (faiss-gpu-cu12, requires CUDA 12)

### Storage Architecture
- **Dual Storage Modes**: 
  - S3 mode (default): Uses AWS S3 buckets with automatic cleanup
  - Local mode (`--no-bucket`): Uses temporary local directories
- **Session Management**: Each browser session gets isolated storage and document context
- **Vector Store**: Optional FAISS-based semantic search across uploaded documents

### Document Processing Pipeline
1. **Upload**: Files stored in session-specific locations (S3 prefix or local folder)
2. **PDF Processing**: 
   - Local conversion using PyMuPDF (fitz) when `PDF_LOCAL_CONVERT=1`
   - Bedrock conversion using native document understanding (default)
   - Parallel processing for multiple PDFs
3. **Vector Store Integration**: Documents chunked and indexed for semantic search
4. **Context Building**: Text content and PDF markdown fed to Bedrock for conversation

## Development Commands

### Initial Setup
```bash
# Create and activate virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

# Install dependencies (lightweight - core only)
pip install --no-cache-dir -r requirements.txt

# Optional: Install with vector store support - CPU (includes PyTorch/CUDA ~1.5GB)
pip install --no-cache-dir -r requirements-vectorstore.txt

# Optional: Install with vector store support - GPU accelerated (requires NVIDIA CUDA 12)
pip install --no-cache-dir -r requirements-vectorstore-gpu.txt
```

**Note on `--no-cache-dir`**: Used to prevent pip from caching the large PyTorch/CUDA binaries. This saves local disk space if you're not using the vector store functionality.

**GPU variant**: Use `requirements-vectorstore-gpu.txt` for GPU-accelerated vector search with `faiss-gpu-cu12` instead of `faiss-cpu`.
- Requires NVIDIA CUDA 12 toolkit installed
- Uses `faiss-gpu-cu12` package optimized for CUDA 12

### Running the Application
```bash
# Make sure .venv is activated first!

# Local filesystem mode (no AWS S3)
python bedbot.py --no-bucket

# S3 storage mode with debug
python bedbot.py --debug

# Custom model
python bedbot.py --model us.anthropic.claude-sonnet-4-5-20250929-v1:0

# With vector store enabled
VECTOR_STORE=1 python bedbot.py
```

### Testing AWS Configuration
```bash
# Verify AWS setup
aws bedrock list-foundation-models --region us-west-2

# Test specific profile
aws bedrock list-foundation-models --profile bedbot --region us-west-2
```

## Environment Configuration

### Required AWS Setup
- AWS credentials configured (via `aws configure` or environment)
- Bedrock model access requested in AWS console
- S3 permissions for bucket creation (S3 mode only)

### Environment Variables (.env file)
```bash
AWS_PROFILE=bedbot
AWS_DEFAULT_REGION=us-west-2
PORT=5001  # Flask server port (default: 5001)
BEDROCK_MODEL=us.anthropic.claude-sonnet-4-5-20250929-v1:0
VECTOR_STORE=0  # Set to 1 to enable, note this should only enable the vector store, the user still needs to activate it with a checkbox
PDF_LOCAL_CONVERT=1  # Set to 0 for Bedrock native PDF processing
BEDROCK_TIMEOUT=900  # Timeout in seconds
```

### Command Line Arguments
- `--no-bucket`: Use local filesystem instead of S3
- `--debug`: Enable debug logging and detailed API output
- `--model MODEL_ID`: Specify Bedrock model to use

## Code Architecture Patterns

### Session Management
- Flask-Session with filesystem backend
- Session-specific upload locations prevent cross-contamination
- Automatic cleanup on exit with signal handlers

### AWS Integration
- Boto3 clients initialized with configurable timeouts
- S3 bucket auto-creation with encryption and private access
- Bedrock Converse API for chat and document processing
- Graceful fallback between S3Location and bytes methods

### Error Handling
- Comprehensive try/catch blocks with detailed logging
- Graceful degradation (S3 → local, Bedrock → local PDF processing)
- User-friendly error messages distinguishing client vs server issues

### Parallel Processing
- ThreadPoolExecutor for concurrent PDF conversions (configurable via `BEDROCK_MAX_CONCURRENT`, default: 3)
- Batch processing for vector store embeddings (batch_size=32 for efficiency)
- Memory-safe chunking strategies with overlap
- Progress tracking via queues and SSE for real-time user feedback

## Important Implementation Details

### PDF Processing Strategy
The app supports two PDF conversion methods:
- **Bedrock Native** (default): Uses Bedrock's native document understanding with Converse API. Very accurate but consumes API tokens and has context limits.
- **Local Processing** (`PDF_LOCAL_CONVERT=1`): Uses PyMuPDF (fitz) for text extraction. Faster and no API cost, but may have lower accuracy for complex PDFs with tables/images.

### Smart Extractor System
The Smart Extractor is a revolutionary hybrid approach for comprehensive document analysis:
- **Phase 1 - Pattern Development**: LLM analyzes a representative sample (~300K chars) to develop custom regex patterns
- **Phase 2 - Full Processing**: Regex patterns process entire document (2M+ chars) instantly
- **Automatic Activation**: Triggered by extraction queries like "list all applicants", "find all emails", "extract GitHub URLs"
- **Benefits**: No context limits, complete results guaranteed, cost-effective (one LLM call), fast regex processing
- **Implementation**: `smart_extractor.py` - SmartExtractor class with `extract_comprehensive()` method

### Vector Store Integration
- Optional FAISS-based semantic search using IndexFlatIP for 100% accurate similarity
- **Hierarchical Chunking Strategy**:
  - Child chunks: 200 chars (precision matching)
  - Parent chunks: 800 chars (medium context)
  - Grandparent chunks: 3200 chars (broad context)
- Session-isolated vector stores for privacy (each session_id gets dedicated slice)
- Automatic embedding generation using sentence-transformers
- Vector store manager pattern: `initialize_vector_store_manager()` called once at startup
- Top-k retrieval with configurable results (default: 50 child chunks → expanded to parents/grandparents)

### Security Considerations
- Server-side session management with temporary directories
- S3 buckets created with full privacy and encryption
- Automatic cleanup of temporary files and buckets
- Input validation and secure filename handling

### Memory Management
- Batch processing for large documents
- Streaming responses for real-time chat
- Automatic cleanup on shutdown with signal handlers
- Memory-safe chunking and embedding generation

## Common Development Tasks

### Adding New Bedrock Models
1. Update `BEDROCK_MODEL` environment variable with comma-separated list
2. Models automatically detected and tested via `/test_models` endpoint
3. Frontend model selector populated dynamically
4. Extended context window (1M tokens) automatically enabled for Claude Sonnet 4 models (detection: `"anthropic.claude-sonnet-4-"` in model ID)

### Debugging Issues
- Enable `--debug` for detailed logging of Bedrock API calls and responses
- Check `/vector_stats` endpoint for vector store diagnostics (session counts, chunk statistics)
- Use `/all_bedrock_models` to list and test all configured models
- Check console logs for PDF conversion progress (parallel processing with ThreadPoolExecutor)
- Monitor AWS CloudTrail for API call issues and permission problems

### Testing File Processing
- Use `/all_bedrock_models` to verify model access
- Upload test documents and monitor real-time progress via Server-Sent Events (SSE)
- Vector store endpoints provide detailed diagnostics (embedding counts, retrieval stats)
- PDF conversion supports both local (PyMuPDF) and Bedrock native processing
- Test Smart Extractor with queries like "list all applicants" on large documents (2M+ chars)

## Key Flask Routes and Endpoints

### Main Routes
- **`/`** - Main chat interface (renders index.html)
- **`/chat`** (POST) - Main chat endpoint, handles user messages and document context
- **`/upload`** (POST) - File upload endpoint with SSE progress tracking
- **`/upload_progress/<upload_id>`** - Server-Sent Events stream for real-time upload status
- **`/remove_file/<filename>`** (POST) - Remove uploaded file from session
- **`/clear_session`** (POST) - Clear all files and reset chat history
- **`/list_files`** (GET) - List all uploaded files in current session

### Model Management Routes
- **`/all_bedrock_models`** (GET) - List and test all configured Bedrock models
- **`/test_models`** (POST) - Test accessibility of specific models
- **`/switch_model`** (POST) - Switch to different Bedrock model (resets pdfs_initialized)

### Diagnostic Routes
- **`/vector_stats`** (GET) - Vector store statistics and diagnostics
- **`/session_info`** (GET) - Current session information and state
- **`/debug_text_context`** (GET) - View current text-based document context
- **`/debug_pdf_context`** (GET) - View PDF document metadata and status

### Smart Extractor Integration
- Automatically triggered in `/chat` endpoint for extraction queries
- Pattern matching: "list all", "extract all", "find all", "show all"
- Bypasses standard RAG pipeline when activated
- Returns structured JSON results formatted for display

## Session and State Management

### Key Session Variables
- **`session_id`**: Unique identifier per browser session (UUID)
- **`chat_history`**: Conversation history stored as list of dicts with markdown and HTML versions
- **`uploaded_files`**: List of successfully uploaded files with metadata
- **`pdf_files`**: PDF-specific metadata including markdown conversion status and Bedrock processing state
- **`pdfs_initialized`**: Boolean flag - whether PDFs have been sent to current Bedrock model
- **`selected_model`**: Current Bedrock model ID for the session
- **`text_context`**: Accumulated text content from non-PDF files
- **`use_vector_store`**: Boolean - whether user has enabled vector store via checkbox

### State Reset Scenarios
- Model changes reset `pdfs_initialized` to re-send PDFs to new model
- File removal rebuilds document context from remaining files
- Session clearing removes all files and resets state
- Vector store cleared when session cleared or all files removed

### Session Isolation
- Flask-Session with filesystem backend (flask_session/ directory)
- Temporary directories per session (tempfile.mkdtemp)
- S3 prefixes per session (s3://bucket/session_id/)
- Vector store manager maintains per-session FAISS indices

This architecture enables robust document analysis with intelligent conversation context while maintaining security and performance through careful resource management.

## Important Code Patterns and Conventions

### Bedrock API Integration
- **Converse API**: All Bedrock interactions use the `converse()` method (not deprecated `invoke_model()`)
- **Extended Context**: Claude Sonnet 4 models automatically get `anthropic_beta: ["context-1m-2025-08-07"]` header
- **Timeout Configuration**: Bedrock client initialized with `Config(read_timeout=BEDROCK_TIMEOUT)` (default: 900s)
- **Error Handling**: Distinguish between `ClientError` (AWS/permissions) and other exceptions
- **Message Format**: Messages structured as `[{"role": "user", "content": [...]}]` with content blocks

### Document Context Building
- **Text Files**: Content accumulated in `session['text_context']` string
- **PDF Files**: Sent as document blocks in Bedrock Converse API with `documentFormat: "pdf"` and `source: bytes/s3Location`
- **Priority**: PDFs sent first in message, then text context, then user query
- **Re-initialization**: PDFs must be resent to Bedrock when model changes (`pdfs_initialized` flag)

### File Upload and Storage Patterns
- **Filename Security**: All filenames processed with `secure_filename()` from werkzeug
- **Session Isolation**: Files stored in session-specific directories/S3 prefixes
- **Cleanup**: `atexit.register()` and signal handlers ensure cleanup on shutdown
- **Progress Tracking**: Upload status tracked in `upload_status_queues` dict with threading locks

### Vector Store Patterns
- **Conditional Import**: Vector store only imported if `VECTOR_STORE=1` environment variable set
- **Manager Pattern**: Single `VectorStoreManager` instance created at startup via `initialize_vector_store_manager()`
- **Session Isolation**: Each session gets dedicated vector store via `session_id` parameter
- **Graceful Degradation**: All vector store code wrapped in try/except to handle missing dependencies

### Smart Extractor Activation
- **Query Detection**: Regex pattern matches extraction keywords: `r'\b(list|extract|find|show)\s+(all|every)\s+\w+'`
- **Automatic Routing**: When detected, bypasses normal RAG pipeline and uses `SmartExtractor.extract_comprehensive()`
- **Large Document Handling**: No context window limits - processes full document content
- **Pattern Caching**: Developed regex patterns cached in `SmartExtractor.pattern_cache` dict

### Logging and Debug Mode
- **Debug Flag**: `DEBUG_MODE` global variable controlled by `--debug` command line arg
- **Conditional Logging**: Detailed Bedrock API messages only logged when `DEBUG_MODE=True`
- **Logger Hierarchy**: Module-level loggers (`logger = logging.getLogger(__name__)`)
- **FAISS Logging**: Suppressed in production mode, enabled only when `DEBUG=1` environment variable set
