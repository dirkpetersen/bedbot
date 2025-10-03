# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BedBot is an AI chat assistant powered by AWS Bedrock that provides intelligent conversations and document analysis capabilities. It's a Flask-based web application with advanced PDF processing, vector store integration, and dual storage modes (S3/local).
It was developed to ensure that 100s of PDFs can be analysed is a RAG style application. To process a large number of documents the user can optionally activate a vector store with a checkbox. Each user should have access to a decicated slice of the vector store 

## Key Architecture

### Core Components
- **bedbot.py**: Main Flask application handling chat, file uploads, and AWS Bedrock integration
- **vector_store.py**: FAISS-based vector store for document embeddings and retrieval (optional)
- **templates/index.html**: Frontend web interface
- **requirements.txt**: Python dependencies

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

### Running the Application
```bash
# Local filesystem mode (no AWS S3)
python bedbot.py --no-bucket

# S3 storage mode with debug
python bedbot.py --debug

# Custom model
python bedbot.py --model us.anthropic.claude-sonnet-4-5-20250929-v1:0

# With vector store enabled
VECTOR_STORE=1 python bedbot.py
```

### Installation
```bash
pip install -r requirements.txt
```

### Testing AWS Configuration
```bash
# Verify AWS setup
aws bedrock list-foundation-models --region us-east-1

# Test specific profile
aws bedrock list-foundation-models --profile bedbot --region us-east-1
```

## Environment Configuration

### Required AWS Setup
- AWS credentials configured (via `aws configure` or environment)
- Bedrock model access requested in AWS console
- S3 permissions for bucket creation (S3 mode only)

### Environment Variables (.env file)
```bash
AWS_PROFILE=bedbot
AWS_DEFAULT_REGION=us-east-1
SECRET_KEY=your-secret-key-change-this
BEDROCK_MODEL=us.anthropic.claude-sonnet-4-5-20250929-v1:0
VECTOR_STORE=0  # Set to 1 to enable, note this should only enable the vector store, the user still needs to activate it with a checkbox 
PDF_LOCAL_CONVERT=0  # Set to 1 for local PDF processing
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
- ThreadPoolExecutor for concurrent PDF conversions
- Batch processing for vector store embeddings
- Memory-safe chunking strategies

## Important Implementation Details

### PDF Processing Strategy
The app supports two PDF conversion methods:
- **Bedrock Native**: Uses Bedrock's native document understanding (default)
- **Local Processing**: Uses PyMuPDF for text extraction (`PDF_LOCAL_CONVERT=1`)
Bedrock native is very accurate but 

### Vector Store Integration
- Optional FAISS-based semantic search
- Documents chunked with overlap for better retrieval
- Session-isolated vector stores with comprehensive analysis modes
- Automatic embedding generation using sentence-transformers

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

### Debugging Issues
- Enable `--debug` for detailed logging
- Check `/vector_stats` endpoint for vector store diagnostics
- Use `/debug_*` endpoints for document processing inspection
- Monitor AWS CloudTrail for API call issues

### Testing File Processing
- Use `/all_bedrock_models` to verify model access
- Upload test documents and check console logs
- Vector store endpoints provide detailed diagnostics
- PDF conversion supports both local and Bedrock processing

## Session and State Management

### Key Session Variables
- `session_id`: Unique identifier per browser session
- `chat_history`: Conversation history (markdown and HTML versions)
- `uploaded_files`: List of successfully uploaded files
- `pdf_files`: PDF-specific metadata including markdown conversion status
- `pdfs_initialized`: Whether PDFs have been sent to current Bedrock model
- `selected_model`: Current Bedrock model for the session

### State Reset Scenarios
- Model changes reset `pdfs_initialized` to re-send PDFs to new model
- File removal rebuilds document context from remaining files
- Session clearing removes all files and resets state

This architecture enables robust document analysis with intelligent conversation context while maintaining security and performance through careful resource management.
