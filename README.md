

# BedBot - AI Chat Assistant

BedBot is a powerful AI chat assistant powered by AWS Bedrock that allows you to have intelligent conversations and analyze documents. It supports multiple model providers including Claude, Nova, Llama, DeepSeek, and OpenAI models through Bedrock, with both text-based conversations and document analysis capabilities.


<!-- <img width="591" height="694" alt="image" src="https://github.com/user-attachments/assets/55520913-c8ea-4533-b747-25cf7325fe58" /> -->
<img width="508" height="689" alt="image" src="https://github.com/user-attachments/assets/d815620b-57d2-460f-b02c-6eef152c809f" />


## Table of Contents

- [BedBot - AI Chat Assistant](#bedbot---ai-chat-assistant)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Use Cases](#use-cases)
    - [Document Analysis and Comparison](#document-analysis-and-comparison)
    - [Content Creation and Review](#content-creation-and-review)
    - [Educational and Training](#educational-and-training)
    - [Business Intelligence](#business-intelligence)
  - [Prerequisites](#prerequisites)
  - [AWS Setup](#aws-setup)
    - [Creating a New AWS Profile](#creating-a-new-aws-profile)
  - [Installation](#installation)
  - [Configuration](#configuration)
    - [Environment Variables](#environment-variables)
    - [Model Selection](#model-selection)
  - [Usage](#usage)
    - [Basic Startup](#basic-startup)
    - [Basic Chat](#basic-chat)
    - [Document Upload and Analysis](#document-upload-and-analysis)
    - [Voice Input](#voice-input)
  - [Command Line Options](#command-line-options)
  - [Storage Modes](#storage-modes)
    - [Local Filesystem Mode (`--no-bucket`)](#local-filesystem-mode---no-bucket)
    - [S3 Storage Mode (default)](#s3-storage-mode-default)
  - [Troubleshooting](#troubleshooting)
    - [Common Issues](#common-issues)
      - [1. "Bedrock client not initialized"](#1-bedrock-client-not-initialized)
      - [2. "Model access denied" or "ValidationException"](#2-model-access-denied-or-validationexception)
      - [3. "Failed to create S3 bucket"](#3-failed-to-create-s3-bucket)
      - [4. File upload fails](#4-file-upload-fails)
      - [5. Voice input not working](#5-voice-input-not-working)
    - [Debug Mode](#debug-mode)
    - [Log Analysis](#log-analysis)
  - [Advanced Features](#advanced-features)
    - [Session Management](#session-management)
    - [Document Context](#document-context)
    - [PDF Processing](#pdf-processing)
    - [Security Features](#security-features)
  - [Security Considerations](#security-considerations)
    - [Data Privacy](#data-privacy)
    - [AWS Permissions](#aws-permissions)
    - [Network Security](#network-security)
    - [Recommended IAM Policy](#recommended-iam-policy)
  - [Contributing](#contributing)
    - [Development Setup](#development-setup)

## Features

- **AI-Powered Conversations**: Chat with AWS Bedrock models (Claude, Nova, Llama, DeepSeek, OpenAI) for intelligent responses
- **Document Analysis**: Upload and analyze multiple document types (PDF, TXT, DOCX, MD, JSON, CSV)
- **DOCX to PDF Conversion**: Automatic conversion of DOCX files to PDF using Pandoc (required for DOCX support)
- **Smart Extractor System**: LLM→Regex hybrid approach for comprehensive document analysis that bypasses context window limits
- **Vector Store Integration**: Optional FAISS-based semantic search with hierarchical chunking for enhanced document retrieval
- **PDF Processing**: Advanced PDF handling with automatic merging and content extraction
- **Voice Input**: Speech-to-text functionality for hands-free interaction
- **Dual Storage Modes**: Choose between local filesystem or AWS S3 for document storage
- **Session Management**: Persistent chat history and document context within sessions
- **Modern Web Interface**: Responsive, mobile-friendly chat interface with 2-column adaptive layout
- **Real-time Processing**: Live typing indicators, instant responses, and Server-Sent Events for upload progress
- **File Management**: Easy upload, preview, and removal of documents with drag-and-drop support
- **Comprehensive Extraction**: Extract structured data (applicants, URLs, emails) from large documents without context limits

## Use Cases

### Document Analysis and Comparison
- **Legal Document Review**: Upload contracts, agreements, or legal documents for analysis and comparison
- **Research Paper Analysis**: Compare multiple research papers, extract key findings, and identify relationships
- **Technical Documentation**: Analyze technical specifications, requirements documents, and implementation guides
- **Financial Report Analysis**: Review financial statements, compare quarterly reports, and extract insights
- **Large Document Processing**: Handle documents with millions of characters using Smart Extractor to bypass context limits
- **Application Processing**: Extract all applicants, candidates, or participants from large application documents

### Content Creation and Review
- **Content Summarization**: Upload long documents for concise summaries and key point extraction
- **Writing Assistance**: Get help with writing, editing, and improving documents
- **Translation and Localization**: Analyze documents in different languages and get translation assistance
- **Compliance Checking**: Compare documents against regulatory requirements or standards

### Educational and Training
- **Study Material Analysis**: Upload textbooks, papers, and study materials for Q&A sessions
- **Training Document Review**: Analyze training materials and create interactive learning experiences
- **Curriculum Development**: Compare educational standards and develop comprehensive curricula

### Business Intelligence
- **Market Research**: Analyze market reports, competitor analysis, and industry trends
- **Policy Analysis**: Review company policies, procedures, and compliance documents
- **Strategic Planning**: Analyze business plans, strategic documents, and performance reports

## Prerequisites

- Python 3.8 or higher
- AWS Account with Bedrock access
- AWS CLI installed and configured
- Modern web browser with JavaScript enabled

## AWS Setup

### Creating a New AWS Profile

If you haven't set up AWS yet, you'll need AWS credentials from your AWS administrator. Ideally, ask for a dedicated IAM user with `AmazonBedrockFullAccess` permissions.

1. **Install AWS CLI** (if not already installed):
   
   **Linux/Mac:**
   ```bash
   curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
   unzip awscliv2.zip
   sudo ./aws/install
   ```
   
   **Windows:**
   ```powershell
   msiexec.exe /i https://awscli.amazonaws.com/AWSCLIV2.msi
   ```

2. **Configure AWS Profile:**
   ```bash
   aws configure --profile bedbot
   ```
   
   Enter your credentials:
   ```
   AWS Access Key ID [None]: YOUR_ACCESS_KEY
   AWS Secret Access Key [None]: YOUR_SECRET_KEY
   Default region name [None]: us-east-1
   Default output format [None]: json
   ```

3. **For Enterprise/SSO Users:**
   If your organization uses AWS SSO, configure it instead:
   ```bash
   aws configure sso --profile bedbot
   ```

## Installation

1. **Clone or download BedBot:**
   ```bash
   git clone https://github.com/dirkpetersen/bedbot
   cd bedbot
   ```

2. **Create and activate Python virtual environment:**
   ```bash
   # Create virtual environment
   python -m venv .venv

   # Activate it
   # On Linux/macOS:
   source .venv/bin/activate

   # On Windows:
   .venv\Scripts\activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Pandoc for DOCX Support (Required)**

   To upload DOCX files, you must install Pandoc for automatic PDF conversion:

   **Ubuntu/Debian:**
   ```bash
   sudo apt-get install pandoc texlive-xetex
   ```

   **macOS:**
   ```bash
   brew install pandoc
   brew install --cask mactex
   ```

   **Windows:**
   - Download Pandoc from https://pandoc.org/installing.html
   - Install MiKTeX from https://miktex.org/download

4. **Optional: Vector Store Dependencies**

   Vector store support is optional. Choose one of the options below (activate .venv first):

   **CPU-based vector store:**
   ```bash
   pip install --no-cache-dir -r requirements-vectorstore.txt
   ```

   **GPU-accelerated vector store (requires NVIDIA CUDA 12):**
   ```bash
   pip install --no-cache-dir -r requirements-vectorstore-gpu.txt
   ```

5. **Verify AWS configuration:**
   ```bash
   aws bedrock list-foundation-models --profile bedbot --region us-west-2
   ```

## Configuration

### Environment Variables

Create a `.env` file in the project directory (optional):

```bash
AWS_PROFILE=bedbot
AWS_DEFAULT_REGION=us-west-2
BEDROCK_MODEL=us.anthropic.claude-sonnet-4-5-20250929-v1:0,us.amazon.nova-premier-v1:0
VECTOR_STORE=0
PDF_LOCAL_CONVERT=1
BEDROCK_TIMEOUT=900
```

**Environment Variable Details:**
- `AWS_PROFILE`: AWS profile to use for authentication
- `AWS_DEFAULT_REGION`: AWS region for Bedrock and S3 services
- `BEDROCK_MODEL`: Default Bedrock model ID to use
- `VECTOR_STORE`: Enable vector store functionality (0=disabled, 1=enabled)
- `PDF_LOCAL_CONVERT`: Use local PDF processing instead of Bedrock (0=Bedrock, 1=local)
- `BEDROCK_TIMEOUT`: Timeout in seconds for Bedrock API calls

### Model Selection

BedBot supports various Bedrock models. You can specify the model using the `--model` parameter:

- `us.anthropic.claude-sonnet-4-5-20250929-v1:0` - Latest Claude model with 1M token context (default)
- `us.anthropic.claude-haiku-4-5-20251001-v1:0` - Fast Claude Haiku model with 1M token context
- `global.anthropic.claude-opus-4-5-20251101-v1:0` - Most capable Claude Opus model (global cross-region)
- `us.amazon.nova-premier-v1:0` - Most capable Amazon Nova model
- `us.meta.llama4-scout-17b-instruct-v1:0` - Meta Llama 4 Scout model
- `us.qwen.qwen3-235b-a22b-2507-v1:0` - Alibaba Qwen 3 model
- `us.deepseek.r1-v1:0` - DeepSeek R1 model

## Usage

### Basic Startup

**Local filesystem mode (default):**
```bash
python bedbot.py
```

**S3 storage mode:**
```bash
python bedbot.py --no-bucket=false
```

**With custom model:**
```bash
python bedbot.py --model us.anthropic.claude-sonnet-4-5-20250929-v1:0
```

**Debug mode:**
```bash
python bedbot.py --debug
```

### Basic Chat

1. Open your browser to `http://localhost:5000`
2. Type your message in the chat input
3. Press Enter or click the send button
4. BedBot will respond using the configured Bedrock model

### Document Upload and Analysis

1. **Upload Documents:**
   - Click the upload area or drag and drop files
   - Supported formats: PDF, TXT, DOCX, MD, JSON, CSV (Note: .doc files not supported, use .docx)
   - Maximum file size: 4.5MB per file (local mode) or unlimited (S3 mode)
   - Maximum files: 1000 per session

2. **Document Processing:**
   - PDFs are processed using Bedrock's native document understanding capabilities
   - DOCX files are automatically converted to PDF using Pandoc (required - files rejected if Pandoc unavailable)
   - Text files are extracted and added to conversation context
   - Multiple PDFs are automatically merged in S3 mode for better analysis

3. **Ask Questions:**
   ```
   "What are the key points in the uploaded document?"
   "Compare the requirements in document A with the proposal in document B"
   "Summarize the main findings from all uploaded documents"
   ```

4. **Comprehensive Extraction (Smart Extractor):**
   For large documents that exceed context limits, use extraction queries:
   ```
   "list all applicants"
   "extract all GitHub URLs"
   "find all email addresses"
   "show all primary investigators"
   ```
   
   The Smart Extractor automatically:
   - Uses LLM to develop regex patterns from document samples
   - Applies patterns to full document content (no context limits)
   - Returns comprehensive, structured results

### Voice Input

1. Click the microphone button
2. Speak your message clearly
3. Click the stop button or the microphone again to finish
4. Your speech will be converted to text and can be edited before sending

**Note**: Voice input requires a modern browser with Web Speech API support (Chrome, Edge, Safari).

## Command Line Options

```bash
python bedbot.py [OPTIONS]
```

**Options:**
- `--no-bucket`: Use local filesystem instead of S3 bucket (default: false)
- `--debug`: Enable debug mode to print API messages and detailed logs
- `--model MODEL`: Specify Bedrock model to use (default: first model in BEDROCK_MODEL list)

**Examples:**
```bash
# Local mode with debug
python bedbot.py --no-bucket --debug

# S3 mode with Claude model
python bedbot.py --model us.anthropic.claude-3-5-sonnet-20241022-v2:0

# Claude Sonnet 4 model with debug
python bedbot.py --model us.anthropic.claude-sonnet-4-5-20250929-v1:0 --debug
```

## Storage Modes

### Local Filesystem Mode (`--no-bucket`)
- **Pros**: No AWS S3 costs, faster for small files, works offline for chat
- **Cons**: Limited file size (4.5MB), no persistence across server restarts
- **Use Case**: Development, testing, small document analysis

### S3 Storage Mode (default)
- **Pros**: Unlimited file sizes, automatic cleanup, better for large documents
- **Cons**: Requires S3 permissions, incurs AWS storage costs
- **Use Case**: Production use, large document analysis, team collaboration

## Troubleshooting

### Common Issues

#### 1. "Bedrock client not initialized"
**Cause**: AWS credentials not configured properly
**Solution**:
```bash
# Check AWS configuration
aws configure list --profile bedbot

# Test Bedrock access
aws bedrock list-foundation-models --profile bedbot --region us-west-2
```

#### 2. "Model access denied" or "ValidationException"
**Cause**: Model access not requested in Bedrock console
**Solution**:
1. Go to AWS Bedrock console → Model access
2. Request access to the model you're trying to use
3. Wait for approval (usually immediate)

#### 3. "Failed to create S3 bucket"
**Cause**: Insufficient S3 permissions or bucket name conflicts
**Solution**:
```bash
# Use local mode instead
python bedbot.py --no-bucket

# Or check S3 permissions
aws s3 ls --profile bedbot
```

#### 4. File upload fails
**Cause**: File too large, unsupported format, or DOCX without Pandoc
**Solution**:
- Check file size (4.5MB limit in local mode)
- Verify file format is supported
- For DOCX files: Install Pandoc (see installation section)
- Try S3 mode for larger files: remove `--no-bucket` flag

#### 5. Voice input not working
**Cause**: Browser doesn't support Web Speech API or microphone permissions denied
**Solution**:
- Use Chrome, Edge, or Safari
- Allow microphone access when prompted
- Check browser console for errors

#### 6. "Input is too long for requested model" error
**Cause**: Document exceeds Bedrock model's context window (often with large PDFs)
**Solution**:
- Smart Extractor should automatically handle this for extraction queries
- Use extraction queries like "list all applicants" to trigger Smart Extractor
- Enable vector store for better document chunking: `VECTOR_STORE=1`

#### 7. Vector store not working
**Cause**: FAISS dependencies not installed
**Solution**:
```bash
pip install faiss-cpu sentence-transformers
# Set environment variable
VECTOR_STORE=1 python bedbot.py
```

#### 8. DOCX files not uploading
**Cause**: Pandoc not installed, LaTeX not installed, or conversion failed
**Solution**:
- Install complete requirements: `sudo apt-get install pandoc texlive-xetex` (Ubuntu/Debian)
- macOS: `brew install pandoc && brew install --cask mactex`
- Verify Pandoc: `pandoc --version`
- Verify XeLaTeX: `xelatex --version`
- Alternative: Convert DOCX to PDF manually before uploading
- **Note**: DOCX files are never stored - they are always converted to PDF first

#### 9. .doc files rejected
**Cause**: .doc format not supported (only .docx supported for Word documents)
**Solution**:
- Convert .doc files to .docx format in Microsoft Word
- Alternative: Export .doc files as PDF from Microsoft Word
- Use LibreOffice to convert: File → Export as PDF or File → Save As → .docx

### Debug Mode

Enable debug mode for detailed logging:
```bash
python bedbot.py --debug
```

This will show:
- AWS API calls and responses
- File processing details
- Session management information
- Error stack traces

### Log Analysis

Check the console output for:
- AWS credential issues
- Model access problems
- File processing errors
- Network connectivity issues

## Advanced Features

### Advanced Prompt Engineering for RAG
BedBot implements sophisticated prompt engineering techniques to improve RAG (Retrieval-Augmented Generation) accuracy and relevance:

**Structured Prompt Templates:**
- **Role-Task-Instructions Framework**: Clear separation of AI role, specific task, and detailed instructions
- **Source Citation Integration**: Automatic inclusion of source attribution requirements in all prompts
- **Query Type Detection**: Different prompt templates for comprehensive analysis, extraction, cross-document comparison, and standard Q&A
- **Context Organization**: Logical separation of document context, conversation history, and user queries

**Template Types:**
1. **Comprehensive Analysis**: For queries requiring exhaustive document analysis
2. **Cross-Document Comparison**: For relationship analysis across multiple documents  
3. **Structured Extraction**: For finding and organizing specific information types
4. **Detailed Explanation**: For complex questions requiring thorough explanations
5. **Standard RAG**: For general document-based questions

**Benefits:**
- ✅ **Improved Accuracy**: Structured prompts reduce AI confusion and improve response quality
- ✅ **Source Attribution**: Automatic citation of sources in format (Source: document_name.pdf)
- ✅ **Model Identification**: Every response includes the generating model name for transparency
- ✅ **Consistent Formatting**: Standardized response structure across different query types
- ✅ **Context Optimization**: Better utilization of available context window
- ✅ **Reduced Hallucination**: Clear instructions minimize speculation beyond provided content

**Implementation:**
Uses `textwrap.dedent` for clean multiline prompt formatting and maintains separation of concerns between different prompt components.

**Automatic Template Selection:**
BedBot automatically selects the appropriate prompt template based on your query:
- "What are all the requirements?" → **Comprehensive Analysis** template
- "Compare document A with document B" → **Cross-Document Comparison** template  
- "List all applicants" → **Structured Extraction** template
- "How does this process work?" → **Detailed Explanation** template
- "What is the main point?" → **Standard RAG** template

**Model Identification:**
Every response automatically includes model identification for transparency:
```
[Your detailed analysis here...]

---
Generated by: us.anthropic.claude-sonnet-4-5-20250929-v1:0
```

**Extended Context Window Support:**
Automatically enables 1M token context window for Claude Sonnet 4 models:
- **Detection**: Any model ID containing "anthropic.claude-sonnet-4-" triggers extended context
- **Header Added**: `"anthropic_beta": ["context-1m-2025-08-07"]` enables 1M token processing
- **Model Support**: Currently works with Claude Sonnet 4 only (Claude Opus 4 uses standard context)
- **Transparent**: Works automatically with no configuration required
- **Performance**: Enables processing of extremely large documents without truncation

### Smart Extractor System
The Smart Extractor is a revolutionary LLM→Regex hybrid approach that bypasses traditional context window limitations:

**How It Works:**
1. **Sample Analysis**: Takes a representative sample (300K chars) from your large document
2. **Pattern Development**: LLM analyzes sample and develops custom regex patterns
3. **Full Document Processing**: Regex patterns process the entire document (2M+ chars) instantly
4. **Quality Filtering**: Statistical analysis removes noisy patterns automatically

**Benefits:**
- ✅ **No Context Limits**: Process documents of any size (tested with 2.5M+ characters)
- ✅ **Fast Processing**: Regex operates in milliseconds vs minutes for LLM chunking
- ✅ **Complete Results**: Guaranteed to find all matches without missing content
- ✅ **Cost Effective**: One LLM call instead of multiple chunked requests
- ✅ **Generic Approach**: Works with any document type without hard-coded rules

**Automatic Activation:**
Smart Extractor automatically activates for queries like:
- "list all applicants"
- "extract all GitHub URLs" 
- "find all email addresses"
- "show all primary investigators"

### Vector Store Integration
Optional FAISS-based semantic search with advanced capabilities:

**Features:**
- **Hierarchical Chunking**: Child (200 chars) → Parent (800 chars) → Grandparent (3200 chars)
- **Perfect Accuracy**: Uses IndexFlatIP for 100% accurate similarity search
- **Context Expansion**: Small chunks for precision, large chunks for context
- **Session Isolation**: Each user gets dedicated vector store slice

**When to Enable:**
- Large document collections (100+ pages)
- Semantic search requirements
- Cross-document relationship analysis
- Research and discovery workflows

**Usage:**
```bash
# Enable vector store
VECTOR_STORE=1 python bedbot.py

# In web interface, check "Use Vector Store" checkbox
```

### Session Management
- Each browser session maintains separate chat history and document context
- Sessions persist until explicitly cleared or server restart (local mode)
- S3 mode provides better session persistence
- Vector stores are session-isolated for privacy

### Document Context
- Uploaded documents become part of the conversation context
- BedBot can reference and compare multiple documents
- Context is maintained throughout the session
- Smart routing between vector store and direct processing

### PDF Processing
- **Dual Processing Modes**: Bedrock native (default) or local PyMuPDF conversion
- **Automatic PDF merging** for multiple uploads (S3 mode)
- **Native PDF understanding** using Bedrock's document capabilities
- **Intelligent content extraction** with markdown conversion
- **Real-time progress tracking** with Server-Sent Events

### Security Features
- Server-side session management
- Automatic cleanup of temporary files
- S3 bucket encryption and access controls
- No persistent storage of sensitive data
- FAISS logging suppression in production mode

## Security Considerations

### Data Privacy
- Documents are processed temporarily and cleaned up automatically
- S3 buckets are created with private access and encryption
- No data is permanently stored unless explicitly configured

### AWS Permissions
- Use dedicated IAM users with minimal required permissions
- Regularly rotate AWS access keys
- Monitor AWS CloudTrail for API usage

### Network Security
- BedBot runs on localhost by default
- Use HTTPS in production deployments
- Consider VPN or private network access for sensitive documents

### Recommended IAM Policy

For the IAM user account running BedBot create 2 new inline policies 

Policy: UseBedrock

```json
{
	"Version": "2012-10-17",
	"Statement": [
		{
			"Effect": "Allow",
			"Action": [
				"bedrock:InvokeModel",
				"bedrock:InvokeModelWithResponseStream",
				"bedrock:ListFoundationModels"
			],
			"Resource": "*"
		}
	]
}
```

Policy: AllowS3BucketsWithPrefix_bedbot

```json
{
	"Version": "2012-10-17",
	"Statement": [
		{
			"Effect": "Allow",
			"Action": "s3:*",
			"Resource": [
				"arn:aws:s3:::bedbot-*",
				"arn:aws:s3:::bedbot-*/*"
			]
		}
	]
}
```


## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup
```bash
# Create and activate virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt

# Run in debug mode
python bedbot.py --debug --no-bucket

# Run tests (if available)
python -m pytest tests/
```

---

**Need Help?** 
- Check the [Troubleshooting](#troubleshooting) section
- Enable `--debug` mode for detailed logs
- Review AWS Bedrock documentation
- Check AWS service status if experiencing connectivity issues

**Version**: 1.0.0  
**License**: MIT  
**AWS Services**: Bedrock, S3 (optional)
