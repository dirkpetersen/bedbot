#! /usr/bin/env python3

import os, json, uuid, tempfile, shutil, atexit, argparse, random, string,  logging, textwrap
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, Response
import queue
import threading
from flask_session import Session
from werkzeug.utils import secure_filename
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
import markdown
import fitz  # PyMuPDF
import re
from dotenv import load_dotenv
import signal
import sys
import threading
import contextlib

# Map-Reduce extractor removed - using optimized Child-Parent-Grandparent approach instead


# Load environment variables from .env file
load_dotenv()

# Global status tracking for upload progress
upload_status_queues = {}
upload_status_lock = threading.Lock()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import vector store functionality
try:
    if os.getenv('VECTOR_STORE', '0').strip().lower() in ('1', 'true', 'yes', 'on'):
        from vector_store import initialize_vector_store_manager, get_vector_store_manager, is_vector_store_available, VECTOR_STORE_AVAILABLE
        VECTOR_STORE_MODULE_AVAILABLE = True
        logger.info("Vector store module loaded successfully")
    else:
        VECTOR_STORE_MODULE_AVAILABLE = False
        VECTOR_STORE_AVAILABLE = False
        logger.info("Vector store module not enabled (VECTOR_STORE=0)")
except ImportError as e:
    VECTOR_STORE_MODULE_AVAILABLE = False
    VECTOR_STORE_AVAILABLE = False
    logger.warning(f"Vector store module not available: {e}")

# Import smart extractor for comprehensive analysis
try:
    from smart_extractor import SmartExtractor
    SMART_EXTRACTOR_AVAILABLE = True
    logger.info("Smart extractor available")
except ImportError as e:
    SMART_EXTRACTOR_AVAILABLE = False
    logger.warning(f"Smart extractor not available: {e}")
    logger.info("Running without vector store functionality")

# Parse command line arguments
parser = argparse.ArgumentParser(description='BedBot - AI Chat Assistant')
parser.add_argument('--no-bucket', action='store_true', help='Use local filesystem instead of S3 bucket')
parser.add_argument('--debug', action='store_true', help='Enable debug mode to print API messages')
parser.add_argument('--model', action='store', help='Bedrock model to use', default='us.amazon.nova-premier-v1:0')
args = parser.parse_args()

# Configuration
USE_S3_BUCKET = not args.no_bucket
DEBUG_MODE = args.debug
MAX_FILE_SIZE = 4.5 * 1024 * 1024  # 4.5 MB per file
MAX_FILES_PER_SESSION = 1000
BEDROCK_MODEL = os.getenv('BEDROCK_MODEL', args.model)
# Vector Store Configuration
USE_VECTOR_STORE = (VECTOR_STORE_MODULE_AVAILABLE and VECTOR_STORE_AVAILABLE and 
                   os.getenv('VECTOR_STORE', '0').strip().lower() in ('1', 'true', 'yes', 'on'))
# PDF merging configuration removed - no longer supported
# PDF conversion configuration - temporarily disable to test
PDF_CONVERSION_ENABLED = os.getenv('PDF_CONVERSION', '1').strip().lower() in ('1', 'true', 'yes', 'on')
# Local PDF conversion using fitz instead of Bedrock
PDF_LOCAL_CONVERT = os.getenv('PDF_LOCAL_CONVERT', '0').strip().lower() in ('1', 'true', 'yes', 'on')
# Maximum concurrent Bedrock connections for parallel PDF conversion
BEDROCK_MAX_CONCURRENT = int(os.getenv('BEDROCK_MAXCON', '3'))
# PDF conversion model - use different model for PDF conversion if specified
PDF_CONVERT_MODEL = os.getenv('PDF_CONVERT_MODEL', BEDROCK_MODEL)
# Bedrock timeout configuration
BEDROCK_TIMEOUT = int(os.getenv('BEDROCK_TIMEOUT', '900'))
# PDF chunking configuration removed - no longer supported

if DEBUG_MODE:
    logger.info(f"PDF conversion enabled: {PDF_CONVERSION_ENABLED} (PDF_CONVERSION={os.getenv('PDF_CONVERSION', '1')})")
    logger.info(f"PDF local convert enabled: {PDF_LOCAL_CONVERT} (PDF_LOCAL_CONVERT={os.getenv('PDF_LOCAL_CONVERT', '0')})")
    logger.info(f"Vector store enabled: {USE_VECTOR_STORE} (VECTOR_STORE={os.getenv('VECTOR_STORE', '0')}, module_available={VECTOR_STORE_MODULE_AVAILABLE}, vector_store_available={VECTOR_STORE_AVAILABLE})")
    logger.info(f"Max concurrent Bedrock connections: {BEDROCK_MAX_CONCURRENT} (BEDROCK_MAXCON={os.getenv('BEDROCK_MAXCON', '3')})")
    logger.info(f"PDF conversion model: {PDF_CONVERT_MODEL} (PDF_CONVERT_MODEL={os.getenv('PDF_CONVERT_MODEL', 'default')})")
    logger.info(f"Chat model: {BEDROCK_MODEL} (BEDROCK_MODEL={os.getenv('BEDROCK_MODEL', 'default')}, --model={args.model})")
    # Log parsed models for debugging
    parsed_models = [model.strip() for model in BEDROCK_MODEL.split(',') if model.strip()]
    logger.info(f"Parsed models: {parsed_models} (count: {len(parsed_models)})")
    logger.info(f"Bedrock timeout: {BEDROCK_TIMEOUT}s (BEDROCK_TIMEOUT={os.getenv('BEDROCK_TIMEOUT', '900')})")
log_args_info = f"Using command line args: --no-bucket={args.no_bucket}, --debug={args.debug}, --model={args.model}"

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this')
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max file size

# Configure Flask-Session for server-side sessions
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_KEY_PREFIX'] = 'bedbot:'
app.config['SESSION_FILE_THRESHOLD'] = 100  # Max number of sessions before cleanup

# Flask session directory will be created on demand
temp_session_dir = None

# Track all session upload folders for cleanup (local mode only)
session_upload_folders = set()

# Remove complex locking - use Flask's built-in session handling

# S3 bucket for file storage (S3 mode only)
s3_bucket_name = None
bucket_name_file = None

# Global clients and managers
bedrock_client = None
s3_client = None
session_aws = None
profile_region = None
# Map-Reduce extractor removed

# Cleanup flag to prevent duplicate cleanup
cleanup_performed = False

# Initialize Flask-Session (will create session dir when first needed)
Session(app)

def send_status_update(status_session_id, stage, message, progress=None):
    """Send status update to frontend via Server-Sent Events"""
    with upload_status_lock:
        if status_session_id in upload_status_queues:
            status_data = {
                'stage': stage,
                'message': message,
                'timestamp': datetime.now().isoformat()
            }
            if progress is not None:
                status_data['progress'] = progress
            
            try:
                upload_status_queues[status_session_id].put_nowait(status_data)
            except queue.Full:
                # If queue is full, remove oldest item and add new one
                try:
                    upload_status_queues[status_session_id].get_nowait()
                    upload_status_queues[status_session_id].put_nowait(status_data)
                except queue.Empty:
                    pass

def generate_bucket_name():
    """Generate a unique bucket name with random suffix"""
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
    return f"bedbot-{random_suffix}"

@contextlib.contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr output unless in debug mode"""
    if DEBUG_MODE:
        # In debug mode, don't suppress anything
        yield
    else:
        # Suppress output in normal mode
        with open(os.devnull, 'w') as devnull:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            try:
                sys.stdout = devnull
                sys.stderr = devnull
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

def get_session_markdown_content(session_id: str, source_filter: str = None) -> str:
    """Get markdown content from session files for smart extraction"""
    try:
        if 'pdf_files' not in session or not session['pdf_files']:
            logger.warning("No PDF files in session for markdown extraction")
            return ""
        
        all_markdown = []
        
        for pdf_info in session['pdf_files']:
            filename = pdf_info['filename']
            
            # Skip if source filter specified and doesn't match
            if source_filter and source_filter not in filename:
                continue
                
            # Try to get already processed markdown content first
            if 'markdown_content' in pdf_info and pdf_info['markdown_content']:
                markdown_content = pdf_info['markdown_content']
                all_markdown.append(f"\n=== DOCUMENT: {filename} ===\n\n{markdown_content}\n")
                logger.info(f"📄 Added {len(markdown_content):,} chars from cached {filename}")
                continue
            
            # Get markdown content from the PDF processing
            if USE_S3_BUCKET:
                # S3 mode - get the already processed .md file
                s3_key = pdf_info.get('s3_key', '')
                md_s3_key = pdf_info.get('md_s3_key', '')  # Look for the markdown S3 key
                
                # If we don't have md_s3_key, construct it from the PDF key
                if not md_s3_key and s3_key:
                    md_s3_key = s3_key.replace('.pdf', '.md')
                
                if md_s3_key and s3_client:
                    try:
                        # Get the markdown file directly from S3
                        bucket_name = s3_bucket_name  # Use the global bucket name
                        logger.info(f"📥 Getting markdown from S3: {bucket_name}/{md_s3_key}")
                        
                        response = s3_client.get_object(Bucket=bucket_name, Key=md_s3_key)
                        markdown_content = response['Body'].read().decode('utf-8')
                        
                        if markdown_content:
                            all_markdown.append(f"\n=== DOCUMENT: {filename} ===\n\n{markdown_content}\n")
                            logger.info(f"📄 Added {len(markdown_content):,} chars from S3 markdown {filename}")
                    except Exception as e:
                        logger.warning(f"Failed to get S3 markdown for {filename}: {e}")
                        logger.info(f"   - Tried S3 key: {md_s3_key}")
                else:
                    logger.warning(f"Missing S3 markdown key for {filename}: md_s3_key={bool(md_s3_key)}, s3_client={bool(s3_client)}")
            else:
                # Local mode - get from local file
                local_path = pdf_info.get('local_path', '')
                if local_path and os.path.exists(local_path):
                    try:
                        markdown_content = convert_pdf_to_markdown_bedrock(local_path, is_s3=False)
                        if markdown_content:
                            all_markdown.append(f"\n=== DOCUMENT: {filename} ===\n\n{markdown_content}\n")
                            logger.info(f"📄 Added {len(markdown_content):,} chars from {filename}")
                    except Exception as e:
                        logger.warning(f"Failed to get markdown for {filename}: {e}")
                else:
                    logger.warning(f"Missing local path for {filename}: {local_path}")
        
        combined_markdown = "\n".join(all_markdown)
        logger.info(f"📋 Combined markdown content: {len(combined_markdown):,} characters from {len(all_markdown)} documents")
        
        return combined_markdown
        
    except Exception as e:
        logger.error(f"❌ Error getting session markdown content: {e}")
        return ""

def save_bucket_name(bucket_name):
    """Save bucket name to a temporary file"""
    global bucket_name_file
    if not bucket_name_file:
        bucket_name_file = tempfile.NamedTemporaryFile(mode='w', delete=False, prefix='bedbot_bucket_', suffix='.txt')
        bucket_name_file.write(bucket_name)
        bucket_name_file.close()
        if DEBUG_MODE:
            logger.info(f"Saved bucket name to: {bucket_name_file.name}")

def load_bucket_name():
    """Load bucket name from temporary file if it exists"""
    global bucket_name_file
    # Only load if we have a reference to the current session's bucket file
    if bucket_name_file and hasattr(bucket_name_file, 'name') and os.path.exists(bucket_name_file.name):
        try:
            with open(bucket_name_file.name, 'r') as f:
                bucket_name = f.read().strip()
                if bucket_name:
                    if DEBUG_MODE:
                        logger.info(f"Loaded existing bucket name: {bucket_name}")
                    return bucket_name
        except Exception as e:
            logger.error(f"Error reading bucket name file {bucket_name_file.name}: {e}")
    return None

def cleanup_bucket_name_file():
    """Clean up the bucket name file"""
    global bucket_name_file
    if bucket_name_file and hasattr(bucket_name_file, 'name') and os.path.exists(bucket_name_file.name):
        try:
            os.unlink(bucket_name_file.name)
            if DEBUG_MODE:
                logger.info(f"Cleaned up bucket name file: {bucket_name_file.name}")
        except Exception as e:
            logger.error(f"Error cleaning up bucket name file: {e}")

def create_s3_bucket():
    """Create S3 bucket for file storage"""
    global s3_bucket_name, USE_S3_BUCKET
    if not USE_S3_BUCKET or not s3_client:
        return True
    
    try:
        s3_bucket_name = generate_bucket_name()
        
        # Create bucket
        if profile_region == 'us-east-1':
            s3_client.create_bucket(Bucket=s3_bucket_name)
        else:
            s3_client.create_bucket(
                Bucket=s3_bucket_name,
                CreateBucketConfiguration={'LocationConstraint': profile_region}
            )
        
        # Set bucket policy to private
        s3_client.put_public_access_block(
            Bucket=s3_bucket_name,
            PublicAccessBlockConfiguration={
                'BlockPublicAcls': True,
                'IgnorePublicAcls': True,
                'BlockPublicPolicy': True,
                'RestrictPublicBuckets': True
            }
        )
        
        # Enable server-side encryption with AWS managed keys (AES256)
        s3_client.put_bucket_encryption(
            Bucket=s3_bucket_name,
            ServerSideEncryptionConfiguration={
                'Rules': [
                    {
                        'ApplyServerSideEncryptionByDefault': {
                            'SSEAlgorithm': 'AES256'
                        },
                        'BucketKeyEnabled': True
                    }
                ]
            }
        )
        
        # Save bucket name for Flask restart
        save_bucket_name(s3_bucket_name)
        
        if DEBUG_MODE:
            logger.info(f"Created S3 bucket: {s3_bucket_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create S3 bucket: {e}")
        logger.warning("Falling back to local filesystem mode due to S3 bucket creation failure")
        s3_bucket_name = None
        USE_S3_BUCKET = False
        return False

def delete_s3_bucket_with_timeout(timeout_seconds=10):
    """Delete S3 bucket with timeout to prevent hanging on shutdown"""
    def delete_bucket():
        delete_s3_bucket()
    
    thread = threading.Thread(target=delete_bucket)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout_seconds)
    
    if thread.is_alive():
        logger.warning(f"S3 bucket cleanup timed out after {timeout_seconds} seconds - bucket may need manual cleanup")

def delete_s3_bucket():
    """Delete S3 bucket and all its contents"""
    global s3_bucket_name
    if not USE_S3_BUCKET or not s3_client or not s3_bucket_name:
        if DEBUG_MODE:
            logger.info("Skipping S3 bucket deletion - not using S3 or no bucket name")
        return
    
    try:
        logger.info(f"Cleaning up S3 bucket: {s3_bucket_name}")
        
        # Check if bucket exists first
        try:
            s3_client.head_bucket(Bucket=s3_bucket_name)
            if DEBUG_MODE:
                logger.info(f"S3 bucket {s3_bucket_name} exists, proceeding with deletion")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                if DEBUG_MODE:
                    logger.info(f"S3 bucket {s3_bucket_name} doesn't exist - already deleted")
                return
            else:
                raise e
        
        # Delete all objects in bucket with timeout
        paginator = s3_client.get_paginator('list_objects_v2')
        object_count = 0
        
        # Use shorter config for cleanup operations
        cleanup_config = Config(
            read_timeout=5,
            connect_timeout=5,
            retries={'max_attempts': 1}
        )
        cleanup_s3_client = session_aws.client('s3', region_name=profile_region, config=cleanup_config)
        
        for page in paginator.paginate(Bucket=s3_bucket_name):
            if 'Contents' in page:
                objects = [{'Key': obj['Key']} for obj in page['Contents']]
                object_count += len(objects)
                cleanup_s3_client.delete_objects(
                    Bucket=s3_bucket_name,
                    Delete={'Objects': objects}
                )
        
        if object_count > 0:
            logger.info(f"Deleted {object_count} objects from bucket")
        
        # Delete bucket
        cleanup_s3_client.delete_bucket(Bucket=s3_bucket_name)
        logger.info(f"Successfully deleted S3 bucket: {s3_bucket_name}")
        
    except Exception as e:
        logger.warning(f"S3 bucket cleanup failed (bucket may need manual cleanup): {e}")

# Signal handler for graceful shutdown
def signal_handler(signum, frame):
    """Handle interrupt signals for graceful shutdown"""
    signal_name = 'SIGINT' if signum == signal.SIGINT else f'Signal {signum}'
    logger.info(f"Received {signal_name} - shutting down gracefully...")
    cleanup_resources()
    sys.exit(0)

# Register cleanup function to remove temp directories and S3 bucket on shutdown
def cleanup_resources():
    global temp_session_dir, session_upload_folders, cleanup_performed
    
    # Debug logging to understand the environment
    werkzeug_run_main = os.environ.get('WERKZEUG_RUN_MAIN')
    logger.info(f"cleanup_resources called: WERKZEUG_RUN_MAIN={werkzeug_run_main}, PID={os.getpid()}")
    
    # In debug mode with reloader:
    # - Parent process (reloader): WERKZEUG_RUN_MAIN is not set
    # - Child process (actual app): WERKZEUG_RUN_MAIN is set to 'true'
    # We want cleanup to happen in the child process only
    if werkzeug_run_main is None:
        # This is the reloader process in debug mode - don't cleanup
        logger.info("Skipping cleanup - this is the Flask reloader process (parent)")
        return
    
    # Prevent duplicate cleanup
    if cleanup_performed:
        logger.info("Skipping cleanup - already performed")
        return
    cleanup_performed = True
    
    logger.info("Starting cleanup process...")
    
    try:
        # Clean up S3 bucket if using S3 mode (with timeout)
        if USE_S3_BUCKET:
            delete_s3_bucket_with_timeout(timeout_seconds=8)
        
        # Clean up bucket name file
        cleanup_bucket_name_file()
        
        # Clean up all session upload folders (local mode only)
        for folder in list(session_upload_folders):
            try:
                if os.path.exists(folder):
                    shutil.rmtree(folder)
            except Exception as e:
                logger.warning(f"Could not remove session folder {folder}: {e}")
        
        # Clean up Flask session directory
        if temp_session_dir and os.path.exists(temp_session_dir):
            try:
                shutil.rmtree(temp_session_dir)
            except Exception as e:
                logger.warning(f"Could not remove session directory {temp_session_dir}: {e}")
        
        logger.info("Cleanup completed")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

def initialize_aws_clients():
    """Initialize AWS clients with proper configuration"""
    global bedrock_client, s3_client, session_aws, profile_region
    
    try:
        # Create a session that uses the current AWS profile
        session_aws = boto3.Session()
        
        # Get the region from the profile configuration
        profile_region = session_aws.region_name
        if not profile_region:
            # Fallback to default region if not set in profile
            profile_region = 'us-east-1'
            logger.warning("No region found in AWS profile, defaulting to us-east-1")
        
        # Configure clients with increased timeout for large document processing
        config = Config(
            read_timeout=BEDROCK_TIMEOUT,  # Configurable timeout via BEDROCK_TIMEOUT env var
            connect_timeout=60,  # 1 minute for connection
            retries={
                'max_attempts': 3,
                'mode': 'adaptive'
            }
        )
        
        bedrock_client = session_aws.client('bedrock-runtime', region_name=profile_region, config=config)
        s3_client = session_aws.client('s3', region_name=profile_region, config=config) if USE_S3_BUCKET else None
        
        # Using optimized Child-Parent-Grandparent vector store approach instead
        
        if DEBUG_MODE:
            logger.info(f"Initialized Bedrock client with profile: {session_aws.profile_name or 'default'}")
            logger.info(f"Using region: {profile_region}")
            logger.info(f"Configured clients with {BEDROCK_TIMEOUT}s read timeout for large document processing")
            if USE_S3_BUCKET:
                logger.info("S3 bucket mode enabled")
            else:
                logger.info("Local filesystem mode enabled (--no-bucket)")
            if USE_VECTOR_STORE:
                logger.info("Vector store mode enabled (independent of file storage)")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize AWS clients: {e}")
        bedrock_client = None
        s3_client = None
        return False

def initialize_vector_store_if_enabled():
    """Initialize vector store manager - independent of S3"""
    if USE_VECTOR_STORE:
        try:
            logger.info("🔄 Initializing vector store manager")
            vector_manager = initialize_vector_store_manager()
            if vector_manager:
                logger.info("✅ Vector store manager initialized successfully")
            else:
                logger.error("❌ Failed to initialize vector store manager")
        except Exception as e:
            logger.error(f"❌ Error initializing vector store manager: {e}")

def ensure_vector_store_configured():
    """Ensure vector store manager is properly configured"""
    if not USE_VECTOR_STORE:
        return True
        
    vector_manager = get_vector_store_manager()
    if not vector_manager:
        logger.error("❌ Vector store manager not available")
        # Try to initialize it
        try:
            logger.info("🔄 Attempting to initialize vector store manager")
            vector_manager = initialize_vector_store_manager()
            if vector_manager:
                logger.info("✅ Vector store manager initialized successfully")
                return True
            else:
                logger.error("❌ Failed to initialize vector store manager")
                return False
        except Exception as e:
            logger.error(f"❌ Error initializing vector store manager: {e}")
            return False
    
    return True

# Map-Reduce extractor removed - using Child-Parent-Grandparent approach

def handle_flask_restart_bucket():
    """Handle S3 bucket name loading during Flask debug restart"""
    global s3_bucket_name, bucket_name_file, USE_S3_BUCKET
    
    # During Flask debug restart, find and load the existing bucket name
    temp_dir = tempfile.gettempdir()
    bucket_files = []
    
    for filename in os.listdir(temp_dir):
        if filename.startswith('bedbot_bucket_') and filename.endswith('.txt'):
            filepath = os.path.join(temp_dir, filename)
            try:
                mtime = os.path.getmtime(filepath)
                bucket_files.append((mtime, filepath))
            except Exception as e:
                logger.error(f"Error checking bucket name file {filepath}: {e}")
    
    if bucket_files:
        # Sort by modification time (newest first) and use the most recent
        bucket_files.sort(reverse=True)
        most_recent_file = bucket_files[0][1]
        
        try:
            with open(most_recent_file, 'r') as f:
                s3_bucket_name = f.read().strip()
                if s3_bucket_name:
                    # Set the bucket_name_file reference for cleanup
                    bucket_name_file = type('obj', (object,), {'name': most_recent_file})()
                    if DEBUG_MODE:
                        logger.info(f"Flask debug restart detected - reusing existing S3 bucket: {s3_bucket_name}")
                else:
                    logger.warning("Flask debug restart detected but bucket name file is empty")
                    USE_S3_BUCKET = False
        except Exception as e:
            logger.error(f"Error reading bucket name file {most_recent_file}: {e}")
            USE_S3_BUCKET = False
    else:
        logger.warning("Flask debug restart detected but no existing bucket name found")
        USE_S3_BUCKET = False

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'md', 'json', 'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# PDF merging functionality removed - no longer supported

# PDF filename merging helper removed - no longer needed

# PDF chunking functionality removed - no longer supported

# PDF chunk conversion functionality removed - no longer supported

# PDF parallel chunking functionality removed - no longer supported

def pdf2markdown(pdf_input, is_bytes=False):
    """
    Convert PDF to markdown using fitz text extraction
    Args:
        pdf_input: Path to PDF file or PDF bytes
        is_bytes: True if pdf_input is bytes, False if it's a file path
    Returns:
        Markdown string ready for LLM consumption
    """
    try:
        # Use fitz text extraction with output suppression in normal mode
        with suppress_output():
            if is_bytes:
                doc = fitz.open(stream=pdf_input, filetype="pdf")
            else:
                doc = fitz.open(pdf_input)
            
            markdown_parts = []
            
            # Process each page with fitz
            total_pages = len(doc)
            pages_with_content = 0
            
            logger.info(f"PDF has {total_pages} pages - processing all pages")
            
            for page_num in range(total_pages):
                page = doc[page_num]
                page_markdown = f"\n## Page {page_num + 1}\n\n"
                
                try:
                    # Use basic text extraction
                    text = page.get_text("text")
                    if text and text.strip():
                        page_markdown += f"{text.strip()}\n\n"
                        pages_with_content += 1
                        
                        # Log page content for debugging (first few and last few pages)
                        if page_num < 3 or page_num >= total_pages - 3:
                            text_preview = text.strip()[:200] + "..." if len(text.strip()) > 200 else text.strip()
                            logger.info(f"📄 Page {page_num + 1} content preview: {text_preview}")
                        
                except Exception as e:
                    logger.warning(f"Error processing page {page_num + 1}: {e}")
                    page_markdown += "*Error extracting page content*\n\n"
                
                if page_markdown.strip() != f"## Page {page_num + 1}":
                    markdown_parts.append(page_markdown)
            
            logger.info(f"✅ Processed {total_pages} pages, {pages_with_content} pages contained text content")
            
            doc.close()
        
        # Explicit cleanup to prevent memory leaks
        import gc
        gc.collect()
        
        # Assemble final document
        if not markdown_parts:
            logger.error("❌ No content could be extracted from PDF")
            return "# Document\n\n*No content could be extracted from this PDF*"
        
        full_content = "".join(markdown_parts)
        
        # Only show conversion method in debug mode
        if DEBUG_MODE:
            doc_header = "# Document\n\n*Converted from PDF using fitz*\n\n"
            doc_footer = f"\n\n---\n*Document processed: {len(markdown_parts)} pages*"
        else:
            doc_header = "# Document\n\n"
            doc_footer = ""
        
        final_markdown = doc_header + full_content + doc_footer
        
        logger.info(f"✅ Final PDF conversion: {len(final_markdown):,} chars total")
        logger.info(f"📊 Content breakdown: {len(doc_header)} header + {len(full_content):,} content + {len(doc_footer)} footer")
        
        # Log beginning of final content for debugging
        content_start = final_markdown[:1000] + "..." if len(final_markdown) > 1000 else final_markdown
        logger.info(f"📝 Final markdown start: {content_start}")
        
        return final_markdown
        
    except Exception as e:
        logger.error(f"Error in pdf2markdown conversion: {e}")
        return f"# Document Conversion Error\n\n*Failed to convert PDF: {str(e)}*"


def convert_pdf_to_markdown_local(pdf_path_or_s3key, is_s3=False, s3_bucket=None):
    """Convert a PDF file to markdown using advanced PDF2Markdown class"""
    doc_name = os.path.basename(pdf_path_or_s3key).replace('.pdf', '')
    
    logger.info(f"Using advanced PDF2Markdown conversion for {doc_name}")
    
    try:
        if is_s3:
            # S3 mode - download PDF first
            if not s3_bucket:
                logger.error("S3 bucket name required for S3 mode")
                return None
            
            try:
                response = s3_client.get_object(Bucket=s3_bucket, Key=pdf_path_or_s3key)
                pdf_bytes = response['Body'].read()
                logger.info(f"Downloaded PDF from S3 for conversion: {doc_name} ({len(pdf_bytes)} bytes)")
                
                # Use the new pdf2markdown function with bytes
                markdown_content = pdf2markdown(pdf_bytes, is_bytes=True)
                logger.info(f"Advanced PDF conversion successful ({len(markdown_content)} characters)")
                return markdown_content
                
            except Exception as download_error:
                logger.error(f"Failed to download PDF from S3: {download_error}")
                return f"# {doc_name}\n\n*PDF conversion failed and could not download from S3*"
        else:
            # Local mode
            if not os.path.exists(pdf_path_or_s3key):
                logger.error(f"PDF file not found: {pdf_path_or_s3key}")
                return f"# {doc_name}\n\n*PDF file not found*"
            
            try:
                # Use the new pdf2markdown function with file path
                markdown_content = pdf2markdown(pdf_path_or_s3key, is_bytes=False)
                logger.info(f"Advanced PDF conversion successful ({len(markdown_content)} characters)")
                return markdown_content
                
            except Exception as conversion_error:
                logger.error(f"Failed to convert PDF: {conversion_error}")
                return f"# {doc_name}\n\n*PDF conversion failed: {str(conversion_error)}*"
            
    except Exception as e:
        logger.error(f"PDF conversion failed: {e}")
        return f"# {doc_name}\n\n*PDF conversion failed completely*"

def convert_pdf_to_markdown_bedrock(pdf_path_or_s3key, is_s3=False, s3_bucket=None, model_id=None):
    """Convert a PDF file to markdown using Bedrock model"""
    if PDF_LOCAL_CONVERT:
        # If local conversion is enabled, use fitz instead
        return convert_pdf_to_markdown_local(pdf_path_or_s3key, is_s3=is_s3, s3_bucket=s3_bucket)
    
    if not bedrock_client:
        logger.error("Bedrock client not initialized")
        return None
    
    if not model_id:
        model_id = PDF_CONVERT_MODEL
    
    # Retry logic for rate limiting
    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            # Prepare the prompt for markdown conversion
            conversion_prompt = """Please convert this PDF document to markdown format in full. 

IMPORTANT INSTRUCTIONS:
- Extract all text content from the document
- Preserve the document structure using appropriate markdown headers (# ## ###)
- Convert tables to markdown table format if present
- Include all relevant text content
- Format lists using markdown list syntax
- Preserve paragraph breaks and spacing
- If there are multiple sections, use appropriate header levels
- Do not add any commentary or explanation - just provide the markdown conversion

VERY IMPORTANT:
- Convert the entire document in full even if it is large, do not truncate or skip any sections
- If the document is too large to convert, return a single error message "document too large"
- If the document is empty or contains no text, return a single error message "empty document"
- If the document is not a valid format, return a single error message "invalid format"
"""
            
            content = []
            
            if is_s3:
                # S3 mode - try S3 location first, fall back to bytes if not supported
                if not s3_bucket:
                    logger.error("S3 bucket name required for S3 mode")
                    return None
                
                # Sanitize document name for Bedrock
                doc_name = os.path.basename(pdf_path_or_s3key).replace('.pdf', '')
                doc_name = re.sub(r'[^a-zA-Z0-9\s\-\(\)\[\]]', '_', doc_name)
                doc_name = re.sub(r'\s+', ' ', doc_name).strip()
                doc_name = doc_name[:50]
                
                # Try S3 location first (for models that support it)
                try:
                    # Get AWS account ID for bucket owner
                    account_id = session_aws.client('sts').get_caller_identity()['Account']
                    
                    logger.info(f"Converting PDF to markdown via S3 location: {doc_name} (s3://{s3_bucket}/{pdf_path_or_s3key})")
                    
                    content.append({
                        "document": {
                            "format": "pdf",
                            "name": doc_name,
                            "source": {
                                "s3Location": {
                                    "uri": f"s3://{s3_bucket}/{pdf_path_or_s3key}",
                                    "bucketOwner": account_id
                                }
                            }
                        }
                    })
                    
                    # Try the conversion with S3 location
                    content.append({
                        "text": conversion_prompt
                    })
                    
                    messages = [{
                        "role": "user",
                        "content": content
                    }]
                    
                    inference_config = {
                        "maxTokens": 4000,
                        "temperature": 0.3,
                        "topP": 0.9
                    }
                    
                    api_params = get_bedrock_api_params(model_id, messages, inference_config)
                    response = bedrock_client.converse(**api_params)
                    
                    markdown_content = response['output']['message']['content'][0]['text']
                    logger.info(f"Successfully converted PDF to markdown using S3 location ({len(markdown_content)} characters)")
                    return markdown_content
                    
                except ClientError as e:
                    error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                    if 'ValidationException' in str(e) and 'S3Location' in str(e):
                        logger.info(f"Model {model_id} doesn't support S3Location, falling back to chunked bytes method")
                        # Fall back to downloading and chunking for large PDFs
                        try:
                            response = s3_client.get_object(Bucket=s3_bucket, Key=pdf_path_or_s3key)
                            pdf_bytes = response['Body'].read()
                            
                            logger.info(f"Downloaded PDF from S3: {doc_name} ({len(pdf_bytes)} bytes)")
                            
                            # Process as single document (chunking removed)
                            content = [{
                                "document": {
                                    "format": "pdf",
                                    "name": doc_name,
                                    "source": {
                                        "bytes": pdf_bytes
                                    }
                                }
                            }]
                        except Exception as download_error:
                            logger.error(f"Failed to download PDF from S3: {download_error}")
                            return None
                    else:
                        # Other error, re-raise for retry logic
                        raise e
            else:
                # Local mode - read file bytes
                if not os.path.exists(pdf_path_or_s3key):
                    logger.error(f"PDF file not found: {pdf_path_or_s3key}")
                    return None
                    
                with open(pdf_path_or_s3key, 'rb') as f:
                    pdf_bytes = f.read()
                
                # Sanitize document name for Bedrock
                doc_name = os.path.basename(pdf_path_or_s3key).replace('.pdf', '')
                doc_name = re.sub(r'[^a-zA-Z0-9\s\-\(\)\[\]]', '_', doc_name)
                doc_name = re.sub(r'\s+', ' ', doc_name).strip()
                doc_name = doc_name[:50]
                
                logger.info(f"Converting PDF to markdown: {doc_name} ({len(pdf_bytes)} bytes)")
                
                # Process as single document (chunking removed)
                content.append({
                    "document": {
                        "format": "pdf",
                        "name": doc_name,
                        "source": {
                            "bytes": pdf_bytes
                        }
                    }
                })
            
            content.append({
                "text": conversion_prompt
            })
            
            messages = [{
                "role": "user",
                "content": content
            }]
            
            inference_config = {
                "maxTokens": 4000,
                "temperature": 0.3,  # Lower temperature for more consistent conversion
                "topP": 0.9
            }
            
            logger.info(f"Sending PDF conversion request to Bedrock model: {model_id} (attempt {attempt + 1})")
            
            api_params = get_bedrock_api_params(model_id, messages, inference_config)
            response = bedrock_client.converse(**api_params)
            
            markdown_content = response['output']['message']['content'][0]['text']
            logger.info(f"Successfully converted PDF to markdown ({len(markdown_content)} characters)")
            
            return markdown_content
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code in ['ServiceUnavailableException', 'ThrottlingException'] and attempt < max_retries - 1:
                # Rate limiting or service unavailable - wait and retry
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Rate limit hit on attempt {attempt + 1}, retrying in {delay} seconds...")
                import time
                time.sleep(delay)
                continue
            else:
                logger.error(f"Bedrock API error during PDF conversion: {e}")
                return None
        except Exception as e:
            logger.error(f"Unexpected error during PDF conversion: {e}")
            return None
    
    logger.error(f"Failed to convert PDF after {max_retries} attempts")
    return None

def convert_pdfs_parallel(pdf_conversion_tasks):
    """Convert multiple PDFs to markdown in parallel using ThreadPoolExecutor"""
    if not pdf_conversion_tasks:
        return {}
    
    # Determine max workers based on conversion method
    max_workers = BEDROCK_MAX_CONCURRENT if not PDF_LOCAL_CONVERT else min(len(pdf_conversion_tasks), 8)
    logger.info(f"Starting parallel PDF conversion for {len(pdf_conversion_tasks)} files (max concurrent: {max_workers}, method: {'local' if PDF_LOCAL_CONVERT else 'bedrock'})")
    
    results = {}
    
    def convert_single_pdf(task):
        """Convert a single PDF - wrapper for thread pool"""
        file_key, s3_key, s3_bucket, filename = task
        try:
            logger.info(f"[Parallel] Starting conversion: {filename}")
            
            markdown_content = None
            try:
                if PDF_LOCAL_CONVERT:
                    markdown_content = convert_pdf_to_markdown_local(s3_key, is_s3=True, s3_bucket=s3_bucket)
                else:
                    markdown_content = convert_pdf_to_markdown_bedrock(s3_key, is_s3=True, s3_bucket=s3_bucket, model_id=PDF_CONVERT_MODEL)
            except SystemError as sys_err:
                logger.error(f"[Parallel] System error during conversion: {filename} - {sys_err}")
                return file_key, None, f"System error: {str(sys_err)}"
            except MemoryError as mem_err:
                logger.error(f"[Parallel] Memory error during conversion: {filename} - {mem_err}")
                return file_key, None, f"Memory error: {str(mem_err)}"
            except Exception as inner_e:
                logger.error(f"[Parallel] Conversion error: {filename} - {inner_e}")
                return file_key, None, str(inner_e)
            
            logger.info(f"[Parallel] Completed conversion: {filename} ({len(markdown_content) if markdown_content else 0} chars)")
            return file_key, markdown_content, None
            
        except Exception as e:
            logger.error(f"[Parallel] Failed conversion: {filename} - {e}")
            return file_key, None, str(e)
    
    # Use ThreadPoolExecutor for parallel processing
    try:
        # In debug mode, reduce max_workers to prevent reloader conflicts
        if DEBUG_MODE and max_workers > 2:
            max_workers = 2
            logger.info(f"Reduced max_workers to {max_workers} for debug mode")
        
        # Further reduce if we detect memory pressure
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            if available_memory_gb < 2:  # Less than 2GB available
                max_workers = min(max_workers, 1)
                logger.warning(f"Low memory detected ({available_memory_gb:.1f}GB), reducing to {max_workers} workers")
        except ImportError:
            logger.info("psutil not available - cannot check memory pressure")
            
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(convert_single_pdf, task): task 
                for task in pdf_conversion_tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    file_key, markdown_content, error = future.result()
                    
                    results[file_key] = {
                        'markdown_content': markdown_content,
                        'error': error,
                        'filename': task[3]  # filename is the 4th element
                    }
                except Exception as e:
                    logger.error(f"Error getting result for task {task}: {e}")
                    # Add error result for failed task
                    file_key = f"{task[3]}_{len(results)}"  # Generate unique key
                    results[file_key] = {
                        'markdown_content': None,
                        'error': str(e),
                        'filename': task[3]
                    }
    except Exception as e:
        logger.error(f"Critical error in parallel PDF processing: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        # Return empty results to prevent app crash
        return {}
    
    logger.info(f"Completed parallel PDF conversion. Successful: {sum(1 for r in results.values() if r['markdown_content'])}, Failed: {sum(1 for r in results.values() if r['error'])}")
    return results

def read_file_content(filepath_or_s3key, is_s3=False):
    """Read content from uploaded file or return file info for PDF processing"""
    try:
        if is_s3:
            # S3 mode - filepath_or_s3key is the S3 key
            filename = os.path.basename(filepath_or_s3key).lower()
            
            if filename.endswith('.pdf'):
                # For PDFs, return file info for Bedrock document processing
                return {
                    'type': 'pdf',
                    's3_key': filepath_or_s3key,
                    'filename': os.path.basename(filepath_or_s3key)
                }
            else:
                # For text files, read content from S3
                response = s3_client.get_object(Bucket=s3_bucket_name, Key=filepath_or_s3key)
                return response['Body'].read().decode('utf-8', errors='ignore')
        else:
            # Local mode - filepath_or_s3key is the local file path
            filename = os.path.basename(filepath_or_s3key).lower()
            
            if filename.endswith('.pdf'):
                # For PDFs, return file info for Bedrock document processing
                return {
                    'type': 'pdf',
                    'path': filepath_or_s3key,
                    'filename': os.path.basename(filepath_or_s3key)
                }
            else:
                # For text files, read content normally
                with open(filepath_or_s3key, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
    except Exception as e:
        logger.error(f"Error reading file {filepath_or_s3key}: {e}")
        return None

def build_pdf_content(pdf_files):
    """Build PDF content for Bedrock messages, preferring markdown when available"""
    content = []
    
    for pdf_info in pdf_files:
        try:
            # Check if we have markdown version available
            has_markdown = pdf_info.get('has_markdown', False)
            doc_name = pdf_info['filename'].replace('.pdf', '')
            doc_name = re.sub(r'[^a-zA-Z0-9\s\-\(\)\[\]]', '_', doc_name)
            doc_name = re.sub(r'\s+', ' ', doc_name).strip()
            doc_name = doc_name[:50]
            
            if DEBUG_MODE:
                logger.info(f"build_pdf_content: Processing {pdf_info['filename']}, has_markdown={has_markdown}")
                if has_markdown:
                    logger.info(f"build_pdf_content: Markdown keys in pdf_info: {[k for k in pdf_info.keys() if 'md' in k.lower()]}")
            
            if has_markdown:
                # Use markdown content instead of PDF for faster processing
                if USE_S3_BUCKET and 'md_s3_key' in pdf_info:
                    # S3 mode - read markdown from S3
                    try:
                        response = s3_client.get_object(Bucket=s3_bucket_name, Key=pdf_info['md_s3_key'])
                        markdown_content = response['Body'].read().decode('utf-8', errors='ignore')
                        logger.info(f"Using markdown content from S3: {doc_name} ({len(markdown_content)} chars)")
                        
                        content.append({
                            "text": f"\n\n--- Content from {pdf_info['filename']} (PDF converted to markdown) ---\n{markdown_content}"
                        })
                        continue
                    except Exception as e:
                        logger.warning(f"Failed to read markdown from S3, falling back to PDF: {e}")
                        has_markdown = False
                elif 'md_path' in pdf_info and os.path.exists(pdf_info['md_path']):
                    # Local mode - read markdown from file
                    try:
                        with open(pdf_info['md_path'], 'r', encoding='utf-8', errors='ignore') as md_file:
                            markdown_content = md_file.read()
                        logger.info(f"Using markdown content from file: {doc_name} ({len(markdown_content)} chars)")
                        
                        content.append({
                            "text": f"\n\n--- Content from {pdf_info['filename']} (PDF converted to markdown) ---\n{markdown_content}"
                        })
                        continue
                    except Exception as e:
                        logger.warning(f"Failed to read markdown from file, falling back to PDF: {e}")
                        has_markdown = False
                else:
                    logger.warning(f"has_markdown=True but no markdown source found for {pdf_info['filename']}")
                    has_markdown = False
            
            # Fall back to PDF processing if no markdown or markdown read failed
            if USE_S3_BUCKET and 's3_key' in pdf_info:
                # S3 mode - try S3 location first, fall back to bytes if model doesn't support it
                s3_key = pdf_info['s3_key']
                
                if 'nooooooooova' in current_model.lower():
                    # Nova models support S3Location
                    try:
                        account_id = session_aws.client('sts').get_caller_identity()['Account']
                        logger.info(f"Adding PDF document from S3 location: {doc_name}")
                        
                        content.append({
                            "document": {
                                "format": "pdf",
                                "name": doc_name,
                                "source": {
                                    "s3Location": {
                                        "uri": f"s3://{s3_bucket_name}/{s3_key}",
                                        "bucketOwner": account_id
                                    }
                                }
                            }
                        })
                    except Exception as e:
                        logger.warning(f"Failed to get account ID, falling back to bytes method: {e}")
                        # Fall back to bytes method
                        response = s3_client.get_object(Bucket=s3_bucket_name, Key=s3_key)
                        pdf_bytes = response['Body'].read()
                        logger.info(f"Adding PDF document (downloaded from S3): {doc_name}")
                        
                        content.append({
                            "document": {
                                "format": "pdf",
                                "name": doc_name,
                                "source": {"bytes": pdf_bytes}
                            }
                        })
                else:
                    # Non-Nova models - use bytes method directly
                    response = s3_client.get_object(Bucket=s3_bucket_name, Key=s3_key)
                    pdf_bytes = response['Body'].read()
                    logger.info(f"Adding PDF document (downloaded from S3): {doc_name}")
                    
                    content.append({
                        "document": {
                            "format": "pdf",
                            "name": doc_name,
                            "source": {"bytes": pdf_bytes}
                        }
                    })
            else:
                # Local mode - read file bytes
                pdf_path = pdf_info['path']
                if not os.path.exists(pdf_path):
                    logger.error(f"PDF file not found: {pdf_path}")
                    continue
                    
                with open(pdf_path, 'rb') as f:
                    pdf_bytes = f.read()
                
                logger.info(f"Adding PDF document: {doc_name}")
                
                content.append({
                    "document": {
                        "format": "pdf",
                        "name": doc_name,
                        "source": {"bytes": pdf_bytes}
                    }
                })
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_info}: {e}")
    
    return content

def send_bedrock_message(messages):
    """Send a single message to Bedrock and return the response"""
    # Use session-selected model or default
    current_model = session.get('selected_model', BEDROCK_MODEL.split(',')[0].strip())
    inference_config = {"maxTokens": 1000, "temperature": 0.5, "topP": 0.9}
    
    logger.info(f"send_bedrock_message using model: {current_model}")
    
    try:
        api_params = get_bedrock_api_params(current_model, messages, inference_config)
        response = bedrock_client.converse(**api_params)
        return response['output']['message']['content'][0]['text']
    except ClientError as api_error:
        logger.error(f"send_bedrock_message failed with model {current_model}: {api_error}")
        # Handle S3Location fallback
        if 'ValidationException' in str(api_error) and 'S3Location' in str(api_error):
            logger.warning("Model doesn't support S3Location, retrying with bytes method")
            # This would need bytes fallback logic but for now just re-raise
            raise api_error
        else:
            raise api_error

def get_bedrock_api_params(model_id, messages, inference_config):
    """
    Build Bedrock API parameters with appropriate headers for different models.
    Adds 1M token context window for Claude Sonnet 4 models.
    """
    params = {
        "modelId": model_id,
        "messages": messages,
        "inferenceConfig": inference_config
    }
    
    # Add extended context window for Claude Sonnet 4 models only
    # Note: Claude Opus 4 doesn't support the context-1m-2025-08-07 beta flag
    if "anthropic.claude-sonnet-4-" in model_id:
        params["additionalModelRequestFields"] = {
            "anthropic_beta": ["context-1m-2025-08-07"]
        }
        logger.info(f"Added 1M token context window for Claude Sonnet 4 model: {model_id}")
    
    return params

def build_structured_rag_prompt(user_query, vector_context="", traditional_context="", conversation_history=None, query_type=None, current_model=None):
    """
    Build a structured RAG prompt using advanced prompt engineering techniques.
    Uses textwrap.dedent for clean multiline formatting with proper source citations.
    """
    if query_type is None:
        query_type = {}
    
    # Determine the primary context source
    primary_context = vector_context if vector_context else traditional_context
    
    if not primary_context:
        # No context available - add model identification to simple query
        if current_model:
            return f"{user_query}\n\nIMPORTANT: End your response with: '---\\nGenerated by: {current_model}'"
        return user_query
    
    # Build structured prompt with clear sections
    if query_type.get('is_comprehensive') or query_type.get('is_document_listing'):
        # Comprehensive analysis template
        structured_prompt = textwrap.dedent(f"""
            ROLE: You are an expert document analyst providing comprehensive analysis.
            
            TASK: Provide a complete, thorough analysis of the user's question using ALL relevant information from the provided documents.
            
            INSTRUCTIONS:
            - Analyze ALL provided document content thoroughly
            - Provide comprehensive information covering all relevant aspects
            - Do NOT summarize or truncate - give detailed, complete responses
            - Cite specific sources using format: (Source: document_name.pdf)
            - Organize information logically with clear structure
            - Include all relevant details, examples, and supporting information
            
            DOCUMENT CONTEXT:
            {primary_context}
            
            USER QUESTION: {user_query}
            
            RESPONSE REQUIREMENTS:
            - Provide exhaustive analysis covering all aspects
            - Use structured formatting (headers, lists, tables where appropriate)
            - Cite sources for each piece of information
            - Ensure no important information is missed
        """).strip()
        
    elif query_type.get('needs_full_context'):
        # Cross-document analysis template
        structured_prompt = textwrap.dedent(f"""
            ROLE: You are an expert analyst specializing in cross-document comparison and relationship analysis.
            
            TASK: Analyze relationships, patterns, and connections across ALL provided documents to answer the user's question.
            
            INSTRUCTIONS:
            - Compare and contrast information across different documents
            - Identify relationships, correlations, and patterns
            - Highlight agreements, disagreements, and complementary information
            - Provide synthesis that draws from multiple sources
            - Always cite sources using format: (Source: document_name.pdf)
            
            DOCUMENT CONTEXT:
            {primary_context}
            
            USER QUESTION: {user_query}
            
            ANALYSIS FRAMEWORK:
            1. Individual document insights
            2. Cross-document patterns and relationships
            3. Synthesis and comprehensive conclusions
            4. Source-attributed evidence for all claims
        """).strip()
        
    elif query_type.get('is_complex'):
        # Detailed explanation template
        structured_prompt = textwrap.dedent(f"""
            ROLE: You are a knowledgeable expert providing detailed explanations based on document analysis.
            
            TASK: Provide a thorough, well-explained response to the user's question using the provided document content.
            
            INSTRUCTIONS:
            - Give detailed explanations with supporting evidence from documents
            - Break down complex concepts into understandable parts
            - Provide context and background information where relevant
            - Use examples and specifics from the documents
            - Always cite sources using format: (Source: document_name.pdf)
            
            DOCUMENT CONTEXT:
            {primary_context}
            
            USER QUESTION: {user_query}
            
            RESPONSE STRUCTURE:
            - Clear, detailed explanation
            - Supporting evidence with source citations
            - Relevant examples and specifics
            - Context and implications
        """).strip()
        
    elif query_type.get('is_extraction'):
        # Structured extraction template
        structured_prompt = textwrap.dedent(f"""
            ROLE: You are a data extraction specialist focused on finding and organizing specific information.
            
            TASK: Extract and organize all instances of the requested information type from the provided documents.
            
            INSTRUCTIONS:
            - Find ALL instances of the requested information type
            - Present results in a clear, organized format (tables, lists)
            - Include source attribution for each item: (Source: document_name.pdf)
            - Ensure completeness - don't miss any relevant items
            - Group and categorize results logically
            
            DOCUMENT CONTEXT:
            {primary_context}
            
            EXTRACTION REQUEST: {user_query}
            
            OUTPUT FORMAT:
            - Structured presentation (table/list format)
            - Complete source attribution
            - Organized by category or document source
            - Summary count of total items found
        """).strip()
        
    else:
        # Standard RAG template with source citations
        structured_prompt = textwrap.dedent(f"""
            ROLE: You are a helpful assistant analyzing documents to answer user questions.
            
            TASK: Answer the user's question using the provided document content as your primary source of information.
            
            INSTRUCTIONS:
            - Base your response primarily on the provided document content
            - Always cite sources using format: (Source: document_name.pdf)
            - If information is not in the documents, clearly state this
            - Provide specific details and examples from the documents
            - Maintain accuracy and avoid speculation beyond the provided content
            
            DOCUMENT CONTEXT:
            {primary_context}
            
            USER QUESTION: {user_query}
            
            Please provide a helpful response based on the document content, with proper source citations.
        """).strip()
    
    # Add conversation history context if available
    if conversation_history and len(conversation_history) > 0:
        history_context = "\n\nRECENT CONVERSATION CONTEXT:\n"
        for i, entry in enumerate(conversation_history[-3:], 1):  # Last 3 exchanges
            history_context += f"Exchange {i}:\n"
            history_context += f"User: {entry.get('user', '')}\n"
            history_context += f"Assistant: {entry.get('bot', '')[:200]}...\n\n"
        
        structured_prompt = structured_prompt + history_context
    
    # Add model identification requirement to all prompts
    if current_model:
        model_instruction = f"\n\nIMPORTANT: End your response with: '---\\nGenerated by: {current_model}'"
        structured_prompt += model_instruction
    
    logger.info(f"Built structured RAG prompt: {len(structured_prompt):,} chars, type: {[k for k, v in query_type.items() if v]}, model: {current_model}")
    
    return structured_prompt

def call_bedrock(prompt, context="", pdf_files=None, conversation_history=None, send_pdfs=False, use_vector_store=False):
    """Call AWS Bedrock model using Converse API with document support and conversation history"""
    if not bedrock_client:
        return "Error: Bedrock client not initialized. Please check your AWS credentials."
    
    # Use session-selected model or default to BEDROCK_MODEL
    current_model = session.get('selected_model', BEDROCK_MODEL.split(',')[0].strip())
    
    # Validate the model is not empty and is in the allowed list
    if not current_model or not current_model.strip():
        current_model = BEDROCK_MODEL.split(',')[0].strip()
        logger.warning(f"Empty model found, using default: {current_model}")
    
    available_models = [model.strip() for model in BEDROCK_MODEL.split(',') if model.strip()]
    if current_model not in available_models:
        current_model = available_models[0]
        logger.warning(f"Invalid model found, using default: {current_model}")
    
    logger.info(f"Using model: {current_model} (from session: {session.get('selected_model', 'not set')})")
    
    # Test if the model is actually available
    try:
        api_params = get_bedrock_api_params(
            current_model,
            [{"role": "user", "content": [{"text": "test"}]}],
            {"maxTokens": 10, "temperature": 0.1}
        )
        test_response = bedrock_client.converse(**api_params)
        logger.info(f"Model {current_model} is accessible")
    except ClientError as test_error:
        logger.error(f"Model {current_model} test failed: {test_error}")
        if "ValidationException" in str(test_error) and "model identifier is invalid" in str(test_error):
            return f"❌ **Model Error:** The model `{current_model}` is not available in your AWS region or account. Please check:\n\n• Model is available in region `{profile_region}`\n• You have access to this model in AWS Bedrock\n• The model identifier is correct\n\nAvailable models may vary by region and account permissions."
    except Exception as test_error:
        logger.error(f"Model {current_model} test failed with unexpected error: {test_error}")
    
    try:
        messages = []
        
        # Debug logging
        logger.info(f"PDF files provided: {len(pdf_files) if pdf_files else 0}")
        logger.info(f"Send PDFs this call: {send_pdfs}")
        logger.info(f"Conversation history entries: {len(conversation_history) if conversation_history else 0}")
        
        if pdf_files:
            for i, pdf_info in enumerate(pdf_files):
                if USE_S3_BUCKET:
                    logger.info(f"PDF {i+1}: {pdf_info.get('filename', 'unknown')} at s3://{s3_bucket_name}/{pdf_info.get('s3_key', 'unknown')}")
                else:
                    logger.info(f"PDF {i+1}: {pdf_info.get('filename', 'unknown')} at {pdf_info.get('path', 'unknown')}")
        
        # Prepare PDF content if needed (skip if vector store is enabled by user)
        pdf_content = []
        if send_pdfs and pdf_files and not use_vector_store:
            logger.info(f"Including PDFs in conversation ({'first time' if not conversation_history else 'model switched'})")
            
            # Build PDF content using existing logic
            pdf_content = build_pdf_content(pdf_files)
        elif send_pdfs and pdf_files and use_vector_store:
            logger.info(f"Skipping PDF upload to Bedrock - using vector store retrieval instead ({len(pdf_files)} PDFs indexed)")
                
        # Add conversation history if provided (text-only, no document re-uploads)
        if conversation_history:
            logger.info(f"Adding {len(conversation_history)} conversation history entries as text-only")
            for i, entry in enumerate(conversation_history):
                try:
                    # Validate entry structure
                    if not isinstance(entry, dict):
                        logger.warning(f"Skipping conversation history entry {i}: not a dictionary")
                        continue
                    
                    if 'user' not in entry or 'bot' not in entry:
                        logger.warning(f"Skipping conversation history entry {i}: missing 'user' or 'bot' key")
                        continue
                    
                    messages.append({"role": "user", "content": [{"text": entry['user']}]})
                    messages.append({"role": "assistant", "content": [{"text": entry['bot']}]})
                except Exception as e:
                    logger.error(f"Error processing conversation history entry {i}: {e}")
                    continue
        
        # Create current message prompt with vector store context if enabled
        vector_context = ""
        # Initialize query analysis variables
        is_comprehensive_query = False
        is_document_listing = False
        needs_full_context = False
        complex_query_indicators = [
            'explain', 'how', 'why', 'what', 'where', 'when', 'which',
            'analyze', 'describe', 'detail', 'elaborate', 'clarify'
        ]
        
        if USE_VECTOR_STORE and use_vector_store and 'session_id' in session:
            try:
                vector_manager = get_vector_store_manager()
                if vector_manager:
                    # Get vector store stats for debugging
                    stats = vector_manager.get_session_stats(session['session_id'])
                    logger.info(f"Vector store stats: {stats}")
                    
                    # Enhanced comprehensive analysis detection including extraction patterns
                    comprehensive_keywords = [
                        'all', 'list all', 'all candidates', 'all resumes', 'comprehensive', 
                        'complete list', 'entire', 'every', 'controls', 'requirements', 
                        'sections', 'everything', 'full list', 'complete', 'summary',
                        'overview', 'catalog', 'inventory', 'enumerate',
                        # Add extraction patterns as suggested by Grok
                        'extract all', 'find all', 'identify all', 'show all', 'get all',
                        'github.com', 'github urls', 'all urls', 'all links', 'all names',
                        'all github', 'extract urls', 'list urls', 'find urls'
                    ]
                    is_comprehensive_query = any(keyword in prompt.lower() for keyword in comprehensive_keywords)
                    
                    # Separate extraction patterns for structured data
                    extraction_keywords = [
                        'github.com', 'github urls', 'all urls', 'all links', 'extract urls', 
                        'list urls', 'find urls', 'email', 'emails', 'phone'
                    ]
                    applicant_extraction_keywords = [
                        'list all applicant', 'all applicant', 'applicant names', 'all names',
                        'extract all applicant', 'find all applicant', 'show all applicant'
                    ]
                    is_extraction_query = any(keyword in prompt.lower() for keyword in extraction_keywords)
                    is_applicant_extraction = any(keyword in prompt.lower() for keyword in applicant_extraction_keywords)
                    
                    # Smart extraction using LLM→Regex hybrid approach
                    if is_applicant_extraction and SMART_EXTRACTOR_AVAILABLE:
                        logger.info("🚀 Detected applicant extraction query - using Smart Extractor (LLM→Regex hybrid)")
                        try:
                            # Get markdown content from session files
                            markdown_content = get_session_markdown_content(session['session_id'], source_filter=None)
                            
                            if markdown_content:
                                logger.info(f"📄 Got markdown content: {len(markdown_content):,} characters")
                                
                                # Initialize smart extractor
                                smart_extractor = SmartExtractor(bedrock_client, current_model)
                                
                                # Run smart extraction
                                extraction_result = smart_extractor.extract_comprehensive(
                                    full_content=markdown_content,
                                    extraction_type="applicants",
                                    sample_size=300000  # 300K sample for better pattern development
                                )
                                
                                if extraction_result and not extraction_result.get('error'):
                                    results = extraction_result.get('results', [])
                                    total_found = extraction_result.get('total_found', 0)
                                    patterns_used = extraction_result.get('patterns_used', 0)
                                    
                                    logger.info(f"🎯 Smart extraction found {total_found} applicants using {patterns_used} patterns")
                                    
                                    # Format results for display
                                    if results:
                                        formatted_response = f"# Smart Extraction Results\n\n"
                                        formatted_response += f"Found **{total_found} primary applicants** using {patterns_used} regex patterns:\n\n"
                                        
                                        # Create table
                                        formatted_response += "| # | Name | Pattern | Value |\n"
                                        formatted_response += "|---|------|---------|-------|\n"
                                        
                                        for i, result in enumerate(results, 1):
                                            name = result.get('value', 'Unknown')
                                            pattern_name = result.get('pattern_name', 'unknown')
                                            formatted_response += f"| {i} | {name} | {pattern_name} | {name} |\n"
                                        
                                        formatted_response += f"\n**Extraction Method**: LLM→Regex Hybrid (Smart Extractor)\n"
                                        formatted_response += f"**Document Size**: {len(markdown_content):,} characters\n"
                                        formatted_response += f"**Processing Time**: Fast (regex-based)\n"
                                        
                                        html_response = markdown.markdown(formatted_response, extensions=['tables'])
                                        
                                        # Update chat history
                                        chat_history = session.get('chat_history', [])
                                        chat_history.append({
                                            'user': user_message,
                                            'bot': formatted_response,
                                            'bot_html': html_response,
                                            'timestamp': datetime.now().isoformat()
                                        })
                                        session['chat_history'] = chat_history
                                        session.modified = True
                                        
                                        logger.info("✅ Smart extraction complete")
                                        return jsonify({
                                            'response': html_response,
                                            'extraction_used': True,
                                            'extraction_type': 'Smart Extractor (LLM→Regex Hybrid)',
                                            'total_found': total_found,
                                            'patterns_used': patterns_used
                                        })
                                    else:
                                        logger.warning("⚠️ Smart extraction found no results")
                                        
                                else:
                                    error_msg = extraction_result.get('error', 'Unknown error')
                                    logger.error(f"❌ Smart extraction failed: {error_msg}")
                            else:
                                logger.warning("⚠️ No markdown content available for smart extraction")
                                    
                        except Exception as e:
                            logger.error(f"❌ Smart extraction failed: {e}")
                            # Fall back to normal processing
                            pass
                    
                    # Enhanced document analysis patterns
                    document_analysis_patterns = [
                        'list all controls', 'all controls', 'all requirements', 'list controls', 
                        'show controls', 'what are the controls', 'summarize controls',
                        'show all', 'list everything', 'complete analysis', 'full analysis',
                        'document summary', 'key points', 'main sections', 'all sections',
                        'what does the document contain', 'document contents'
                    ]
                    is_document_listing = any(pattern in prompt.lower() for pattern in document_analysis_patterns)
                    
                    # Check for document-wide queries that need full context
                    full_context_patterns = [
                        'compare', 'contrast', 'relationship', 'correlation', 'across',
                        'throughout', 'entire document', 'whole document', 'full document'
                    ]
                    needs_full_context = any(pattern in prompt.lower() for pattern in full_context_patterns)
                    
# Old Map-Reduce extraction code removed - using enhanced full document retrieval instead
                    
                    if is_comprehensive_query or is_document_listing or needs_full_context:
                        logger.info("Detected comprehensive analysis request - retrieving from all documents")
                        # Significantly increase limits for document listing tasks
                        # Use different retrieval strategies based on query type
                        if needs_full_context:
                            # For comparison/relationship queries, get maximum context using full retrieval
                            vector_context = vector_manager.get_all_documents_for_session(
                                session_id=session['session_id'],
                                max_chars=400000,  # 400K chars for cross-document analysis
                                force_full_retrieval=True
                            )
                            logger.info(f"Full context retrieval: {len(vector_context)} chars for cross-document analysis")
                        else:
                            # For extraction queries, use targeted search with parent expansion instead of full retrieval
                            # This avoids context limits while getting comprehensive results
                            vector_context = vector_manager.get_context_for_session(
                                session_id=session['session_id'],
                                query=prompt,
                                max_chunks=150,  # Higher chunk count for comprehensive extraction  
                                max_chars=500000,  # Higher limit for grandparent expansion
                                extraction_mode=True,  # Enable extraction mode
                                expansion_level='grandparent'  # Use largest context chunks
                            )
                            logger.info(f"Comprehensive retrieval: {len(vector_context)} chars for document listing")
                    else:
                        logger.info("Using targeted vector search")
                        # Enhanced targeted search with better parameters
                        search_chunks = 30  # Default for normal queries
                        search_chars = 25000  # Default char limit
                        
                        # Increase search scope for complex queries
                        complex_query_indicators = [
                            'explain', 'how', 'why', 'what', 'where', 'when', 'which',
                            'analyze', 'describe', 'detail', 'elaborate', 'clarify'
                        ]
                        if any(indicator in prompt.lower() for indicator in complex_query_indicators):
                            search_chunks = 50
                            search_chars = 50000
                            logger.info("Enhanced search for complex query")
                        
                        vector_context = vector_manager.get_context_for_session(
                            session_id=session['session_id'],
                            query=prompt,
                            max_chunks=search_chunks,
                            max_chars=search_chars,
                            extraction_mode=is_extraction_query
                        )
                    
                    if vector_context:
                        logger.info(f"Retrieved vector context: {len(vector_context)} chars")
                        
                        # Check if retrieved context seems insufficient for comprehensive queries
                        if (is_comprehensive_query or is_document_listing) and len(vector_context) < 10000:
                            logger.warning(f"Vector context ({len(vector_context)} chars) may be insufficient for comprehensive query")
                            # Try alternative retrieval strategy
                            try:
                                alternative_context = vector_manager.get_all_documents_for_session(
                                    session_id=session['session_id'],
                                    max_chars=None,  # No limit for vector store mode
                                    force_full_retrieval=True  # Use comprehensive method
                                )
                                if alternative_context and len(alternative_context) > len(vector_context):
                                    vector_context = alternative_context
                                    logger.info(f"Using alternative retrieval: {len(vector_context)} chars")
                            except Exception as alt_error:
                                logger.warning(f"Alternative retrieval failed: {alt_error}")
                    else:
                        logger.info("No relevant vector context found - check if documents are properly indexed")
                        
                        # Try to index existing documents that were uploaded without vector store
                        if pdf_files and len(pdf_files) > 0:
                            logger.info("Attempting to index existing documents for vector store")
                            try:
                                indexed_count = 0
                                for pdf_file in pdf_files:
                                    # Check if we have markdown content for this PDF
                                    if 'markdown_content' in pdf_file and pdf_file['markdown_content']:
                                        success = vector_manager.add_document_from_content(
                                            session_id=session['session_id'],
                                            content=pdf_file['markdown_content'],
                                            source_filename=pdf_file['filename'],
                                            metadata={'type': 'pdf_markdown', 'indexed_on_demand': True}
                                        )
                                        if success:
                                            indexed_count += 1
                                            logger.info(f"Successfully indexed: {pdf_file['filename']}")
                                
                                if indexed_count > 0:
                                    logger.info(f"Successfully indexed {indexed_count} documents for vector store")
                                    # Retry the vector search now that documents are indexed
                                    if is_comprehensive_query or is_document_listing or needs_full_context:
                                        vector_context = vector_manager.get_all_documents_for_session(
                                            session_id=session['session_id'],
                                            max_chars=None,
                                            force_full_retrieval=True
                                        )
                                        if vector_context:
                                            logger.info(f"Retrieved context after indexing: {len(vector_context)} chars")
                                    else:
                                        vector_context = vector_manager.get_context_for_session(
                                            session_id=session['session_id'],
                                            query=prompt,
                                            max_chunks=20,
                                            max_chars=None,
                                            extraction_mode=extraction_keywords and any(keyword in prompt.lower() for keyword in extraction_keywords) if 'extraction_keywords' in locals() else False
                                        )
                                        if vector_context:
                                            logger.info(f"Retrieved targeted context after indexing: {len(vector_context)} chars")
                                else:
                                    logger.warning("No documents could be indexed - they may not have markdown content")
                            except Exception as indexing_error:
                                logger.error(f"Error indexing existing documents: {indexing_error}")
                        
                        # Fallback: try to get any available context for comprehensive queries
                        if not vector_context and (is_comprehensive_query or is_document_listing):
                            try:
                                logger.info("Attempting fallback retrieval for comprehensive query")
                                fallback_context = vector_manager.get_all_documents_for_session(
                                    session_id=session['session_id'],
                                    max_chars=300000,  # 300K chars for fallback
                                    force_full_retrieval=True
                                )
                                if fallback_context:
                                    vector_context = fallback_context
                                    logger.info(f"Fallback retrieval successful: {len(vector_context)} chars")
                            except Exception as fallback_error:
                                logger.error(f"Fallback retrieval failed: {fallback_error}")
                else:
                    logger.warning("Vector store manager not available")
            except Exception as e:
                logger.error(f"Error retrieving vector context: {e}")
        
        # Build structured prompt using advanced RAG prompt engineering
        enhanced_prompt = build_structured_rag_prompt(
            user_query=prompt,
            vector_context=vector_context,
            traditional_context=context if not send_pdfs else "",
            conversation_history=conversation_history,
            query_type={
                'is_comprehensive': is_comprehensive_query,
                'is_document_listing': is_document_listing,
                'needs_full_context': needs_full_context,
                'is_complex': any(indicator in prompt.lower() for indicator in complex_query_indicators),
                'is_extraction': any(keyword in prompt.lower() for keyword in extraction_keywords) if 'extraction_keywords' in locals() else False
            },
            current_model=current_model
        )
        
        # Add current user message with PDFs if needed
        current_content = []
        
        # Add PDFs first if this is a re-initialization
        if pdf_content:
            current_content.extend(pdf_content)
            logger.info(f"Added {len(pdf_content)} PDF content items to current message")
        
        # Add the user's text message
        current_content.append({"text": enhanced_prompt})
        
        messages.append({
            "role": "user", 
            "content": current_content
        })
        
        # Adjust inference parameters based on query type
        max_tokens = 4000  # Default
        temperature = 0.7  # Default
        
        # Use different parameters for comprehensive analysis
        if is_comprehensive_query or is_document_listing or needs_full_context:
            max_tokens = 4000  # Keep same for now, but could increase if needed
            temperature = 0.5  # Lower temperature for more structured, comprehensive responses
            logger.info("Using comprehensive analysis inference parameters")
        elif any(indicator in prompt.lower() for indicator in complex_query_indicators):
            temperature = 0.6  # Slightly lower for detailed explanations
            logger.info("Using detailed explanation inference parameters")
        
        inference_config = {
            "maxTokens": max_tokens,
            "temperature": temperature,
            "topP": 0.9
        }
        
        logger.info(f"Sending request to Bedrock with {len(messages)} message entries")
        
        # Debug: Print messages JSON structure if debug mode is enabled (truncated for readability)
        if DEBUG_MODE and messages:  # Only log if we have messages
            logger.info("=== DEBUG: Converse API Messages Structure ===")
            messages_json = json.dumps(messages, indent=2, default=str)
            if len(messages_json) > 2000:  # Truncate long messages
                truncated = messages_json[:1000] + f"\n... [TRUNCATED {len(messages_json)-2000} chars] ...\n" + messages_json[-1000:]
                logger.info(truncated)
            else:
                logger.info(messages_json)
            logger.info("=== END DEBUG ===")
        
        try:
            api_params = get_bedrock_api_params(current_model, messages, inference_config)
            response = bedrock_client.converse(**api_params)
            
            # Handle different response structures for different models
            try:
                return response['output']['message']['content'][0]['text']
            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"Unexpected response structure from model {current_model}: {e}")
                logger.error(f"Full response structure: {json.dumps(response, indent=2, default=str)}")
                
                # Try alternative response structure parsing
                if 'output' in response and 'message' in response['output']:
                    message = response['output']['message']
                    if 'content' in message and isinstance(message['content'], list) and len(message['content']) > 0:
                        
                        # For OpenAI models, look for the actual text response (usually not in the first item)
                        for i, content_item in enumerate(message['content']):
                            logger.info(f"Content item {i} keys: {list(content_item.keys()) if isinstance(content_item, dict) else 'not a dict'}")
                            
                            # Skip reasoning content, look for actual text response
                            if isinstance(content_item, dict) and 'text' in content_item:
                                logger.info(f"Found text content in item {i}")
                                return content_item['text']
                            
                            # Try other possible keys
                            for key in ['message', 'content', 'response']:
                                if isinstance(content_item, dict) and key in content_item:
                                    logger.info(f"Found content using key: {key} in item {i}")
                                    return content_item[key]
                
                return f"❌ **Response Parsing Error:** Unable to extract text from model response. Response structure: {type(response)}"
            
        except ClientError as api_error:
            # Check if this is an S3Location validation error (only for initial PDF upload)
            if ('ValidationException' in str(api_error) and 'S3Location' in str(api_error) and 
                USE_S3_BUCKET and pdf_files and send_pdfs):
                logger.warning(f"Model {current_model} doesn't support S3Location, retrying with bytes method")
                
                # Rebuild content with bytes method for PDFs (only for initial upload)
                retry_content = []
                
                # Re-add PDF documents using bytes method
                for pdf_info in pdf_files:
                    try:
                        # Check if we have markdown version available (prefer markdown)
                        has_markdown = pdf_info.get('has_markdown', False)
                        doc_name = pdf_info['filename'].replace('.pdf', '')
                        doc_name = re.sub(r'[^a-zA-Z0-9\s\-\(\)\[\]]', '_', doc_name)
                        doc_name = re.sub(r'\s+', ' ', doc_name).strip()
                        doc_name = doc_name[:50]
                        
                        if has_markdown:
                            # Use markdown content
                            if USE_S3_BUCKET and 'md_s3_key' in pdf_info:
                                try:
                                    response = s3_client.get_object(Bucket=s3_bucket_name, Key=pdf_info['md_s3_key'])
                                    markdown_content = response['Body'].read().decode('utf-8', errors='ignore')
                                    retry_content.append({
                                        "text": f"\n\n--- Content from {pdf_info['filename']} (PDF converted to markdown) ---\n{markdown_content}"
                                    })
                                    continue
                                except Exception as md_error:
                                    logger.warning(f"Failed to read markdown in retry, falling back to PDF bytes: {md_error}")
                            elif 'md_path' in pdf_info and os.path.exists(pdf_info['md_path']):
                                try:
                                    with open(pdf_info['md_path'], 'r', encoding='utf-8', errors='ignore') as md_file:
                                        markdown_content = md_file.read()
                                    retry_content.append({
                                        "text": f"\n\n--- Content from {pdf_info['filename']} (PDF converted to markdown) ---\n{markdown_content}"
                                    })
                                    continue
                                except Exception as md_error:
                                    logger.warning(f"Failed to read markdown file in retry, falling back to PDF bytes: {md_error}")
                        
                        # Fall back to PDF bytes method
                        if USE_S3_BUCKET and 's3_key' in pdf_info:
                            try:
                                response = s3_client.get_object(Bucket=s3_bucket_name, Key=pdf_info['s3_key'])
                                pdf_bytes = response['Body'].read()
                                
                                logger.info(f"Retry: Adding PDF document (downloaded from S3): {doc_name} ({len(pdf_bytes)} bytes)")
                                
                                retry_content.append({
                                    "document": {
                                        "format": "pdf",
                                        "name": doc_name,
                                        "source": {
                                            "bytes": pdf_bytes
                                        }
                                    }
                                })
                            except Exception as download_error:
                                logger.error(f"Retry: Failed to download PDF from S3: {download_error}")
                                continue
                        else:
                            # Local mode
                            pdf_path = pdf_info['path']
                            if os.path.exists(pdf_path):
                                with open(pdf_path, 'rb') as f:
                                    pdf_bytes = f.read()
                                
                                retry_content.append({
                                    "document": {
                                        "format": "pdf",
                                        "name": doc_name,
                                        "source": {
                                            "bytes": pdf_bytes
                                        }
                                    }
                                })
                    except Exception as pdf_error:
                        logger.error(f"Retry: Error processing PDF {pdf_info}: {pdf_error}")
                
                # Add the text prompt
                retry_content.append({
                    "text": enhanced_prompt
                })
                
                # Retry the API call with bytes method
                retry_messages = [{
                    "role": "user",
                    "content": retry_content
                }]
                
                logger.info(f"Retrying request with {len(retry_content)} content items using bytes method")
                
                retry_api_params = get_bedrock_api_params(current_model, retry_messages, inference_config)
                retry_response = bedrock_client.converse(**retry_api_params)
                
                return retry_response['output']['message']['content'][0]['text']
            else:
                # Other API error, re-raise
                raise api_error
        
    except ClientError as e:
        error_message = str(e)
        logger.error(f"Bedrock API error: {e}")
        
        # Check for specific error types that should be shown to the user
        if 'ValidationException' in error_message and 'Input is too long' in error_message:
            return "❌ **Error: Input is too long for the selected model.**\n\nThe uploaded documents and conversation history exceed the model's context limit. Please try:\n\n• Removing some uploaded files\n• Using a model with a larger context window\n• Check 'Use Vector Store' to enable chunked document processing"
        elif 'ValidationException' in error_message:
            return f"❌ **Validation Error:** {error_message}"
        elif 'ThrottlingException' in error_message:
            return "❌ **Rate Limit Exceeded:** Too many requests. Please wait a moment and try again."
        elif 'ServiceUnavailableException' in error_message:
            return "❌ **Service Unavailable:** The Bedrock service is temporarily unavailable. Please try again in a few moments."
        else:
            return f"❌ **Bedrock API Error:** {error_message}"
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return f"❌ **Unexpected Error:** {str(e)}"

def convert_markdown_to_html(text):
    """Convert markdown text to HTML with enhanced formatting"""
    # Configure markdown with useful extensions
    md = markdown.Markdown(extensions=[
        'extra',          # Includes tables, fenced code blocks, etc.
        'codehilite',     # Syntax highlighting for code blocks
        'toc',            # Table of contents
        'nl2br',          # Convert newlines to <br> tags
        'sane_lists'      # Better list handling
    ])
    
    return md.convert(text)

def get_session_upload_location():
    """Get or create session-specific upload location (S3 prefix or local folder)"""
    global temp_session_dir, session_upload_folders
    
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    # Create Flask session directory if it doesn't exist
    if not temp_session_dir:
        temp_session_dir = tempfile.mkdtemp(prefix='bedbot_sessions_')
        os.chmod(temp_session_dir, 0o700)  # Only owner can read/write/execute
        app.config['SESSION_FILE_DIR'] = temp_session_dir
        if DEBUG_MODE:
            logger.info(f"Created temporary session directory: {temp_session_dir}")
    
    if USE_S3_BUCKET:
        # S3 mode - return S3 prefix for this session
        s3_prefix = f"session_{session['session_id']}/"
        session['s3_prefix'] = s3_prefix
        session.modified = True
        if DEBUG_MODE:
            logger.info(f"Using S3 session prefix: {s3_prefix}")
        return s3_prefix
    else:
        # Local mode - create temporary directory
        session_folder = tempfile.mkdtemp(prefix=f'bedbot_session_{session["session_id"]}_')
        os.chmod(session_folder, 0o700)  # Only owner can read/write/execute
        
        # Track this folder for cleanup on shutdown
        session_upload_folders.add(session_folder)
        
        # Store the session folder path in the session for cleanup
        session['session_folder'] = session_folder
        session.modified = True
        
        if DEBUG_MODE:
            logger.info(f"Created session upload folder: {session_folder}")
        return session_folder

@app.route('/')
def index():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        session['chat_history'] = []
        session['document_context'] = ""
        session['uploaded_files'] = []
        session['pdf_files'] = []
        session['pdfs_initialized'] = False  # Track if PDFs have been sent to Bedrock
        
        # Check for cookie preference first, then use default
        cookie_model = request.cookies.get('preferred_model')
        if cookie_model:
            session['selected_model'] = cookie_model
            logger.info(f"New session created with cookie model: {cookie_model}")
        else:
            # Set default model (first in the list)
            default_model = BEDROCK_MODEL.split(',')[0].strip()
            session['selected_model'] = default_model
            logger.info(f"New session created with default model: {default_model}")
            
        # Create session upload location
        get_session_upload_location()
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        use_vector_store = data.get('use_vector_store', False)
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Get document context and PDF files
        context = session.get('document_context', '')
        pdf_files = session.get('pdf_files', [])
        chat_history = session.get('chat_history', [])
        pdfs_initialized = session.get('pdfs_initialized', False)
        
        # Debug logging
        logger.info(f"Chat request - PDF files in session: {len(pdf_files)}")
        logger.info(f"Chat request - Chat history length: {len(chat_history)}")
        logger.info(f"Chat request - PDFs initialized: {pdfs_initialized}")
        logger.info(f"Chat request - User vector store preference: {use_vector_store}")
        
        # Determine if we should use vector store (both system enabled and user preference)
        use_vector_for_request = USE_VECTOR_STORE and use_vector_store
        logger.info(f"Chat request - Using vector store: {use_vector_for_request} (system: {USE_VECTOR_STORE}, user: {use_vector_store})")
        
        # Determine if we need to send PDFs (first time with PDFs or PDFs not yet initialized)
        # When vector store is enabled by user, we don't send PDFs to Bedrock at all
        if use_vector_for_request and pdf_files and not pdfs_initialized:
            # Mark PDFs as initialized immediately since we use vector store instead
            session['pdfs_initialized'] = True
            session.modified = True
            send_pdfs = False
            logger.info(f"Vector store enabled - marking {len(pdf_files)} PDFs as initialized without sending to Bedrock")
        else:
            send_pdfs = pdf_files and not pdfs_initialized
        
        logger.info(f"PDF decision: pdf_files={len(pdf_files) if pdf_files else 0}, pdfs_initialized={pdfs_initialized}, send_pdfs={send_pdfs}, vector_store={USE_VECTOR_STORE}")
        
        # Build conversation history for Bedrock (exclude current session initialization if it exists)
        conversation_history = []
        if chat_history:
            # Filter out any PDF initialization messages (they contain "confirm" language)
            # but only skip the very first message if it's a PDF initialization
            for i, entry in enumerate(chat_history):
                user_msg = entry.get('user', '').lower()
                # Skip only the first message if it's clearly a PDF initialization
                if i == 0 and ('confirm' in user_msg and 'access' in user_msg and ('document' in user_msg or 'uploaded' in user_msg)):
                    logger.info("Skipping PDF initialization message from conversation history")
                    continue
                
                conversation_history.append({
                    'user': entry['user'],
                    'bot': entry['bot']  # Use original markdown, not HTML
                })
        
        logger.info(f"Conversation history entries to send: {len(conversation_history)}")
        logger.info(f"Will send PDFs this request: {send_pdfs}")
        
        # Call Bedrock with conversation history
        bot_response = call_bedrock(
            prompt=user_message,
            context=context,
            pdf_files=pdf_files,
            conversation_history=conversation_history,
            send_pdfs=send_pdfs,
            use_vector_store=use_vector_store
        )
        
        # Check if bot_response contains an error (starts with ❌)
        if bot_response.startswith('❌'):
            # Don't mark PDFs as initialized if there was an error
            # Don't store error messages in chat history
            bot_response_html = convert_markdown_to_html(bot_response)
            return jsonify({
                'response': bot_response_html,
                'timestamp': datetime.now().isoformat(),
                'error': True  # Flag to help frontend handle errors differently
            })
        
        # Mark PDFs as initialized if they were sent
        if send_pdfs:
            session['pdfs_initialized'] = True
            logger.info("Marked PDFs as initialized in session")
        
        # Convert markdown to HTML
        bot_response_html = convert_markdown_to_html(bot_response)
        
        # Store in chat history (store both markdown and HTML)
        if 'chat_history' not in session:
            session['chat_history'] = []
        
        session['chat_history'].append({
            'user': user_message,
            'bot': bot_response,  # Store original markdown
            'bot_html': bot_response_html,  # Store HTML version
            'timestamp': datetime.now().isoformat()
        })
        
        session.modified = True
        
        return jsonify({
            'response': bot_response_html,  # Send HTML to frontend
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# Removed complex session locking

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        import time
        request_start_time = time.time()
        logger.info(f"=== UPLOAD REQUEST START (timestamp: {request_start_time}) ===")
        
        # Ensure vector store is properly configured before processing files
        if not ensure_vector_store_configured():
            logger.warning("⚠️ Vector store configuration issue - files will be uploaded but not indexed")
        
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        
        # Get user's vector store preference and status session ID from the upload request
        use_vector_store_param = request.form.get('use_vector_store', 'false')
        user_wants_vector_store = use_vector_store_param.lower() == 'true'
        should_index_documents = USE_VECTOR_STORE and user_wants_vector_store
        status_session_id = request.form.get('status_session_id')
        
        # Send initial status update
        if status_session_id:
            send_status_update(status_session_id, "Upload", f"Processing {len(files)} file(s) ...")
        
        logger.info(f"Upload request - Raw parameter: '{use_vector_store_param}', Vector store system enabled: {USE_VECTOR_STORE}, user preference: {user_wants_vector_store}, will index: {should_index_documents}")
        
        # Check file count limit
        current_file_count = len(session.get('uploaded_files', []))
        if current_file_count + len(files) > MAX_FILES_PER_SESSION:
            return jsonify({'error': f'Maximum {MAX_FILES_PER_SESSION} files allowed per session'}), 400
        
        uploaded_files = []
        new_text_content = ""
        new_pdf_files = []
        
        # Get or create session-specific upload location
        if USE_S3_BUCKET:
            if 's3_prefix' in session:
                session_location = session['s3_prefix']
            else:
                session_location = get_session_upload_location()
        else:
            if 'session_folder' in session and os.path.exists(session['session_folder']):
                session_location = session['session_folder']
            else:
                session_location = get_session_upload_location()
        
        # Process all files individually (PDF merging removed)
        files_to_process = files
        
        logger.info(f"Processing {len(files_to_process)} files individually")
        
        # Collect PDFs for parallel processing
        pdf_conversion_tasks = []
        pdf_files_info = {}  # Track PDF info by file key
        
        for file in files_to_process:
            if file and file.filename and allowed_file(file.filename):
                logger.info(f"Processing file: {file.filename}")
                # Check file size
                file.seek(0, 2)  # Seek to end
                file_size = file.tell()
                file.seek(0)  # Reset to beginning
                
                if file_size > MAX_FILE_SIZE:
                    logger.info(f'File "{file.filename}" is {file_size / (1024*1024):.1f} MB')
                
                filename = secure_filename(file.filename)
                # Add timestamp to avoid conflicts
                timestamped_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
                
                if USE_S3_BUCKET:
                    # S3 mode - upload to S3
                    s3_key = f"{session_location}{timestamped_filename}"
                    
                    try:
                        # Upload original file
                        s3_client.upload_fileobj(file, s3_bucket_name, s3_key)
                        logger.info(f"Uploaded file to S3: s3://{s3_bucket_name}/{s3_key}")
                        
                        # Handle file content based on type
                        try:
                            content = read_file_content(s3_key, is_s3=True)
                            if content:
                                if isinstance(content, dict) and content.get('type') == 'pdf':
                                    # Handle PDF processing - always use traditional markdown conversion for now
                                    # since direct PDF processing is not yet implemented
                                    pdf_processed_directly = False
                                    
                                    # Try vector store direct processing first if user has enabled it
                                    if should_index_documents:
                                        try:
                                            logger.info(f"📄 Attempting direct PDF processing to vector store: {file.filename}")
                                            vector_manager = get_vector_store_manager()
                                            if vector_manager:
                                                success = vector_manager.add_pdf_document_from_s3(
                                                    session_id=session['session_id'],
                                                    s3_key=s3_key,
                                                    s3_bucket=s3_bucket_name,
                                                    source_filename=file.filename,
                                                    metadata={
                                                        's3_key': s3_key,
                                                        's3_bucket': s3_bucket_name,
                                                        'processing_method': 'unstructured.io_direct'
                                                    }
                                                )
                                                if success:
                                                    logger.info(f"✅ Successfully added PDF {file.filename} to vector store with direct processing")
                                                    pdf_processed_directly = True
                                                else:
                                                    logger.info(f"Direct PDF processing not available, falling back to markdown conversion for {file.filename}")
                                        except Exception as vs_error:
                                            logger.info(f"Direct PDF processing failed, falling back to markdown conversion for {file.filename}: {vs_error}")
                                    
                                    # If direct processing failed or wasn't available, use traditional PDF processing
                                    if not pdf_processed_directly:
                                        # Traditional PDF processing (convert to markdown)
                                        pdf_info = {
                                            's3_key': s3_key,
                                            'filename': file.filename,
                                            'timestamped_filename': timestamped_filename,
                                            'has_markdown': False
                                        }
                                        
                                        # Add to parallel conversion queue if enabled
                                        if PDF_CONVERSION_ENABLED:
                                            file_key = f"{file.filename}_{len(pdf_conversion_tasks)}"  # Unique key
                                            pdf_conversion_tasks.append((file_key, s3_key, s3_bucket_name, file.filename))
                                            pdf_files_info[file_key] = {
                                                'pdf_info': pdf_info,
                                                'session_location': session_location,
                                                'timestamped_filename': timestamped_filename
                                            }
                                            conversion_method = "local" if PDF_LOCAL_CONVERT else "Bedrock"
                                            logger.info(f"Added PDF to parallel conversion queue ({conversion_method}): {file.filename}")
                                        else:
                                            logger.info(f"PDF conversion disabled - PDF will be processed directly by Bedrock: {file.filename}")
                                        
                                        # Always add the PDF file (markdown will be added later if conversion succeeds)
                                        new_pdf_files.append(pdf_info)
                                        logger.info(f"Added PDF to new_pdf_files: {pdf_info['filename']} (will attempt conversion: {PDF_CONVERSION_ENABLED})")
                                        
                                        uploaded_files.append({
                                            'filename': file.filename,
                                            'size': file_size,
                                            'status': 'success',
                                            'type': 'pdf',
                                            'processing': 'traditional_markdown'
                                        })
                                    else:
                                        # Direct processing was successful, just add to uploaded files
                                        uploaded_files.append({
                                            'filename': file.filename,
                                            'size': file_size,
                                            'status': 'success',
                                            'type': 'pdf',
                                            'processing': 'vector_store_direct'
                                        })
                                else:
                                    # Regular text content
                                    new_text_content += f"\n\n--- Content from {file.filename} ---\n{content}"
                                    
                                    # Add to vector store if user has enabled it - SIMPLIFIED APPROACH
                                    if should_index_documents:
                                        try:
                                            logger.info(f"📄 Adding text file to vector store: {file.filename}")
                                            vector_manager = get_vector_store_manager()
                                            if vector_manager:
                                                # Simple approach: just pass the content we already have
                                                success = vector_manager.add_document_from_content(
                                                    session_id=session['session_id'],
                                                    content=content,
                                                    source_filename=file.filename,
                                                    metadata={'s3_key': s3_key}
                                                )
                                                if success:
                                                    logger.info(f"✅ Successfully added {file.filename} text file to vector store")
                                                else:
                                                    logger.error(f"❌ Failed to add {file.filename} to vector store")
                                            else:
                                                logger.error("❌ Vector store manager is None for text file")
                                        except Exception as vs_error:
                                            logger.error(f"❌ Critical error adding {file.filename} to vector store: {vs_error}")
                                            import traceback
                                            logger.error(f"Full traceback: {traceback.format_exc()}")
                                    
                                    uploaded_files.append({
                                        'filename': file.filename,
                                        'size': file_size,
                                        'status': 'success',
                                        'type': 'text'
                                    })
                            else:
                                uploaded_files.append({
                                    'filename': file.filename,
                                    'status': 'error',
                                    'message': 'Could not read file content'
                                })
                        except Exception as e:
                            logger.error(f"Error processing file content for {file.filename}: {e}")
                            import traceback
                            logger.error(f"Full traceback: {traceback.format_exc()}")
                            uploaded_files.append({
                                'filename': file.filename,
                                'status': 'error',
                                'message': f'Error processing file: {str(e)}'
                            })
                    except Exception as e:
                        logger.error(f"Error uploading {file.filename} to S3: {e}")
                        import traceback
                        logger.error(f"Full traceback: {traceback.format_exc()}")
                        uploaded_files.append({
                            'filename': file.filename,
                            'status': 'error',
                            'message': 'Failed to upload to S3'
                        })
                    
                    logger.info(f"Completed processing file: {file.filename}")
                else:
                    # Local mode - save to local filesystem
                    filepath = os.path.join(session_location, timestamped_filename)
                    
                    # Save original file
                    file.save(filepath)
                    
                    # Handle file content based on type
                    content = read_file_content(filepath, is_s3=False)
                    if content:
                        if isinstance(content, dict) and content.get('type') == 'pdf':
                            # Convert PDF to markdown
                            conversion_method = "local" if PDF_LOCAL_CONVERT else "Bedrock"
                            logger.info(f"Converting PDF to markdown ({conversion_method}): {file.filename}")
                            if PDF_LOCAL_CONVERT:
                                markdown_content = convert_pdf_to_markdown_local(filepath, is_s3=False)
                            else:
                                markdown_content = convert_pdf_to_markdown_bedrock(filepath, is_s3=False, model_id=PDF_CONVERT_MODEL)
                            
                            if markdown_content:
                                # Create markdown filename and save locally
                                base_name = os.path.splitext(filepath)[0]
                                md_filepath = f"{base_name}.md"
                                
                                try:
                                    with open(md_filepath, 'w', encoding='utf-8') as md_file:
                                        md_file.write(markdown_content)
                                    logger.info(f"Saved markdown locally: {md_filepath}")
                                    
                                    # Store both PDF and markdown info for Bedrock processing
                                    new_pdf_files.append({
                                        'path': filepath,
                                        'md_path': md_filepath,
                                        'filename': file.filename,
                                        'timestamped_filename': timestamped_filename,
                                        'has_markdown': True
                                    })
                                    
                                    # Also add markdown content to text context for immediate use
                                    new_text_content += f"\n\n--- Content from {file.filename} (converted from PDF) ---\n{markdown_content}"
                                    
                                    # Add to vector store if user has enabled it - SIMPLIFIED APPROACH
                                    if should_index_documents:
                                        try:
                                            vector_manager = get_vector_store_manager()
                                            if vector_manager:
                                                logger.info(f"📄 Adding PDF markdown to vector store: {file.filename}")
                                                logger.info(f"📊 Markdown content size: {len(markdown_content):,} chars, {len(markdown_content.split()):,} words")
                                                
                                                # Log content preview for debugging
                                                content_preview = markdown_content[:500] + "..." if len(markdown_content) > 500 else markdown_content
                                                logger.info(f"📝 Content preview: {content_preview}")
                                                
                                                success = vector_manager.add_document_from_content(
                                                    session_id=session['session_id'],
                                                    content=markdown_content,
                                                    source_filename=file.filename,
                                                    metadata={'file_path': md_filepath, 'type': 'pdf_markdown'}
                                                )
                                                if success:
                                                    logger.info(f"✅ Added {file.filename} markdown to vector store")
                                                else:
                                                    logger.error(f"❌ Failed to add {file.filename} to vector store")
                                        except Exception as vs_error:
                                            logger.error(f"❌ Error adding {file.filename} to vector store: {vs_error}")
                                    
                                except Exception as e:
                                    logger.error(f"Error saving markdown locally: {e}")
                                    # Fall back to PDF-only storage
                                    new_pdf_files.append({
                                        'path': filepath,
                                        'filename': file.filename,
                                        'timestamped_filename': timestamped_filename,
                                        'has_markdown': False
                                    })
                            else:
                                logger.warning(f"Failed to convert PDF to markdown: {file.filename}")
                                # Store PDF info for direct Bedrock processing
                                new_pdf_files.append({
                                    'path': filepath,
                                    'filename': file.filename,
                                    'timestamped_filename': timestamped_filename,
                                    'has_markdown': False
                                })
                            
                            uploaded_files.append({
                                'filename': file.filename,
                                'size': os.path.getsize(filepath),
                                'status': 'success',
                                'type': 'pdf'
                            })
                        else:
                            # Regular text content
                            new_text_content += f"\n\n--- Content from {file.filename} ---\n{content}"
                            
                            # Add to vector store if user has enabled it - SIMPLIFIED APPROACH
                            if should_index_documents:
                                try:
                                    vector_manager = get_vector_store_manager()
                                    if vector_manager:
                                        logger.info(f"📄 Adding text file to vector store: {file.filename}")
                                        success = vector_manager.add_document_from_content(
                                            session_id=session['session_id'],
                                            content=content,
                                            source_filename=file.filename,
                                            metadata={'file_path': filepath}
                                        )
                                        if success:
                                            logger.info(f"✅ Added {file.filename} text file to vector store")
                                        else:
                                            logger.error(f"❌ Failed to add {file.filename} to vector store")
                                except Exception as vs_error:
                                    logger.error(f"❌ Error adding {file.filename} to vector store: {vs_error}")
                            
                            uploaded_files.append({
                                'filename': file.filename,
                                'size': os.path.getsize(filepath),
                                'status': 'success',
                                'type': 'text'
                            })
                    else:
                        uploaded_files.append({
                            'filename': file.filename,
                            'status': 'error',
                            'message': 'Could not read file content'
                        })
        
        logger.info(f"Finished processing files. new_pdf_files count: {len(new_pdf_files)}, uploaded_files count: {len(uploaded_files)}")
        logger.info(f"new_pdf_files: {[f['filename'] for f in new_pdf_files]}")
        logger.info(f"uploaded_files: {[f['filename'] for f in uploaded_files]}")
        
        # Process PDFs in parallel if any were queued
        if pdf_conversion_tasks and PDF_CONVERSION_ENABLED:
            if status_session_id:
                send_status_update(status_session_id, "PDF Conversion", f"Converting {len(pdf_conversion_tasks)} PDF(s) to text ...")
            logger.info(f"Starting parallel PDF conversion for {len(pdf_conversion_tasks)} files")
            conversion_results = convert_pdfs_parallel(pdf_conversion_tasks)
            
            # Update PDF files with conversion results
            for file_key, result in conversion_results.items():
                if file_key in pdf_files_info:
                    pdf_info = pdf_files_info[file_key]['pdf_info']
                    session_location = pdf_files_info[file_key]['session_location']
                    timestamped_filename = pdf_files_info[file_key]['timestamped_filename']
                    
                    if result['markdown_content']:
                        # Create markdown filename
                        base_name = os.path.splitext(timestamped_filename)[0]
                        md_filename = f"{base_name}.md"
                        md_s3_key = f"{session_location}{md_filename}"
                        
                        # Upload markdown to S3
                        try:
                            s3_client.put_object(
                                Bucket=s3_bucket_name,
                                Key=md_s3_key,
                                Body=result['markdown_content'].encode('utf-8'),
                                ContentType='text/markdown'
                            )
                            logger.info(f"Uploaded markdown to S3: s3://{s3_bucket_name}/{md_s3_key}")
                            
                            # Update PDF info with markdown details
                            pdf_info['md_s3_key'] = md_s3_key
                            pdf_info['has_markdown'] = True
                            
                            # Also add markdown content to text context for immediate use
                            new_text_content += f"\n\n--- Content from {result['filename']} (converted from PDF) ---\n{result['markdown_content']}"
                            
                            # Add to vector store if user has enabled it - SIMPLIFIED APPROACH
                            if should_index_documents:
                                if status_session_id:
                                    send_status_update(status_session_id, "Vector Store", f"Creating hierarchical chunks and indexing {result['filename']} ...")
                                try:
                                    vector_manager = get_vector_store_manager()
                                    if vector_manager:
                                        logger.info(f"📄 Adding PDF markdown to vector store: {result['filename']}")
                                        logger.info(f"📊 Markdown content size: {len(result['markdown_content']):,} chars, {len(result['markdown_content'].split()):,} words")
                                        
                                        # Log content preview for debugging
                                        content_preview = result['markdown_content'][:500] + "..." if len(result['markdown_content']) > 500 else result['markdown_content']
                                        logger.info(f"📝 Content preview: {content_preview}")
                                        
                                        # Simple approach: pass the markdown content directly
                                        success = vector_manager.add_document_from_content(
                                            session_id=session['session_id'],
                                            content=result['markdown_content'],
                                            source_filename=result['filename'],
                                            metadata={'s3_key': md_s3_key, 'type': 'pdf_markdown'}
                                        )
                                        if success:
                                            logger.info(f"✅ Successfully added {result['filename']} markdown to vector store")
                                        else:
                                            logger.error(f"❌ Failed to add {result['filename']} to vector store - check logs for details")
                                    else:
                                        logger.error("❌ Vector store manager is None - not initialized properly")
                                except Exception as vs_error:
                                    logger.error(f"❌ Critical error adding {result['filename']} to vector store: {vs_error}")
                                    import traceback
                                    logger.error(f"Full traceback: {traceback.format_exc()}")
                            
                        except Exception as e:
                            logger.error(f"Error uploading markdown to S3 for {result['filename']}: {e}")
                    else:
                        logger.warning(f"Failed to convert PDF to markdown: {result['filename']} - {result.get('error', 'Unknown error')}")
        else:
            logger.info("No PDF conversion tasks to process")
        
        # Initialize session lists if they don't exist
        if 'uploaded_files' not in session:
            session['uploaded_files'] = []
        if 'pdf_files' not in session:
            session['pdf_files'] = []
        
        logger.info(f"Session state before processing: {len(session['uploaded_files'])} uploaded files, {len(session['pdf_files'])} PDF files")
        logger.info(f"Existing files in session: {[f['filename'] for f in session['uploaded_files']]}")
        
        # Make copies of session lists to avoid concurrent modification issues
        current_uploaded_files = list(session['uploaded_files'])
        current_pdf_files = list(session['pdf_files'])
        
        # Add new PDF files to existing list (avoid duplicates)
        pdf_added = False
        for new_pdf in new_pdf_files:
            existing = next((f for f in current_pdf_files if f['filename'] == new_pdf['filename']), None)
            if not existing:
                current_pdf_files.append(new_pdf)
                logger.info(f"Added new PDF to session: {new_pdf['filename']}")
                pdf_added = True
        
        # If new PDFs were added, reset the initialization state so they get sent to Bedrock
        if pdf_added:
            session['pdfs_initialized'] = False
            logger.info("Reset PDF initialization state due to new PDF uploads")
        
        # Add new files to uploaded files list (avoid duplicates)
        logger.info(f"Before adding new files - session has {len(current_uploaded_files)} files")
        for file_info in uploaded_files:
            if file_info['status'] == 'success':
                existing = next((f for f in current_uploaded_files if f['filename'] == file_info['filename']), None)
                if not existing:
                    current_uploaded_files.append({
                        'filename': file_info['filename'],
                        'size': file_info['size'],
                        'upload_time': datetime.now().isoformat(),
                        'type': file_info.get('type', 'text')
                    })
                    logger.info(f"Added new file to session: {file_info['filename']}")
                else:
                    logger.info(f"File already exists in session: {file_info['filename']}")
        
        # Update session with the modified lists - do this atomically
        session['uploaded_files'] = current_uploaded_files
        session['pdf_files'] = current_pdf_files
        
        logger.info(f"After adding new files - session has {len(current_uploaded_files)} files")
        logger.info(f"Session files: {[f['filename'] for f in current_uploaded_files]}")
        
        # Rebuild complete document context from all text files in session
        total_content = ""
        for file_info in current_uploaded_files:
            if file_info.get('type') != 'pdf':
                if USE_S3_BUCKET:
                    # S3 mode - find file by S3 key pattern
                    try:
                        paginator = s3_client.get_paginator('list_objects_v2')
                        for page in paginator.paginate(Bucket=s3_bucket_name, Prefix=session.get('s3_prefix', '')):
                            if 'Contents' in page:
                                for obj in page['Contents']:
                                    if obj['Key'].endswith(f"_{secure_filename(file_info['filename'])}"):
                                        content = read_file_content(obj['Key'], is_s3=True)
                                        if content and not isinstance(content, dict):
                                            total_content += f"\n\n--- Content from {file_info['filename']} ---\n{content}"
                                        break
                    except Exception as e:
                        logger.error(f"Error reading S3 file for context rebuild: {e}")
                else:
                    # Local mode - find file on disk
                    session_folder = session.get('session_folder', '')
                    if os.path.exists(session_folder):
                        for uploaded_file in os.listdir(session_folder):
                            if uploaded_file.endswith(f"_{secure_filename(file_info['filename'])}"):
                                filepath = os.path.join(session_folder, uploaded_file)
                                content = read_file_content(filepath, is_s3=False)
                                if content and not isinstance(content, dict):
                                    total_content += f"\n\n--- Content from {file_info['filename']} ---\n{content}"
                                break
        
        session['document_context'] = f"Context from uploaded documents:{total_content}" if total_content else ""
        session.modified = True
        
        logger.info(f"Session updated - final file count: {len(current_uploaded_files)} uploaded, {len(current_pdf_files)} PDFs")
        
        # Removed redundant log - already logged above
        
        # Calculate total context length including PDFs
        total_context_length = len(total_content)
        pdf_context_note = ""
        if current_pdf_files:
            pdf_count = len(current_pdf_files)
            pdf_context_note = f" + {pdf_count} PDF file(s) processed by Bedrock"
            # Estimate PDF content size for display (rough estimate)
            for pdf_info in current_pdf_files:
                if not USE_S3_BUCKET and 'path' in pdf_info and os.path.exists(pdf_info['path']):
                    pdf_size = os.path.getsize(pdf_info['path'])
                    total_context_length += pdf_size // 4  # Rough estimate: 4 bytes per character
                elif USE_S3_BUCKET and 's3_key' in pdf_info:
                    # For S3, we could get object size but it's optional for display
                    total_context_length += 50000  # Rough estimate for display
        
        context_display = f"{len(total_content)} text chars{pdf_context_note}" if pdf_context_note else f"{total_context_length} chars"
        
        logger.info(f"Returning to frontend - uploaded_files count: {len(current_uploaded_files)}")
        logger.info(f"Files being returned: {[f['filename'] for f in current_uploaded_files]}")
        
        # Calculate total content size sent to LLM (markdown from PDFs + text files)
        total_llm_content_bytes = 0
        
        # Add markdown content from PDFs
        for pdf_info in current_pdf_files:
            if pdf_info.get('has_markdown', False):
                if USE_S3_BUCKET and 'md_s3_key' in pdf_info:
                    # S3 mode - get markdown size from S3
                    try:
                        response = s3_client.head_object(Bucket=s3_bucket_name, Key=pdf_info['md_s3_key'])
                        total_llm_content_bytes += response['ContentLength']
                    except Exception as e:
                        logger.warning(f"Could not get size of markdown file from S3: {e}")
                elif 'md_path' in pdf_info and os.path.exists(pdf_info['md_path']):
                    # Local mode - get markdown size from file
                    try:
                        total_llm_content_bytes += os.path.getsize(pdf_info['md_path'])
                    except Exception as e:
                        logger.warning(f"Could not get size of markdown file: {e}")
        
        # Add text content from regular text files
        for file_info in current_uploaded_files:
            if file_info.get('type') != 'pdf':  # Text files
                if USE_S3_BUCKET:
                    # S3 mode - find file by S3 key pattern and get its size
                    try:
                        paginator = s3_client.get_paginator('list_objects_v2')
                        for page in paginator.paginate(Bucket=s3_bucket_name, Prefix=session.get('s3_prefix', '')):
                            if 'Contents' in page:
                                for obj in page['Contents']:
                                    if obj['Key'].endswith(f"_{secure_filename(file_info['filename'])}"):
                                        total_llm_content_bytes += obj['Size']
                                        break
                    except Exception as e:
                        logger.warning(f"Could not get size of text file from S3: {e}")
                else:
                    # Local mode - find file on disk and get its size
                    session_folder = session.get('session_folder', '')
                    if os.path.exists(session_folder):
                        for uploaded_file in os.listdir(session_folder):
                            if uploaded_file.endswith(f"_{secure_filename(file_info['filename'])}"):
                                filepath = os.path.join(session_folder, uploaded_file)
                                try:
                                    total_llm_content_bytes += os.path.getsize(filepath)
                                except Exception as e:
                                    logger.warning(f"Could not get size of text file: {e}")
                                break
        
        # Convert to MiB and format with 3 decimal places
        total_llm_content_mib = total_llm_content_bytes / (1024 * 1024)
        llm_content_size_display = f"{total_llm_content_mib:.3f} MiB" if total_llm_content_bytes > 0 else "0.000 MiB"
        
        result = {
            'message': f'Successfully processed {len(uploaded_files)} files',
            'files': uploaded_files,
            'context_length': total_context_length,
            'context_display': context_display,
            'uploaded_files': current_uploaded_files,
            'pdf_count': len(current_pdf_files),
            'markdown_size_display': llm_content_size_display  # Keep same key name for frontend compatibility
        }
        
        # Send final status update
        if status_session_id:
            send_status_update(status_session_id, "Complete", 
                             f"Successfully processed {len(uploaded_files)} file(s) - Ready for chat!")
        
        request_end_time = time.time()
        request_duration = request_end_time - request_start_time
        logger.info(f"=== UPLOAD REQUEST END - Duration: {request_duration:.2f}s - Returning result ===")
        return jsonify(result)
        
    except Exception as e:
        import traceback
        logger.error(f"Upload error: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/clear')
def clear_session():
    global session_upload_folders
    
    if USE_S3_BUCKET:
        # S3 mode - delete all objects with session prefix
        if 's3_prefix' in session:
            try:
                paginator = s3_client.get_paginator('list_objects_v2')
                for page in paginator.paginate(Bucket=s3_bucket_name, Prefix=session['s3_prefix']):
                    if 'Contents' in page:
                        objects = [{'Key': obj['Key']} for obj in page['Contents']]
                        s3_client.delete_objects(
                            Bucket=s3_bucket_name,
                            Delete={'Objects': objects}
                        )
                logger.info(f"Cleaned up S3 session files with prefix: {session['s3_prefix']}")
            except Exception as e:
                logger.error(f"Error cleaning up S3 session files: {e}")
    else:
        # Local mode - clean up session-specific upload folder
        if 'session_folder' in session:
            session_folder = session['session_folder']
            if os.path.exists(session_folder):
                shutil.rmtree(session_folder)
                logger.info(f"Cleaned up session folder: {session_folder}")
            # Remove from tracking set
            session_upload_folders.discard(session_folder)
    
    # Clear vector store for this session if enabled
    if USE_VECTOR_STORE and 'session_id' in session:
        try:
            vector_manager = get_vector_store_manager()
            if vector_manager:
                vector_manager.clear_session(session['session_id'])
                logger.info(f"Cleared vector store for session: {session['session_id']}")
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
    
    session.clear()
    return redirect(url_for('index'))

@app.route('/history')
def get_history():
    return jsonify(session.get('chat_history', []))

@app.route('/files')
def get_files():
    return jsonify(session.get('uploaded_files', []))

def process_model_name(model_id):
    """Process model name: remove :xxxk suffix and add us. prefix if needed"""
    # Remove :xxxk pattern from the end (where xxx is any number)
    import re
    processed_id = re.sub(r':\d+k$', '', model_id)
    
    # Add us. prefix if not already present and not already a full ARN
    if not processed_id.startswith('us.') and not processed_id.startswith('arn:'):
        processed_id = f"us.{processed_id}"
    
    return processed_id

@app.route('/all_bedrock_models')
def get_all_bedrock_models():
    """Fetch all available Bedrock foundation models"""
    try:
        if not bedrock_client:
            return jsonify({'error': 'Bedrock client not initialized'}), 500
        
        # Use regular boto3 client for listing models (not bedrock-runtime)
        bedrock_list_client = session_aws.client('bedrock', region_name=profile_region)
        
        response = bedrock_list_client.list_foundation_models()
        
        all_models = []
        for model in response.get('modelSummaries', []):
            model_id = model.get('modelId', '')
            model_name = model.get('modelName', model_id)
            
            # Process model name according to requirements
            processed_id = process_model_name(model_id)
            
            all_models.append({
                'id': processed_id,
                'name': f"{model_name} ({processed_id})",
                'original_id': model_id,
                'provider': model.get('providerName', ''),
                'input_modalities': model.get('inputModalities', []),
                'output_modalities': model.get('outputModalities', [])
            })
        
        # Sort by provider and name for better organization
        all_models.sort(key=lambda x: (x['provider'], x['name']))
        
        logger.info(f"Retrieved {len(all_models)} Bedrock foundation models")
        return jsonify({
            'models': all_models,
            'count': len(all_models)
        })
        
    except Exception as e:
        logger.error(f"Error fetching Bedrock models: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/models')
def get_models():
    """Return available Bedrock models from BEDROCK_MODEL configuration plus all Bedrock models"""
    try:
        # Get configured models
        configured_models = [model.strip() for model in BEDROCK_MODEL.split(',') if model.strip()]
        
        model_list = []
        
        # Add configured models first
        for model in configured_models:
            model_list.append({
                'id': model,
                'name': model,
                'type': 'configured'
            })
        
        # Try to get all Bedrock models for enhanced search
        try:
            if bedrock_client:
                bedrock_list_client = session_aws.client('bedrock', region_name=profile_region)
                response = bedrock_list_client.list_foundation_models()
                
                for model in response.get('modelSummaries', []):
                    model_id = model.get('modelId', '')
                    model_name = model.get('modelName', model_id)
                    
                    # Process model name
                    processed_id = process_model_name(model_id)
                    
                    # Skip if already in configured models
                    if processed_id not in configured_models:
                        model_list.append({
                            'id': processed_id,
                            'name': processed_id,  # Just show the model ID
                            'type': 'available',
                            'provider': model.get('providerName', '')
                        })
                
                logger.info(f"Enhanced model list with {len(model_list)} total models")
        except Exception as bedrock_error:
            logger.warning(f"Could not fetch additional Bedrock models: {bedrock_error}")
        
        # Get current model from session or cookie preference
        current_model = session.get('selected_model')
        if not current_model:
            # Check cookie preference
            cookie_model = request.cookies.get('preferred_model')
            if cookie_model:
                current_model = cookie_model
                session['selected_model'] = cookie_model
                session.modified = True
                logger.info(f"Loaded model from cookie: {cookie_model}")
            else:
                current_model = configured_models[0] if configured_models else None
        
        return jsonify({
            'models': model_list,
            'current': current_model,
            'allow_custom': True,
            'vector_store_enabled': USE_VECTOR_STORE
        })
        
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return jsonify({'models': [], 'current': None, 'allow_custom': True})

@app.route('/test_models')
def test_models():
    """Test which models are actually available in AWS Bedrock"""
    try:
        models = [model.strip() for model in BEDROCK_MODEL.split(',') if model.strip()]
        test_results = []
        
        for model in models:
            try:
                # Test each model with a simple request
                api_params = get_bedrock_api_params(
                    model,
                    [{"role": "user", "content": [{"text": "test"}]}],
                    {"maxTokens": 10, "temperature": 0.1}
                )
                test_response = bedrock_client.converse(**api_params)
                test_results.append({
                    'model': model,
                    'status': 'available',
                    'error': None
                })
                logger.info(f"Model test PASSED: {model}")
            except Exception as e:
                test_results.append({
                    'model': model,
                    'status': 'unavailable',
                    'error': str(e)
                })
                logger.error(f"Model test FAILED: {model} - {e}")
        
        return jsonify({
            'region': profile_region,
            'tests': test_results,
            'summary': {
                'total': len(models),
                'available': len([r for r in test_results if r['status'] == 'available']),
                'unavailable': len([r for r in test_results if r['status'] == 'unavailable'])
            }
        })
        
    except Exception as e:
        logger.error(f"Error testing models: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/change_model', methods=['POST'])
def change_model():
    """Change the active Bedrock model for the session"""
    try:
        data = request.get_json()
        new_model = data.get('model', '').strip()
        
        if not new_model:
            return jsonify({'error': 'No model specified'}), 400
        
        # For custom models, we'll validate by attempting a test call later
        # Get predefined models for validation
        available_models = [model.strip() for model in BEDROCK_MODEL.split(',') if model.strip()]
        is_custom_model = new_model not in available_models
        
        # Store the selected model in the session
        session['selected_model'] = new_model
        
        # Reset only the Bedrock conversation state, keep chat history visible
        session['pdfs_initialized'] = False  # Reset PDF initialization so they get re-sent to new model
        session.modified = True
        
        logger.info(f"Model changed to: {new_model} (custom: {is_custom_model}), PDFs will be re-initialized with new model")
        
        # Create response with cookie
        response = jsonify({
            'success': True, 
            'model': new_model, 
            'model_changed': True,
            'is_custom': is_custom_model
        })
        
        # Set cookie to remember user preference (expires in 30 days)
        response.set_cookie(
            'preferred_model', 
            new_model, 
            max_age=30*24*60*60,  # 30 days
            secure=False,  # Set to True in production with HTTPS
            httponly=False,  # Allow JavaScript access for frontend
            samesite='Lax'
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error changing model: {e}")
        return jsonify({'error': 'Failed to change model'}), 500

@app.route('/remove_file', methods=['POST'])
def remove_file():
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        # Remove from uploaded files list
        if 'uploaded_files' in session:
            session['uploaded_files'] = [f for f in session['uploaded_files'] if f['filename'] != filename]
        
        # Remove from PDF files list if it's a PDF and also get markdown info for cleanup
        pdf_to_remove = None
        if 'pdf_files' in session:
            pdf_to_remove = next((f for f in session['pdf_files'] if f['filename'] == filename), None)
            session['pdf_files'] = [f for f in session['pdf_files'] if f['filename'] != filename]
        
        # Remove from vector store if enabled
        if USE_VECTOR_STORE and 'session_id' in session:
            try:
                vector_manager = get_vector_store_manager()
                if vector_manager:
                    success = vector_manager.remove_document_from_session(session['session_id'], filename)
                    if success:
                        logger.info(f"Removed {filename} from vector store")
                    else:
                        logger.info(f"Document {filename} was not found in vector store (may not have been added)")
            except Exception as e:
                logger.error(f"Error removing {filename} from vector store: {e}")
            
            # If we removed a PDF and there are still PDFs remaining, reset initialization
            # If no PDFs remain, we can keep the initialization state as is (no PDFs to initialize)
            if pdf_to_remove and session['pdf_files']:
                session['pdfs_initialized'] = False
                logger.info("Reset PDF initialization state due to PDF file removal (PDFs still remain)")
            elif pdf_to_remove and not session['pdf_files']:
                # All PDFs removed - reset state for cleanliness
                session['pdfs_initialized'] = False
                logger.info("Reset PDF initialization state - no PDFs remain")
        
        # Remove the actual file from storage (including markdown files)
        if USE_S3_BUCKET:
            # S3 mode - find and delete file from S3
            if 's3_prefix' not in session:
                return jsonify({'error': 'Session S3 prefix not found'}), 400
            
            try:
                # Find and delete both PDF and markdown files in S3
                paginator = s3_client.get_paginator('list_objects_v2')
                for page in paginator.paginate(Bucket=s3_bucket_name, Prefix=session['s3_prefix']):
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            obj_key = obj['Key']
                            secured_filename = secure_filename(filename)
                            
                            # Delete PDF file
                            if obj_key.endswith(f"_{secured_filename}"):
                                s3_client.delete_object(Bucket=s3_bucket_name, Key=obj_key)
                                logger.info(f"Removed file from S3: s3://{s3_bucket_name}/{obj_key}")
                            
                            # Delete corresponding markdown file if it exists
                            elif filename.lower().endswith('.pdf'):
                                base_filename = os.path.splitext(secured_filename)[0]
                                if obj_key.endswith(f"_{base_filename}.md"):
                                    s3_client.delete_object(Bucket=s3_bucket_name, Key=obj_key)
                                    logger.info(f"Removed markdown file from S3: s3://{s3_bucket_name}/{obj_key}")
                                    
                # Also delete specific markdown files if we have the S3 key
                if pdf_to_remove and pdf_to_remove.get('md_s3_key'):
                    try:
                        s3_client.delete_object(Bucket=s3_bucket_name, Key=pdf_to_remove['md_s3_key'])
                        logger.info(f"Removed specific markdown file from S3: s3://{s3_bucket_name}/{pdf_to_remove['md_s3_key']}")
                    except Exception as e:
                        logger.warning(f"Could not remove specific markdown file from S3: {e}")
                        
            except Exception as e:
                logger.error(f"Error removing file from S3: {e}")
        else:
            # Local mode - remove file from disk (including markdown)
            if 'session_folder' not in session or not os.path.exists(session['session_folder']):
                return jsonify({'error': 'Session folder not found'}), 400
            
            session_folder = session['session_folder']
            secured_filename = secure_filename(filename)
            
            for uploaded_file in os.listdir(session_folder):
                file_path = os.path.join(session_folder, uploaded_file)
                
                # Remove PDF file
                if uploaded_file.endswith(f"_{secured_filename}"):
                    try:
                        os.remove(file_path)
                        logger.info(f"Removed file from disk: {file_path}")
                    except Exception as e:
                        logger.error(f"Error removing file {file_path}: {e}")
                
                # Remove corresponding markdown file if it exists
                elif filename.lower().endswith('.pdf'):
                    base_filename = os.path.splitext(secured_filename)[0]
                    if uploaded_file.endswith(f"_{base_filename}.md"):
                        try:
                            os.remove(file_path)
                            logger.info(f"Removed markdown file from disk: {file_path}")
                        except Exception as e:
                            logger.error(f"Error removing markdown file {file_path}: {e}")
            
            # Also remove specific markdown file if we have the path
            if pdf_to_remove and pdf_to_remove.get('md_path') and os.path.exists(pdf_to_remove['md_path']):
                try:
                    os.remove(pdf_to_remove['md_path'])
                    logger.info(f"Removed specific markdown file from disk: {pdf_to_remove['md_path']}")
                except Exception as e:
                    logger.warning(f"Could not remove specific markdown file from disk: {e}")
        
        # Rebuild document context without the removed file
        remaining_files = session.get('uploaded_files', [])
        
        if remaining_files:
            # Re-read remaining text files to rebuild context
            total_content = ""
            for file_info in remaining_files:
                if file_info.get('type') != 'pdf':  # Only process text files
                    if USE_S3_BUCKET:
                        # S3 mode - find file by S3 key pattern
                        try:
                            paginator = s3_client.get_paginator('list_objects_v2')
                            for page in paginator.paginate(Bucket=s3_bucket_name, Prefix=session.get('s3_prefix', '')):
                                if 'Contents' in page:
                                    for obj in page['Contents']:
                                        if obj['Key'].endswith(f"_{secure_filename(file_info['filename'])}"):
                                            content = read_file_content(obj['Key'], is_s3=True)
                                            if content and not isinstance(content, dict):
                                                total_content += f"\n\n--- Content from {file_info['filename']} ---\n{content}"
                                            break
                        except Exception as e:
                            logger.error(f"Error reading S3 file for context rebuild: {e}")
                    else:
                        # Local mode - find file on disk
                        session_folder = session.get('session_folder', '')
                        if os.path.exists(session_folder):
                            for uploaded_file in os.listdir(session_folder):
                                if uploaded_file.endswith(f"_{secure_filename(file_info['filename'])}"):
                                    filepath = os.path.join(session_folder, uploaded_file)
                                    content = read_file_content(filepath, is_s3=False)
                                    if content and not isinstance(content, dict):
                                        total_content += f"\n\n--- Content from {file_info['filename']} ---\n{content}"
                                    break
            
            session['document_context'] = f"Context from uploaded documents:{total_content}" if total_content else ""
        else:
            session['document_context'] = ""
        
        session.modified = True
        
        return jsonify({
            'message': f'File {filename} removed successfully',
            'uploaded_files': session.get('uploaded_files', [])
        })
        
    except Exception as e:
        logger.error(f"Remove file error: {e}")
        return jsonify({'error': 'Failed to remove file'}), 500

@app.route('/vector_stats')
def get_vector_stats():
    """Get vector store statistics for debugging"""
    try:
        if not USE_VECTOR_STORE:
            return jsonify({'error': 'Vector store not enabled', 'enabled': False})
        
        if 'session_id' not in session:
            return jsonify({'error': 'No session found', 'enabled': True})
        
        vector_manager = get_vector_store_manager()
        if not vector_manager:
            return jsonify({'error': 'Vector store manager not available', 'enabled': True})
        
        stats = vector_manager.get_session_stats(session['session_id'])
        
        # Check if we got an error from get_session_stats
        if 'error' in stats:
            # Vector store session doesn't exist - return detailed error info
            available_sessions = list(vector_manager.sessions.keys())
            return jsonify({
                'error': stats['error'],
                'enabled': True,
                'debug_info': {
                    'current_session_id': session['session_id'],
                    'available_sessions': available_sessions,
                    'total_sessions': len(vector_manager.sessions),
                    'session_files_count': len(session.get('uploaded_files', [])) + len(session.get('pdf_files', [])),
                    'issue': 'Vector store session was never created or was lost'
                }
            })
        
        stats['enabled'] = True
        
        # Add session file information for comparison
        session_files = session.get('uploaded_files', [])
        session_pdfs = session.get('pdf_files', [])
        
        stats['session_files'] = {
            'uploaded_files': len(session_files),
            'pdf_files': len(session_pdfs),
            'file_list': [f['filename'] for f in session_files],
            'pdf_list': [f['filename'] for f in session_pdfs]
        }
        
        # Check for discrepancies
        all_session_files = set(f['filename'] for f in session_files + session_pdfs)
        vector_files = set(doc['source'] for doc in stats.get('documents', []))
        
        missing_from_vector = all_session_files - vector_files
        extra_in_vector = vector_files - all_session_files
        
        if missing_from_vector:
            stats['warnings'] = stats.get('warnings', [])
            stats['warnings'].append(f"Files uploaded but not in vector store: {list(missing_from_vector)}")
        
        if extra_in_vector:
            stats['warnings'] = stats.get('warnings', [])
            stats['warnings'].append(f"Files in vector store but not in session: {list(extra_in_vector)}")
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting vector stats: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e), 'enabled': USE_VECTOR_STORE})

@app.route('/debug_document/<filename>')
def debug_document(filename):
    """Debug endpoint to examine actual content in vector store"""
    try:
        if not USE_VECTOR_STORE:
            return jsonify({'error': 'Vector store not enabled'})
        
        if 'session_id' not in session:
            return jsonify({'error': 'No session found'})
        
        vector_manager = get_vector_store_manager()
        if not vector_manager:
            return jsonify({'error': 'Vector store manager not available'})
        
        # Get sample chunks from the specified document
        sample_chunks = vector_manager.get_document_sample(
            session_id=session['session_id'], 
            source=filename, 
            max_chunks=10
        )
        
        return jsonify({
            'filename': filename,
            'sample_chunks': sample_chunks,
            'total_samples': len(sample_chunks)
        })
        
    except Exception as e:
        logger.error(f"Error debugging document {filename}: {e}")
        return jsonify({'error': str(e)})

@app.route('/debug_vector_raw')
def debug_vector_raw():
    """Raw debug endpoint to examine vector store internal state"""
    try:
        if not USE_VECTOR_STORE:
            return jsonify({'error': 'Vector store not enabled'})
        
        if 'session_id' not in session:
            return jsonify({'error': 'No session found'})
        
        vector_manager = get_vector_store_manager()
        if not vector_manager:
            return jsonify({'error': 'Vector store manager not available'})
        
        # Get the actual vector store instance
        session_id = session['session_id']
        available_sessions = list(vector_manager.sessions.keys())
        
        debug_session_info = {
            'current_session_id': session_id,
            'available_sessions': available_sessions,
            'session_exists': session_id in vector_manager.sessions,
            'total_sessions': len(vector_manager.sessions)
        }
        
        if session_id not in vector_manager.sessions:
            return jsonify({
                'error': 'Session vector store not found',
                'debug_info': debug_session_info
            })
        
        store = vector_manager.sessions[session['session_id']]
        
        # Get raw debug info
        debug_info = {
            'documents_count': len(store.documents),
            'metadata_count': len(store.doc_metadata),
            'index_total': store.index.ntotal,
            'all_sources': list(set(chunk.source for chunk in store.documents)),
            'metadata_sources': [meta['source'] for meta in store.doc_metadata.values()],
            'first_5_chunks': []
        }
        
        # Get first 5 chunks regardless of source
        for i, chunk in enumerate(store.documents[:5]):
            debug_info['first_5_chunks'].append({
                'index': i,
                'source': chunk.source,
                'chunk_id': chunk.chunk_id,
                'text_length': len(chunk.text),
                'text_preview': chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
            })
        
        return jsonify(debug_info)
        
    except Exception as e:
        logger.error(f"Error in raw vector debug: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)})

@app.route('/debug_vector_manager')
def debug_vector_manager():
    """Debug the vector store manager initialization"""
    try:
        debug_info = {
            'USE_VECTOR_STORE': USE_VECTOR_STORE,
            'VECTOR_STORE_AVAILABLE': VECTOR_STORE_AVAILABLE,
            'VECTOR_STORE_MODULE_AVAILABLE': VECTOR_STORE_MODULE_AVAILABLE
        }
        
        vector_manager = get_vector_store_manager()
        debug_info['vector_manager_exists'] = vector_manager is not None
        
        if vector_manager:
            debug_info['total_sessions'] = len(vector_manager.sessions)
            debug_info['sessions_list'] = list(vector_manager.sessions.keys())
        
        # Check global variables
        global s3_bucket_name, s3_client
        debug_info['global_s3_bucket_name'] = s3_bucket_name
        debug_info['global_s3_client_exists'] = s3_client is not None
        
        return jsonify(debug_info)
        
    except Exception as e:
        logger.error(f"Error debugging vector manager: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)})

@app.route('/search_document/<filename>/<query>')
def search_document(filename, query):
    """Search for specific content within a document"""
    try:
        if not USE_VECTOR_STORE:
            return jsonify({'error': 'Vector store not enabled'})
        
        if 'session_id' not in session:
            return jsonify({'error': 'No session found'})
        
        vector_manager = get_vector_store_manager()
        if not vector_manager:
            return jsonify({'error': 'Vector store manager not available'})
        
        # Search for content in the specific document
        results = vector_manager.search_session(session['session_id'], f"{query} source:{filename}", top_k=10)
        
        # Filter results to only include the specific document
        filtered_results = []
        for chunk, score in results:
            if chunk.source == filename:
                filtered_results.append({
                    'chunk_id': chunk.chunk_id,
                    'text': chunk.text,
                    'score': score,
                    'source': chunk.source
                })
        
        return jsonify({
            'filename': filename,
            'query': query,
            'results': filtered_results,
            'total_results': len(filtered_results)
        })
        
    except Exception as e:
        logger.error(f"Error searching document {filename}: {e}")
        return jsonify({'error': str(e)})

@app.route('/debug_content_flow/<filename>')
def debug_content_flow(filename):
    """Debug the content flow from PDF to vector store"""
    try:
        if not USE_VECTOR_STORE:
            return jsonify({'error': 'Vector store not enabled'})
        
        if 'session_id' not in session:
            return jsonify({'error': 'No session found'})
        
        # Get all chunks for this document from vector store
        vector_manager = get_vector_store_manager()
        if not vector_manager:
            return jsonify({'error': 'Vector store manager not available'})
        
        if session['session_id'] not in vector_manager.sessions:
            return jsonify({'error': 'Session not found in vector store'})
        
        store = vector_manager.sessions[session['session_id']]
        
        # Get all chunks for this document
        document_chunks = []
        total_chars = 0
        for chunk in store.documents:
            if chunk.source == filename:
                document_chunks.append({
                    'chunk_id': chunk.chunk_id,
                    'text_length': len(chunk.text),
                    'text_preview': chunk.text[:300] + "..." if len(chunk.text) > 300 else chunk.text,
                    'metadata': chunk.metadata
                })
                total_chars += len(chunk.text)
        
        # Get document metadata
        doc_metadata = None
        for doc_hash, meta in store.doc_metadata.items():
            if meta['source'] == filename and not meta.get('removed', False):
                doc_metadata = meta
                break
        
        return jsonify({
            'filename': filename,
            'total_chunks': len(document_chunks),
            'total_characters': total_chars,
            'chunks': document_chunks[:10],  # First 10 chunks
            'document_metadata': doc_metadata,
            'session_id': session['session_id']
        })
        
    except Exception as e:
        logger.error(f"Error debugging content flow for {filename}: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)})

@app.route('/extract_structured_data', methods=['POST'])
def extract_structured_data():
    """Extract structured data (URLs, emails, etc.) from documents using enhanced Parent Document Retriever"""
    try:
        if not USE_VECTOR_STORE or not VECTOR_STORE_AVAILABLE:
            return jsonify({'error': 'Vector store not available'})
        
        # Get session ID
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'error': 'No session found'})
        
        # Get query from request
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query required'})
        
        query = data['query']
        top_k = data.get('top_k', 30)  # Higher default for extraction
        
        vector_manager = get_vector_store_manager()
        if not vector_manager:
            return jsonify({'error': 'Vector store manager not available'})
        
        # Extract structured data
        extracted_data = vector_manager.extract_structured_data_from_session(
            session_id, query, top_k
        )
        
        # Add metadata
        result = {
            'query': query,
            'extraction_results': extracted_data,
            'total_items': sum(len(v) for v in extracted_data.values()),
            'strategy': 'Parent Document Retriever with Regex Post-processing',
            'session_id': session_id
        }
        
        logger.info(f"✅ Structured data extraction complete: {result['total_items']} items found")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Error in structured data extraction: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)})

# Map-Reduce endpoints removed - using enhanced Child-Parent-Grandparent vector store approach

@app.route('/upload_status/<status_session_id>')
def upload_status_stream(status_session_id):
    """Server-Sent Events endpoint for upload status updates"""
    
    def event_stream():
        # Create queue for this session
        with upload_status_lock:
            upload_status_queues[status_session_id] = queue.Queue(maxsize=10)
        
        try:
            while True:
                try:
                    # Wait for status update with timeout
                    status_data = upload_status_queues[status_session_id].get(timeout=30)
                    yield f"data: {json.dumps(status_data)}\n\n"
                except queue.Empty:
                    # Send heartbeat to keep connection alive
                    yield f"data: {json.dumps({'heartbeat': True})}\n\n"
                except Exception as e:
                    logger.error(f"Error in status stream: {e}")
                    break
        finally:
            # Clean up queue when client disconnects
            with upload_status_lock:
                if status_session_id in upload_status_queues:
                    del upload_status_queues[status_session_id]
    
    return Response(
        event_stream(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*'
        }
    )

if __name__ == '__main__':
    # Only register signal handlers and cleanup in the actual app process
    # In debug mode, this prevents the reloader process from interfering
    if os.environ.get('WERKZEUG_RUN_MAIN') is not None or not args.debug:
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Register cleanup function for normal exit
        atexit.register(cleanup_resources)
        logger.info(f"Registered signal handlers and cleanup (debug={args.debug}, WERKZEUG_RUN_MAIN={os.environ.get('WERKZEUG_RUN_MAIN')})")

    # Initialize AWS clients
    initialize_aws_clients()

    # Handle S3 bucket creation/loading based on Flask startup mode
    if USE_S3_BUCKET and not os.environ.get('WERKZEUG_RUN_MAIN'):
        # Initial startup - create new bucket
        bucket_created = create_s3_bucket()
        if not bucket_created:
            logger.info("Switched to local filesystem mode (--no-bucket equivalent)")
        else:
            logger.info(f"S3 bucket created successfully: {s3_bucket_name}")
    elif USE_S3_BUCKET and os.environ.get('WERKZEUG_RUN_MAIN'):
        # Flask debug restart - load existing bucket
        handle_flask_restart_bucket()
        logger.info(f"S3 bucket loaded from restart: {s3_bucket_name}")

    # Initialize vector store manager independently
    if USE_VECTOR_STORE:
        logger.info("Initializing vector store")
    initialize_vector_store_if_enabled()

    try:
        logger.info("Starting BedBot application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        # This shouldn't be reached due to signal handler, but kept as fallback
        logger.info("Application interrupted")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        # Signal handler should handle cleanup, but ensure it runs
        if not cleanup_performed:
            cleanup_resources()

