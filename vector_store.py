"""
FAISS Vector Store for BedBot - Document Embeddings and Retrieval
"""

import os
import json
import logging
import tempfile
import pickle
import hashlib
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

# Set up logger
logger = logging.getLogger(__name__)

try:
    # Configure FAISS logging to only show in debug mode
    import os
    debug_mode = os.getenv('DEBUG', '0').lower() in ('1', 'true', 'yes', 'on')
    
    if not debug_mode:
        # Suppress FAISS info/warning messages unless in debug mode
        faiss_logger = logging.getLogger('faiss')
        faiss_logger.setLevel(logging.ERROR)
    
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
    VECTOR_STORE_AVAILABLE = True
    
    if debug_mode:
        logger.info("FAISS and SentenceTransformers loaded successfully")
except ImportError as e:
    FAISS_AVAILABLE = False
    VECTOR_STORE_AVAILABLE = False
    logger.warning(f"FAISS dependencies not available: {e}")
    logger.warning("To enable vector store: pip install faiss-cpu sentence-transformers")

class DocumentChunk:
    """Represents a chunk of text from a document with hierarchical context"""
    def __init__(self, text: str, source: str, chunk_id: str, metadata: Dict = None, 
                 parent_text: str = None, grandparent_text: str = None):
        self.text = text
        self.source = source  # filename
        self.chunk_id = chunk_id
        self.metadata = metadata or {}
        self.parent_text = parent_text  # Medium parent chunk for context
        self.grandparent_text = grandparent_text  # Large grandparent chunk for comprehensive context
        self.timestamp = datetime.now().isoformat()

class FAISSVectorStore:
    """FAISS-based vector store for document embeddings and retrieval"""
    
    def __init__(self, session_id: str, embedding_model: str = "all-MiniLM-L6-v2", 
                 chunk_size: int = 2048, chunk_overlap: int = 200,
                 child_chunk_size: int = 200, parent_chunk_size: int = 800, grandparent_chunk_size: int = 3200):
        """
        Initialize FAISS vector store
        
        Args:
            session_id: Unique session identifier
            embedding_model: SentenceTransformer model name
            chunk_size: Maximum characters per chunk
            chunk_overlap: Character overlap between chunks
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS dependencies not available. Install with: pip install faiss-cpu sentence-transformers")
        
        self.session_id = session_id
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Hierarchical chunking strategy: Child-Parent-Grandparent
        self.child_chunk_size = child_chunk_size  # Small chunks (200 chars) for precise search
        self.parent_chunk_size = parent_chunk_size  # Medium chunks (800 chars) for local context  
        self.grandparent_chunk_size = grandparent_chunk_size  # Large chunks (3200 chars) for comprehensive context
        self.parent_store = {}  # Store parent chunks by child chunk ID
        self.grandparent_store = {}  # Store grandparent chunks by child chunk ID
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model: {embedding_model} (dim: {self.embedding_dim})")
        except Exception as e:
            logger.error(f"Failed to load embedding model {embedding_model}: {e}")
            raise
        
        # Initialize FAISS index - use IndexFlatIP for perfect accuracy
        # IndexFlatIP is fast for up to ~1M vectors and guarantees 100% accuracy
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine similarity)
        self.documents: List[DocumentChunk] = []
        self.doc_metadata: Dict[str, Any] = {}
        
        # State tracking
        self.is_dirty = False  # Track if index needs saving
        
        logger.info(f"Initialized FAISS vector store for session: {session_id}")
    
    # IVF upgrade method removed - IndexFlatIP provides perfect accuracy for our scale
    
    def _create_grandparent_chunks(self, text: str, source: str) -> List[str]:
        """Create largest grandparent chunks for comprehensive context"""
        grandparent_chunks = []
        grandparent_size = self.grandparent_chunk_size
        overlap = self.chunk_overlap
        
        # Split into very large chunks first
        for i in range(0, len(text), grandparent_size - overlap):
            chunk = text[i:i + grandparent_size]
            if chunk.strip():
                grandparent_chunks.append(chunk.strip())
        
        logger.info(f"Created {len(grandparent_chunks)} grandparent chunks ({grandparent_size} chars each) from {source}")
        return grandparent_chunks
    
    def _create_parent_chunks_from_grandparent(self, grandparent_text: str, grandparent_idx: int) -> List[str]:
        """Create medium parent chunks from grandparent chunk"""
        parent_chunks = []
        parent_size = self.parent_chunk_size
        overlap = max(50, self.parent_chunk_size // 10)
        
        # Split grandparent into medium chunks
        for i in range(0, len(grandparent_text), parent_size - overlap):
            chunk = grandparent_text[i:i + parent_size]
            if chunk.strip():
                parent_chunks.append(chunk.strip())
        
        return parent_chunks
    
    def _create_child_chunks_from_parent(self, parent_text: str, parent_idx: int, grandparent_idx: int, 
                                       source: str, grandparent_text: str) -> List[DocumentChunk]:
        """Create small child chunks from parent chunk for precise retrieval"""
        child_chunks = []
        child_size = self.child_chunk_size
        overlap = max(20, self.child_chunk_size // 10)  # Small overlap for precision
        
        # Split parent into very small, precise chunks
        for i in range(0, len(parent_text), child_size - overlap):
            child_text = parent_text[i:i + child_size]
            if child_text.strip():
                chunk_id = f"{source}_gp{grandparent_idx}_p{parent_idx}_c{len(child_chunks)}"
                child_chunk = DocumentChunk(
                    text=child_text.strip(),
                    source=source,
                    chunk_id=chunk_id,
                    parent_text=parent_text,  # Store parent for medium context
                    grandparent_text=grandparent_text,  # Store grandparent for comprehensive context
                    metadata={
                        'grandparent_index': grandparent_idx,
                        'parent_index': parent_idx,
                        'child_index': len(child_chunks),
                        'char_count': len(child_text),
                        'hierarchy_optimized': True
                    }
                )
                child_chunks.append(child_chunk)
                
                # Store hierarchical mappings
                self.parent_store[chunk_id] = parent_text
                self.grandparent_store[chunk_id] = grandparent_text
        
        return child_chunks
    
    def _chunk_text(self, text: str, source: str) -> List[DocumentChunk]:
        """Hierarchical chunking: Child-Parent-Grandparent strategy for multi-level context"""
        logger.info(f"Using Hierarchical Child-Parent-Grandparent chunking for {source}")
        logger.info(f"Child: {self.child_chunk_size} chars, Parent: {self.parent_chunk_size} chars, Grandparent: {self.grandparent_chunk_size} chars")
        
        # Step 1: Create largest grandparent chunks for comprehensive context
        grandparent_chunks = self._create_grandparent_chunks(text, source)
        
        # Step 2: Create medium parent chunks from each grandparent
        # Step 3: Create small child chunks from each parent for precise search
        all_child_chunks = []
        total_parent_chunks = 0
        
        for grandparent_idx, grandparent_text in enumerate(grandparent_chunks):
            # Create parent chunks from this grandparent
            parent_chunks = self._create_parent_chunks_from_grandparent(grandparent_text, grandparent_idx)
            total_parent_chunks += len(parent_chunks)
            
            # Create child chunks from each parent
            for parent_idx, parent_text in enumerate(parent_chunks):
                child_chunks = self._create_child_chunks_from_parent(
                    parent_text, parent_idx, grandparent_idx, source, grandparent_text
                )
                all_child_chunks.extend(child_chunks)
        
        logger.info(f"Hierarchical chunking complete for {source}:")
        logger.info(f"  - {len(grandparent_chunks)} grandparent chunks ({self.grandparent_chunk_size} chars each) - for comprehensive context")
        logger.info(f"  - {total_parent_chunks} parent chunks ({self.parent_chunk_size} chars each) - for local context")
        logger.info(f"  - {len(all_child_chunks)} child chunks ({self.child_chunk_size} chars each) - for precise search")
        logger.info(f"  - FAISS IndexFlatIP: Perfect accuracy guaranteed, no training required")
        
        # Preview for debugging
        if all_child_chunks:
            first_child = all_child_chunks[0]
            logger.info(f"🔍 First child preview ({len(first_child.text)} chars): {first_child.text[:100]}...")
            logger.info(f"🔍 Parent context ({len(first_child.parent_text)} chars): Available")
            logger.info(f"🔍 Grandparent context ({len(first_child.grandparent_text)} chars): Available")
        
        return all_child_chunks
    
    def add_document(self, text: str, source: str, metadata: Dict = None) -> int:
        """
        Add a document to the vector store
        
        Args:
            text: Document text content
            source: Document source/filename
            metadata: Optional metadata dictionary
            
        Returns:
            Number of chunks added
        """
        try:
            # Validate input
            if not text or not text.strip():
                logger.error(f"❌ Document {source} is empty or contains no text")
                return 0
            
            if len(text) < 50:  # Very short documents might indicate processing issues
                logger.warning(f"⚠️ Document {source} is very short ({len(text)} chars) - possible processing issue")
            
            # Generate document hash for deduplication
            doc_hash = hashlib.md5(f"{source}_{text}".encode()).hexdigest()
            
            # Check if document already exists
            if doc_hash in self.doc_metadata:
                logger.info(f"Document {source} already exists in vector store")
                return 0
            
            # Log document stats
            logger.info(f"📄 Processing document: {source}")
            logger.info(f"   - Size: {len(text):,} characters ({len(text.split()):,} words)")
            logger.info(f"   - Expected chunks: ~{len(text) // self.chunk_size}")
            
            # Chunk the document
            chunks = self._chunk_text(text, source)
            
            if not chunks:
                logger.error(f"❌ No chunks generated for document: {source}")
                return 0
            
            # Validate chunk generation
            total_chunk_chars = sum(len(chunk.text) for chunk in chunks)
            coverage_ratio = total_chunk_chars / len(text)
            
            if coverage_ratio < 0.8:  # Less than 80% coverage suggests issues
                logger.warning(f"⚠️ Document {source} chunking coverage is low: {coverage_ratio:.1%}")
                logger.warning(f"   - Original: {len(text):,} chars")
                logger.warning(f"   - Chunks total: {total_chunk_chars:,} chars")
            
            logger.info(f"✅ Chunked document {source}: {len(chunks)} chunks, {coverage_ratio:.1%} coverage")
            
            # Generate embeddings for all chunks with memory safety
            chunk_texts = [chunk.text for chunk in chunks]
            
            logger.info(f"🔄 Generating embeddings for {len(chunk_texts)} chunks...")
            try:
                # Process in smaller batches to prevent memory issues
                batch_size = 50  # Process 50 chunks at a time
                all_embeddings = []
                
                for i in range(0, len(chunk_texts), batch_size):
                    batch = chunk_texts[i:i + batch_size]
                    logger.info(f"Processing embedding batch {i//batch_size + 1}/{(len(chunk_texts) + batch_size - 1)//batch_size}")
                    
                    batch_embeddings = self.embedding_model.encode(batch, convert_to_numpy=True)
                    all_embeddings.append(batch_embeddings)
                
                # Combine all embeddings
                import numpy as np
                embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
                
            except Exception as embed_error:
                logger.error(f"❌ Failed to generate embeddings for {source}: {embed_error}")
                return 0
            
            # Validate embeddings
            if embeddings.shape[0] != len(chunks):
                logger.error(f"❌ Embedding count mismatch for {source}: {embeddings.shape[0]} != {len(chunks)}")
                return 0
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to FAISS index
            try:
                self.index.add(embeddings)
                # IndexFlatIP - no training needed, perfect accuracy guaranteed
                
            except Exception as faiss_error:
                logger.error(f"❌ Failed to add embeddings to FAISS index for {source}: {faiss_error}")
                return 0
            
            # Store document chunks
            start_idx = len(self.documents)
            self.documents.extend(chunks)
            
            # Store document metadata
            self.doc_metadata[doc_hash] = {
                'source': source,
                'chunk_count': len(chunks),
                'start_idx': start_idx,
                'end_idx': len(self.documents),
                'metadata': metadata or {},
                'timestamp': datetime.now().isoformat(),
                'original_size': len(text),
                'coverage_ratio': coverage_ratio
            }
            
            self.is_dirty = True
            logger.info(f"✅ Successfully added document {source} with {len(chunks)} chunks to vector store")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"❌ Critical error adding document {source} to vector store: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return 0
    
    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.1, 
               expansion_level: str = 'parent') -> List[Tuple[DocumentChunk, float]]:
        """
        Search for relevant document chunks
        
        Args:
            query: Search query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score
            expansion_level: 'child', 'parent', or 'grandparent'
            
        Returns:
            List of (DocumentChunk, score) tuples
        """
        try:
            if self.index.ntotal == 0:
                logger.info("Vector store is empty, no search results")
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Search FAISS index
            scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
            
            # Filter and format results with hierarchical expansion
            results = []
            seen_expansions = set()  # Avoid duplicate expanded chunks
            
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and score >= score_threshold and idx < len(self.documents):
                    child_chunk = self.documents[idx]
                    
                    if expansion_level == 'grandparent' and child_chunk.grandparent_text:
                        # Expand to grandparent for maximum context
                        expansion_id = f"{child_chunk.source}_grandparent_{child_chunk.metadata.get('grandparent_index', 0)}"
                        
                        if expansion_id not in seen_expansions:
                            expanded_chunk = DocumentChunk(
                                text=child_chunk.grandparent_text,
                                source=child_chunk.source,
                                chunk_id=expansion_id,
                                metadata={
                                    **child_chunk.metadata,
                                    'expanded_from_child': child_chunk.chunk_id,
                                    'expansion_level': 'grandparent',
                                    'comprehensive_context': True,
                                    'matched_query': query
                                }
                            )
                            results.append((expanded_chunk, float(score)))
                            seen_expansions.add(expansion_id)
                                
                    elif expansion_level == 'parent' and child_chunk.parent_text:
                        # Expand to parent for medium context
                        expansion_id = f"{child_chunk.source}_parent_{child_chunk.metadata.get('grandparent_index', 0)}_{child_chunk.metadata.get('parent_index', 0)}"
                        
                        if expansion_id not in seen_expansions:
                            expanded_chunk = DocumentChunk(
                                text=child_chunk.parent_text,
                                source=child_chunk.source,
                                chunk_id=expansion_id,
                                metadata={
                                    **child_chunk.metadata,
                                    'expanded_from_child': child_chunk.chunk_id,
                                    'expansion_level': 'parent',
                                    'local_context': True,
                                    'matched_query': query
                                }
                            )
                            results.append((expanded_chunk, float(score)))
                            seen_expansions.add(expansion_id)
                                
                    elif expansion_level == 'child':
                        # Use original child chunk for precise search
                        child_chunk.metadata['matched_query'] = query
                        results.append((child_chunk, float(score)))
                    else:
                        # Default to parent expansion
                        if child_chunk.parent_text:
                            expansion_id = f"{child_chunk.source}_parent_{child_chunk.metadata.get('grandparent_index', 0)}_{child_chunk.metadata.get('parent_index', 0)}"
                            if expansion_id not in seen_expansions:
                                expanded_chunk = DocumentChunk(
                                    text=child_chunk.parent_text,
                                    source=child_chunk.source,
                                    chunk_id=expansion_id,
                                    metadata={
                                        **child_chunk.metadata,
                                        'expanded_from_child': child_chunk.chunk_id,
                                        'expansion_level': 'parent',
                                        'matched_query': query
                                    }
                                )
                                results.append((expanded_chunk, float(score)))
                                seen_expansions.add(expansion_id)
                        else:
                            child_chunk.metadata['matched_query'] = query
                            results.append((child_chunk, float(score)))
            
            logger.info(f"Vector search for '{query[:50]}...' returned {len(results)} results")
            expanded_count = len([r for r in results if 'expanded_from_child' in r[0].metadata])
            if expanded_count > 0:
                logger.info(f"Expanded {expanded_count} results to {expansion_level} level")
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    
    def extract_structured_data(self, query: str, top_k: int = 20) -> Dict[str, List[str]]:
        """Extract structured data like URLs, emails, names using regex post-processing"""
        try:
            # Get relevant documents
            results = self.search(query, top_k=top_k, expansion_level='parent')
            
            if not results:
                return {'urls': [], 'emails': [], 'github_urls': []}
            
            # Combine all text for regex processing
            all_text = " ".join([chunk.text for chunk, _ in results])
            
            # Extract different types of structured data
            extracted = {
                'github_urls': self._extract_github_urls(all_text),
                'urls': self._extract_urls(all_text),
                'emails': self._extract_emails(all_text),
                'phone_numbers': self._extract_phone_numbers(all_text)
            }
            
            logger.info(f"Extracted structured data: {sum(len(v) for v in extracted.values())} items total")
            for key, items in extracted.items():
                if items:
                    logger.info(f"  - {key}: {len(items)} items")
            
            return extracted
            
        except Exception as e:
            logger.error(f"Error extracting structured data: {e}")
            return {'urls': [], 'emails': [], 'github_urls': [], 'phone_numbers': []}
    
    def _extract_github_urls(self, text: str) -> List[str]:
        """Extract GitHub URLs with high precision"""
        pattern = r'https?://github\.com/[a-zA-Z0-9\-_]+/[a-zA-Z0-9\-_\.]+(?:/[a-zA-Z0-9\-_/]*)?'
        urls = re.findall(pattern, text)
        return sorted(list(set(urls)))
    
    def _extract_urls(self, text: str) -> List[str]:
        """Extract all URLs"""
        pattern = r'https?://[a-zA-Z0-9\-._~:/?#\[\]@!$&\'()*+,;=%]+'
        urls = re.findall(pattern, text)
        return sorted(list(set(urls)))
    
    def _extract_emails(self, text: str) -> List[str]:
        """Extract email addresses"""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(pattern, text)
        return sorted(list(set(emails)))
    
    def _extract_phone_numbers(self, text: str) -> List[str]:
        """Extract phone numbers in various formats"""
        patterns = [
            r'\b\d{3}-\d{3}-\d{4}\b',  # 123-456-7890
            r'\b\(\d{3}\)\s*\d{3}-\d{4}\b',  # (123) 456-7890
            r'\b\d{3}\.\d{3}\.\d{4}\b',  # 123.456.7890
        ]
        
        phones = []
        for pattern in patterns:
            phones.extend(re.findall(pattern, text))
        
        return sorted(list(set(phones)))
    
    def get_all_documents_context(self, max_chars: int = 50000) -> str:
        """
        Get context from all documents in the store (for comprehensive analysis)
        
        Args:
            max_chars: Maximum total characters in context
            
        Returns:
            Formatted context string with excerpts from all documents
        """
        try:
            if not self.documents:
                return ""
            
            context_parts = []
            total_chars = 0
            
            # Group documents by source
            by_source = {}
            for doc in self.documents:
                if doc.source not in by_source:
                    by_source[doc.source] = []
                by_source[doc.source].append(doc)
            
            num_documents = len(by_source)
            logger.info(f"Processing {num_documents} documents for comprehensive analysis - using ALL available chunks")
            logger.info(f"Maximum context limit: {max_chars:,} characters")
            
            # Use ALL chunks from each document for comprehensive coverage
            for source, chunks in by_source.items():
                source_text = f"\n--- From {source} ---\n"
                context_parts.append(source_text)
                total_chars += len(source_text)
                
                logger.info(f"Processing ALL {len(chunks)} chunks from {source}")
                
                # Use ALL chunks from each document
                for i, chunk in enumerate(chunks):
                    # Add full chunk text without arbitrary limits
                    chunk_text = chunk.text + "\n\n"
                    
                    # Check if adding this chunk would exceed the total limit
                    if total_chars + len(chunk_text) > max_chars:
                        # Try to fit as much as possible
                        remaining_chars = max_chars - total_chars
                        if remaining_chars > 100:  # Only add if we have meaningful space
                            chunk_text = chunk.text[:remaining_chars - 50] + "...\n\n"
                            context_parts.append(chunk_text)
                            total_chars += len(chunk_text)
                        logger.info(f"Reached character limit after {i+1}/{len(chunks)} chunks from {source}")
                        break
                    
                    context_parts.append(chunk_text)
                    total_chars += len(chunk_text)
                
                # Log completion for this document
                logger.info(f"Added complete content from {source}: {len([c for c in chunks if any(c.text in part for part in context_parts)])} chunks included")
                
                # Stop if we've reached the total limit
                if total_chars >= max_chars:
                    break
            
            context = "".join(context_parts).strip()
            logger.info(f"Generated comprehensive context: {len(context)} chars from {len(by_source)} documents")
            return context
            
        except Exception as e:
            logger.error(f"Error generating comprehensive context: {e}")
            return ""

    def get_context_for_query(self, query: str, max_chunks: int = 50, max_chars: int = 50000, 
                             extraction_mode: bool = False, expansion_level: str = 'parent') -> str:
        """
        Get relevant context for a query, formatted for LLM consumption
        
        Args:
            query: User query
            max_chunks: Maximum number of chunks to include
            max_chars: Maximum total characters in context
            
        Returns:
            Formatted context string
        """
        try:
            # Determine expansion level based on query type and extraction mode
            if extraction_mode or any(keyword in query.lower() for keyword in 
                ['extract', 'list all', 'find all', 'all applicant', 'comprehensive']):
                expansion_level = 'grandparent'  # Use largest context for extraction
            elif any(keyword in query.lower() for keyword in ['github', 'url', 'email', 'name']):
                expansion_level = 'parent'  # Medium context for specific searches
            else:
                expansion_level = expansion_level  # Use provided parameter
            
            logger.info(f"Using expansion level: {expansion_level} for query type")
            results = self.search(query, top_k=max_chunks, expansion_level=expansion_level)
            
            if not results:
                return ""
            
            context_parts = []
            total_chars = 0
            
            # Group results by source document
            by_source = {}
            for chunk, score in results:
                source = chunk.source
                if source not in by_source:
                    by_source[source] = []
                by_source[source].append((chunk, score))
            
            # Format context by source
            for source, source_chunks in by_source.items():
                source_text = f"\n--- From {source} ---\n"
                
                for chunk, score in source_chunks:
                    chunk_text = f"{chunk.text}\n"
                    
                    # Check if adding this chunk would exceed character limit
                    if total_chars + len(source_text) + len(chunk_text) > max_chars:
                        break
                    
                    if not any(source in part for part in context_parts):
                        context_parts.append(source_text)
                        total_chars += len(source_text)
                    
                    context_parts.append(chunk_text)
                    total_chars += len(chunk_text)
                
                if total_chars >= max_chars:
                    break
            
            context = "".join(context_parts).strip()
            logger.info(f"Generated context: {len(context)} chars from {len(by_source)} documents")
            return context
            
        except Exception as e:
            logger.error(f"Error generating context for query: {e}")
            return ""
    
    def get_all_chunks_for_extraction(self, source_filter: str = None) -> List[DocumentChunk]:
        """Get ALL chunks for comprehensive extraction tasks (map-reduce pattern)"""
        try:
            if source_filter:
                # Filter by specific document
                return [chunk for chunk in self.documents if chunk.source == source_filter]
            else:
                # Return all chunks from all documents
                return list(self.documents)
        except Exception as e:
            logger.error(f"Error getting all chunks: {e}")
            return []
    
    def get_full_documents_context(self, max_chars: int = None) -> str:
        """
        Get ALL content from ALL documents - no limits for vector store comprehensive analysis
        
        Args:
            max_chars: Character limit (None = no limit for vector store mode)
            
        Returns:
            Complete document content formatted for LLM consumption
        """
        try:
            if not self.documents:
                return ""
            
            context_parts = []
            total_chars = 0
            
            # Group documents by source
            by_source = {}
            for doc in self.documents:
                if doc.source not in by_source:
                    by_source[doc.source] = []
                by_source[doc.source].append(doc)
            
            if max_chars:
                logger.info(f"🔄 Retrieving content from {len(by_source)} documents (limit: {max_chars:,} chars)")
            else:
                logger.info(f"🔄 Retrieving ALL content from {len(by_source)} documents (no character limit - full vector store mode)")
            
            # Process ALL chunks from ALL documents
            for source, chunks in by_source.items():
                source_text = f"\n=== COMPLETE CONTENT FROM {source.upper()} ===\n\n"
                context_parts.append(source_text)
                total_chars += len(source_text)
                
                logger.info(f"📋 Adding ALL {len(chunks)} chunks from {source}")
                
                chunks_added = 0
                for chunk in chunks:
                    chunk_text = chunk.text + "\n\n"
                    
                    # Memory safety: check both total limit and individual chunk size
                    if len(chunk_text) > 50000:  # Skip extremely large chunks that might cause issues
                        logger.warning(f"⚠️ Skipping very large chunk ({len(chunk_text):,} chars) from {source}")
                        continue
                    
                    # Only stop if we have a character limit (traditional mode compatibility)
                    # For applicant extraction, max_chars=None, so this won't trigger
                    if max_chars and total_chars + len(chunk_text) > max_chars:
                        logger.warning(f"⚠️ Reached {max_chars:,} char limit after {chunks_added}/{len(chunks)} chunks from {source}")
                        logger.info(f"📊 Document had {len(chunks)} total chunks, retrieved {chunks_added} chunks ({(chunks_added/len(chunks)*100):.1f}%)")
                        break
                    
                    context_parts.append(chunk_text)
                    total_chars += len(chunk_text)
                    chunks_added += 1
                    
                    # Safety check: prevent memory issues with too many parts (only if max_chars is set)
                    # Skip this check when max_chars=None (Smart Extractor mode)
                    if max_chars is not None and len(context_parts) > 2000:  # Higher limit, only when char limits active
                        logger.warning(f"⚠️ Reached maximum context parts limit (2000) for {source}")
                        break
                
                # Calculate actual character count from context parts for this source
                source_chars = sum(len(part) for part in context_parts if f"From {source.upper()}" in part or source in part)
                logger.info(f"✅ Added {chunks_added}/{len(chunks)} chunks from {source} (~{source_chars:,} chars)")
            
            # Debug: Check if we're missing chunks
            if chunks_added < len(chunks):
                logger.warning(f"⚠️  PARTIAL RETRIEVAL: Only got {chunks_added}/{len(chunks)} chunks ({(chunks_added/len(chunks)*100):.1f}%) from {source}")
                logger.warning(f"⚠️  Missing {len(chunks) - chunks_added} chunks - this will cause incomplete extraction results!")
            else:
                logger.info(f"✅ COMPLETE RETRIEVAL: Got all {chunks_added} chunks from {source}")
                
                # Add document separator (only if we have character limits)
                if max_chars is None or total_chars < max_chars - 100:
                    separator = f"\n--- End of {source} ---\n\n"
                    context_parts.append(separator)
                    total_chars += len(separator)
            
            context = "".join(context_parts).strip()
            
            # Calculate total chunks vs retrieved chunks
            total_chunks_available = sum(len(chunks) for chunks in by_source.values())
            total_chunks_retrieved = sum(len([c for c in chunks if any(c.text in part for part in context_parts)]) for chunks in by_source.values())
            
            logger.info(f"🎯 Generated COMPLETE context: {len(context):,} chars from {len(by_source)} documents")
            logger.info(f"📊 Chunk Statistics: {total_chunks_retrieved}/{total_chunks_available} chunks retrieved ({(total_chunks_retrieved/total_chunks_available*100):.1f}%)")
            
            if total_chunks_retrieved < total_chunks_available:
                logger.error(f"❌ INCOMPLETE EXTRACTION EXPECTED: Missing {total_chunks_available - total_chunks_retrieved} chunks!")
            else:
                logger.info(f"✅ COMPLETE EXTRACTION POSSIBLE: All chunks retrieved")
                
            return context
            
        except Exception as e:
            logger.error(f"❌ Error generating complete documents context: {e}")
            return ""
    
    def remove_document(self, source: str) -> bool:
        """
        Remove a document from the vector store
        Note: FAISS doesn't support deletion, so this marks as removed
        """
        try:
            # Find document by source
            doc_hash = None
            for hash_key, metadata in self.doc_metadata.items():
                if metadata['source'] == source:
                    doc_hash = hash_key
                    break
            
            if not doc_hash:
                logger.info(f"Document {source} not found in vector store")
                return False
            
            # Mark as removed (FAISS doesn't support true deletion)
            self.doc_metadata[doc_hash]['removed'] = True
            self.is_dirty = True
            
            logger.info(f"Marked document {source} as removed from vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error removing document {source}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics with detailed document info"""
        active_docs = []
        total_chunks = 0
        
        for doc_hash, meta in self.doc_metadata.items():
            if not meta.get('removed', False):
                doc_info = {
                    'source': meta['source'],
                    'chunks': meta['chunk_count'],
                    'original_size': meta.get('original_size', 0),
                    'coverage': meta.get('coverage_ratio', 0),
                    'timestamp': meta['timestamp']
                }
                active_docs.append(doc_info)
                total_chunks += meta['chunk_count']
        
        return {
            'session_id': self.session_id,
            'total_documents': len(active_docs),
            'total_chunks': total_chunks,
            'index_size': self.index.ntotal,
            'embedding_model': self.embedding_model_name,
            'embedding_dim': self.embedding_dim,
            'is_dirty': self.is_dirty,
            'documents': active_docs  # Detailed document list
        }
    
    def get_document_sample(self, source: str, max_chunks: int = 5) -> List[str]:
        """Get sample chunks from a specific document for debugging"""
        try:
            logger.info(f"🔍 Searching for chunks from source: '{source}'")
            logger.info(f"🔍 Total documents in store: {len(self.documents)}")
            
            # Debug: show all unique sources
            all_sources = set(chunk.source for chunk in self.documents)
            logger.info(f"🔍 All sources in vector store: {list(all_sources)}")
            
            sample_chunks = []
            for chunk in self.documents:
                logger.debug(f"🔍 Comparing chunk source '{chunk.source}' with target '{source}'")
                if chunk.source == source:
                    sample_chunks.append({
                        'chunk_id': chunk.chunk_id,
                        'text_preview': chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                        'text_length': len(chunk.text)
                    })
                    if len(sample_chunks) >= max_chunks:
                        break
            
            logger.info(f"🔍 Found {len(sample_chunks)} matching chunks for '{source}'")
            return sample_chunks
        except Exception as e:
            logger.error(f"❌ Error getting document sample for {source}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return []
    
    def clear(self):
        """Clear all documents from the vector store"""
        try:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.documents = []
            self.doc_metadata = {}
            self.is_dirty = True
            logger.info("Cleared vector store")
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")

class VectorStoreManager:
    """Manages vector store instances per session"""
    
    def __init__(self):
        """Initialize vector store manager without S3 dependency"""
        self.sessions: Dict[str, FAISSVectorStore] = {}
        logger.info("Vector store manager initialized independently of S3")
    
    def get_or_create_store(self, session_id: str) -> Optional[FAISSVectorStore]:
        """Get or create a vector store for a session"""
        try:
            if not FAISS_AVAILABLE:
                logger.warning("FAISS not available, cannot create vector store")
                return None
            
            if session_id not in self.sessions:
                self.sessions[session_id] = FAISSVectorStore(session_id)
                logger.info(f"Created new vector store for session: {session_id}")
            
            return self.sessions[session_id]
            
        except Exception as e:
            logger.error(f"Error creating vector store for session {session_id}: {e}")
            return None
    
    def add_pdf_document_from_s3(self, session_id: str, s3_key: str, s3_bucket: str, source_filename: str, metadata: dict = None) -> bool:
        """Add a PDF document to vector store from S3 using unstructured.io (placeholder for direct PDF processing)"""
        try:
            # Note: This is a placeholder implementation since we don't have unstructured.io
            # For now, we'll return False to indicate that this method isn't fully implemented
            # The calling code should fall back to markdown conversion
            logger.warning(f"add_pdf_document_from_s3 called for {source_filename} but direct PDF processing not implemented")
            logger.info(f"Falling back to markdown conversion for PDF: {source_filename}")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error in add_pdf_document_from_s3 for {source_filename}: {e}")
            return False
    
    def add_document_from_content(self, session_id: str, content: str, source_filename: str, metadata: dict = None) -> bool:
        """Add a document to vector store from content string - SIMPLE approach"""
        try:
            # Get vector store
            store = self.get_or_create_store(session_id)
            if not store:
                logger.error(f"❌ Failed to get/create vector store for session {session_id}")
                return False
            
            # Add to vector store directly
            chunks_added = store.add_document(content, source_filename, metadata or {})
            logger.info(f"✅ Added document {source_filename} to vector store: {chunks_added} chunks")
            return chunks_added > 0
            
        except Exception as e:
            logger.error(f"❌ Error adding document {source_filename} to vector store: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def add_document_from_file(self, session_id: str, file_path: str, source_filename: str) -> bool:
        """Add a document to vector store from local file"""
        try:
            # Get vector store
            store = self.get_or_create_store(session_id)
            if not store:
                return False
            
            # Read document from file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Add to vector store
            chunks_added = store.add_document(content, source_filename, {'file_path': file_path})
            logger.info(f"Added document {source_filename} from file to vector store: {chunks_added} chunks")
            return chunks_added > 0
            
        except Exception as e:
            logger.error(f"Error adding document from file to vector store: {e}")
            return False
    
    def search_session(self, session_id: str, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Search documents in a session's vector store"""
        try:
            if session_id not in self.sessions:
                logger.info(f"No vector store found for session: {session_id}")
                return []
            
            return self.sessions[session_id].search(query, top_k)
            
        except Exception as e:
            logger.error(f"Error searching session {session_id}: {e}")
            return []
    
    def get_all_chunks_for_extraction(self, session_id: str, source_filter: str = None) -> List[DocumentChunk]:
        """Get ALL chunks for comprehensive extraction using map-reduce pattern"""
        try:
            if session_id not in self.sessions:
                logger.info(f"No vector store found for session: {session_id}")
                return []
            
            return self.sessions[session_id].get_all_chunks_for_extraction(source_filter)
            
        except Exception as e:
            logger.error(f"Error getting all chunks for session {session_id}: {e}")
            return []
    
    def get_all_documents_for_session(self, session_id: str, max_chars: int = None, force_full_retrieval: bool = False) -> str:
        """Get context from all documents in a session's vector store"""
        try:
            if session_id not in self.sessions:
                logger.info(f"No vector store found for session: {session_id}")
                return ""
            
            if force_full_retrieval:
                # Use the new comprehensive method with much higher limits
                return self.sessions[session_id].get_full_documents_context(max_chars)
            else:
                return self.sessions[session_id].get_all_documents_context(max_chars)
            
        except Exception as e:
            logger.error(f"Error getting all documents for session {session_id}: {e}")
            return ""

    def get_context_for_session(self, session_id: str, query: str, max_chunks: int = 20, max_chars: int = 25000, extraction_mode: bool = False, expansion_level: str = 'parent') -> str:
        """Get relevant context for a query from session's vector store with extraction optimization"""
        try:
            if session_id not in self.sessions:
                logger.info(f"No vector store found for session: {session_id}")
                return ""
            
            return self.sessions[session_id].get_context_for_query(
                query, max_chunks, max_chars, extraction_mode=extraction_mode, expansion_level=expansion_level
            )
            
        except Exception as e:
            logger.error(f"Error getting context for session {session_id}: {e}")
            return ""
    
    def extract_structured_data_from_session(self, session_id: str, query: str, top_k: int = 30) -> Dict[str, List[str]]:
        """Extract structured data (URLs, emails, etc.) from session's vector store"""
        try:
            if session_id not in self.sessions:
                logger.info(f"No vector store found for session: {session_id}")
                return {'urls': [], 'emails': [], 'github_urls': [], 'phone_numbers': []}
            
            return self.sessions[session_id].extract_structured_data(query, top_k)
            
        except Exception as e:
            logger.error(f"Error extracting structured data for session {session_id}: {e}")
            return {'urls': [], 'emails': [], 'github_urls': [], 'phone_numbers': []}
    
    def remove_document_from_session(self, session_id: str, source_filename: str) -> bool:
        """Remove a document from session's vector store"""
        try:
            if session_id not in self.sessions:
                logger.info(f"No vector store found for session: {session_id}")
                return False
            
            return self.sessions[session_id].remove_document(source_filename)
            
        except Exception as e:
            logger.error(f"Error removing document from session {session_id}: {e}")
            return False
    
    def clear_session(self, session_id: str):
        """Clear a session's vector store"""
        try:
            if session_id in self.sessions:
                self.sessions[session_id].clear()
                del self.sessions[session_id]
                logger.info(f"Cleared vector store for session: {session_id}")
        except Exception as e:
            logger.error(f"Error clearing session {session_id}: {e}")
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session's vector store"""
        try:
            if session_id not in self.sessions:
                return {'error': 'Session not found'}
            
            return self.sessions[session_id].get_stats()
            
        except Exception as e:
            logger.error(f"Error getting stats for session {session_id}: {e}")
            return {'error': str(e)}
    
    def get_document_sample(self, session_id: str, source: str, max_chunks: int = 5) -> List[str]:
        """Get sample chunks from a specific document for debugging"""
        try:
            if session_id not in self.sessions:
                return []
            
            return self.sessions[session_id].get_document_sample(source, max_chunks)
            
        except Exception as e:
            logger.error(f"Error getting document sample for session {session_id}: {e}")
            return []

# Global vector store manager instance
vector_store_manager = None

def initialize_vector_store_manager() -> Optional[VectorStoreManager]:
    """Initialize the global vector store manager - S3 independent"""
    global vector_store_manager
    
    try:
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, vector store disabled")
            return None
        
        vector_store_manager = VectorStoreManager()
        logger.info("Initialized vector store manager independently")
        return vector_store_manager
        
    except Exception as e:
        logger.error(f"Error initializing vector store manager: {e}")
        return None

def get_vector_store_manager() -> Optional[VectorStoreManager]:
    """Get the global vector store manager instance"""
    return vector_store_manager

# Utility function to check if vector store is available
def is_vector_store_available() -> bool:
    """Check if vector store functionality is available"""
    return FAISS_AVAILABLE and vector_store_manager is not None