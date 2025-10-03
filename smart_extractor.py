"""
Smart Extractor - Hybrid LLM→Regex System for Comprehensive Document Analysis
"""

import os
import re
import json
import logging
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

# Set up logger
logger = logging.getLogger(__name__)

class SmartExtractor:
    """
    Hybrid extraction system that uses LLM to develop regex patterns,
    then applies them to full documents for comprehensive extraction
    """
    
    def __init__(self, bedrock_client, model_id: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"):
        """
        Initialize Smart Extractor
        
        Args:
            bedrock_client: AWS Bedrock client
            model_id: Bedrock model to use for pattern development
        """
        self.bedrock_client = bedrock_client
        self.model_id = model_id
        self.pattern_cache = {}  # Cache discovered patterns
        
        logger.info(f"SmartExtractor initialized with model: {model_id}")
    
    def extract_comprehensive(self, full_content: str, extraction_type: str, 
                            sample_size: int = 300000) -> Dict[str, Any]:
        """
        Main extraction method: LLM develops patterns → Regex processes full document
        
        Args:
            full_content: Complete document content (markdown)
            extraction_type: Type of extraction (e.g., "applicants", "emails", "github_urls")
            sample_size: Size of sample for pattern development
            
        Returns:
            Dictionary with extracted data and metadata
        """
        try:
            logger.info(f"🧠 Starting smart extraction for: {extraction_type}")
            logger.info(f"📄 Full document size: {len(full_content):,} characters")
            
            # Step 1: Create sample for pattern development
            sample_content = self._create_representative_sample(full_content, sample_size)
            logger.info(f"📋 Created sample: {len(sample_content):,} characters")
            
            # Step 2: LLM develops regex patterns
            patterns = self._develop_patterns(sample_content, extraction_type)
            if not patterns:
                logger.error(f"❌ Failed to develop patterns for {extraction_type}")
                return {"error": "Pattern development failed", "results": []}
            
            logger.info(f"🔍 Developed {len(patterns)} regex patterns")
            
            # Step 3: Apply patterns to full document
            results = self._apply_patterns_to_document(full_content, patterns, extraction_type)
            
            # Step 4: Post-process and format results
            formatted_results = self._format_results(results, extraction_type)
            
            logger.info(f"✅ Smart extraction complete: {len(formatted_results.get('results', []))} items found")
            
            return {
                "extraction_type": extraction_type,
                "total_found": len(formatted_results.get('results', [])),
                "patterns_used": len(patterns),
                "document_size": len(full_content),
                "results": formatted_results.get('results', []),
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "patterns": patterns,
                    "sample_size": len(sample_content)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Smart extraction failed: {e}")
            return {"error": str(e), "results": []}
    
    def _create_representative_sample(self, content: str, sample_size: int) -> str:
        """
        Create a representative sample from the document for pattern development
        """
        if len(content) <= sample_size:
            return content
        
        # Strategy: Take beginning, middle, and end to get representative patterns
        chunk_size = sample_size // 3
        
        beginning = content[:chunk_size]
        middle_start = len(content) // 2 - chunk_size // 2
        middle = content[middle_start:middle_start + chunk_size]
        end = content[-chunk_size:]
        
        sample = f"{beginning}\n\n--- MIDDLE SECTION ---\n\n{middle}\n\n--- END SECTION ---\n\n{end}"
        
        logger.info(f"📝 Sample composition: Beginning({len(beginning)}), Middle({len(middle)}), End({len(end)})")
        return sample
    
    def _develop_patterns(self, sample_content: str, extraction_type: str) -> List[Dict[str, str]]:
        """
        Use LLM to analyze sample and develop regex patterns
        """
        try:
            # Check cache first
            cache_key = f"{extraction_type}_{hash(sample_content[:1000])}"
            if cache_key in self.pattern_cache:
                logger.info("🔄 Using cached patterns")
                return self.pattern_cache[cache_key]
            
            # Create extraction-specific prompt
            pattern_prompt = self._create_pattern_prompt(sample_content, extraction_type)
            
            logger.info("🧠 Sending pattern development request to LLM...")
            
            # Build API parameters with Claude Sonnet 4 context window support
            messages = [{
                "role": "user",
                "content": [{"text": pattern_prompt}]
            }]
            inference_config = {
                "maxTokens": 2000,
                "temperature": 0.1,  # Low temperature for precise pattern development
                "topP": 0.9
            }
            
            api_params = {
                "modelId": self.model_id,
                "messages": messages,
                "inferenceConfig": inference_config
            }
            
            # Add extended context window for Claude Sonnet 4 models only
            # Note: Claude Opus 4 doesn't support the context-1m-2025-08-07 beta flag
            if "anthropic.claude-sonnet-4-" in self.model_id:
                api_params["additionalModelRequestFields"] = {
                    "anthropic_beta": ["context-1m-2025-08-07"]
                }
                logger.info(f"Added 1M token context window for Claude Sonnet 4 model: {self.model_id}")
            
            response = self.bedrock_client.converse(**api_params)
            
            response_text = response['output']['message']['content'][0]['text']
            
            # Parse patterns from LLM response
            patterns = self._parse_pattern_response(response_text)
            
            # Cache successful patterns
            if patterns:
                self.pattern_cache[cache_key] = patterns
            
            return patterns
            
        except Exception as e:
            logger.error(f"❌ Pattern development failed: {e}")
            return []
    
    def _create_pattern_prompt(self, sample_content: str, extraction_type: str) -> str:
        """
        Create LLM prompt for pattern development based on extraction type
        """
        base_prompt = f"""You are a regex pattern development expert. Analyze this document sample and create Python regex patterns to extract {extraction_type}.

DOCUMENT SAMPLE:
{sample_content}

TASK: Create regex patterns that will reliably extract {extraction_type} from similar documents.

"""
        
        if extraction_type.lower() in ['applicants', 'applicant', 'primary applicants']:
            specific_prompt = """
Focus on finding patterns for:
1. Primary applicant names (usually after "Primary Investigator:", "PI:", "Lead:", etc.)
2. Application numbers/IDs (like #1001, #1002, etc.)
3. Name-ID pairs
4. Section headers that indicate new applications

Provide working Python regex patterns with named groups. Format your response as JSON:
```json
{
  "patterns": [
    {
      "name": "primary_investigator",
      "pattern": "Primary Investigator:\\s*([^\\n]+)",
      "description": "Extracts primary investigator names"
    },
    {
      "name": "application_id", 
      "pattern": "#(\\d{4})",
      "description": "Extracts application IDs"
    }
  ]
}
```

Make sure patterns are:
- Robust (work with slight formatting variations)
- Specific (avoid false positives)
- Complete (capture all variations you see in the sample)
"""
        
        elif extraction_type.lower() in ['github', 'github_urls', 'urls']:
            specific_prompt = """
Focus on finding patterns for:
1. GitHub URLs (https://github.com/...)
2. Other URLs that might be repositories
3. GitHub usernames mentioned in text

Provide working Python regex patterns. Format as JSON with pattern objects.
"""
        
        elif extraction_type.lower() in ['emails', 'email']:
            specific_prompt = """
Focus on finding patterns for:
1. Email addresses in various formats
2. Contact information sections
3. Email patterns specific to this document type

Provide working Python regex patterns. Format as JSON with pattern objects.
"""
        
        else:
            specific_prompt = f"""
Analyze the document structure and create patterns to extract {extraction_type}.
Look for consistent formatting, headers, delimiters, and text patterns.

Provide working Python regex patterns. Format as JSON with pattern objects.
"""
        
        return base_prompt + specific_prompt
    
    def _parse_pattern_response(self, response_text: str) -> List[Dict[str, str]]:
        """
        Parse LLM response to extract regex patterns
        """
        try:
            # Try to find JSON in the response
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                pattern_data = json.loads(json_str)
                patterns = pattern_data.get('patterns', [])
                
                logger.info(f"📋 Parsed {len(patterns)} patterns from JSON response")
                return patterns
            
            # Fallback: look for patterns in code blocks
            patterns = []
            pattern_blocks = re.findall(r'```(?:python)?\s*(.*?)```', response_text, re.DOTALL)
            
            for block in pattern_blocks:
                # Look for regex patterns in the code
                regex_matches = re.findall(r'["\']([^"\']*\\[sd\\][^"\']*)["\']', block)
                for match in regex_matches:
                    patterns.append({
                        "name": f"pattern_{len(patterns)}",
                        "pattern": match,
                        "description": "Auto-extracted pattern"
                    })
            
            logger.info(f"📋 Fallback parsing found {len(patterns)} patterns")
            return patterns
            
        except Exception as e:
            logger.error(f"❌ Failed to parse pattern response: {e}")
            logger.debug(f"Response text: {response_text[:500]}...")
            return []
    
    def _apply_patterns_to_document(self, content: str, patterns: List[Dict[str, str]], 
                                  extraction_type: str) -> List[Dict[str, Any]]:
        """
        Apply developed regex patterns to the full document
        """
        try:
            logger.info(f"🔍 Applying {len(patterns)} patterns to full document...")
            all_results = []
            
            for pattern_info in patterns:
                pattern = pattern_info.get('pattern', '')
                pattern_name = pattern_info.get('name', 'unknown')
                
                try:
                    # Compile and apply regex
                    compiled_pattern = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                    matches = compiled_pattern.findall(content)
                    
                    logger.info(f"  📍 Pattern '{pattern_name}': {len(matches)} matches")
                    
                    # Store results with metadata
                    for match in matches:
                        if isinstance(match, tuple):
                            # Multiple groups - take the first non-empty one
                            value = next((g for g in match if g.strip()), str(match))
                        else:
                            value = str(match)
                        
                        all_results.append({
                            'value': value.strip(),
                            'pattern_name': pattern_name,
                            'pattern': pattern,
                            'type': extraction_type
                        })
                        
                except re.error as regex_error:
                    logger.warning(f"⚠️ Invalid regex pattern '{pattern}': {regex_error}")
                    continue
            
            logger.info(f"🎯 Total raw matches: {len(all_results)}")
            return all_results
            
        except Exception as e:
            logger.error(f"❌ Pattern application failed: {e}")
            return []
    
    def _format_results(self, raw_results: List[Dict[str, Any]], 
                       extraction_type: str) -> Dict[str, Any]:
        """
        Format and deduplicate results
        """
        try:
            if not raw_results:
                return {"results": [], "summary": "No results found"}
            
            # Generic quality filtering based on pattern match frequency
            # Filter out overly noisy patterns (patterns with too many matches are likely false positives)
            if len(raw_results) > 100:  # Only filter if we have too many results
                pattern_match_counts = {}
                for result in raw_results:
                    pattern_name = result.get('pattern_name', 'unknown')
                    pattern_match_counts[pattern_name] = pattern_match_counts.get(pattern_name, 0) + 1
                
                # Calculate median match count to identify outliers
                match_counts = list(pattern_match_counts.values())
                if match_counts:
                    match_counts.sort()
                    median_matches = match_counts[len(match_counts) // 2]
                    
                    # Filter out patterns with excessive matches (likely noise)
                    max_reasonable_matches = median_matches * 5  # Allow up to 5x median
                    
                    filtered_results = []
                    for result in raw_results:
                        pattern_name = result.get('pattern_name', 'unknown')
                        if pattern_match_counts[pattern_name] <= max_reasonable_matches:
                            filtered_results.append(result)
                    
                    if len(filtered_results) < len(raw_results):
                        logger.info(f"🎯 Filtered out noisy patterns: {len(raw_results)} → {len(filtered_results)} results")
                        logger.info(f"   Median matches per pattern: {median_matches}, Max allowed: {max_reasonable_matches}")
                        raw_results = filtered_results
            
            # Deduplicate by value (case-insensitive)
            seen = set()
            unique_results = []
            
            for result in raw_results:
                value_lower = result['value'].lower().strip()
                if value_lower and value_lower not in seen:
                    seen.add(value_lower)
                    unique_results.append(result)
            
            # Sort results
            unique_results.sort(key=lambda x: x['value'])
            
            # Create summary
            pattern_counts = {}
            for result in unique_results:
                pattern_name = result['pattern_name']
                pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1
            
            summary = {
                "total_unique": len(unique_results),
                "total_raw": len(raw_results),
                "duplicates_removed": len(raw_results) - len(unique_results),
                "pattern_breakdown": pattern_counts
            }
            
            logger.info(f"📊 Results summary: {summary}")
            
            return {
                "results": unique_results,
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"❌ Result formatting failed: {e}")
            return {"results": raw_results, "summary": {"error": str(e)}}

# Utility function to get markdown content from session
def get_session_markdown_content(session_id: str, source_filter: str = None) -> str:
    """
    Get markdown content for a session (placeholder - needs integration with bedbot)
    """
    # This would be implemented to get the actual markdown content
    # from the session's uploaded files
    logger.warning("get_session_markdown_content needs integration with bedbot session management")
    return ""

# Example usage and testing
if __name__ == "__main__":
    # This would be used for testing
    print("SmartExtractor - Hybrid LLM→Regex Extraction System")
    print("Ready for integration with bedbot.py")