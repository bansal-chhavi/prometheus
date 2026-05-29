"""
Preprocessor module: Text normalization, chunking, and span extraction.
"""
import re
from typing import Optional


def normalize_text(text: str) -> str:
    """
    Normalize text: lowercase, strip extra whitespace, remove special chars for normalization.
    
    Args:
        text: Raw text to normalize
        
    Returns:
        Normalized text string
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Collapse multiple whitespace into single space
    text = re.sub(r'\s+', ' ', text)
    
    return text


def chunk_document(text: str, max_chars: int = 1500, overlap: int = 200) -> list[str]:
    """
    Split large documents into overlapping chunks for better LLM processing.
    
    Args:
        text: Document text to chunk
        max_chars: Maximum characters per chunk
        overlap: Number of overlapping characters between chunks
        
    Returns:
        List of text chunks
    """
    if not text or len(text) <= max_chars:
        return [text] if text else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)
        start += max_chars - overlap
    
    return chunks


def extract_candidate_spans(text: str) -> list[str]:
    """
    Extract numeric values, percentages, and named quantities using regex.
    Useful for identifying hallucination targets like "31%" vs "24%".
    
    Args:
        text: Text to extract spans from
        
    Returns:
        List of candidate span strings
    """
    if not text:
        return []
    
    spans = []
    
    # Percentages: 24%, 31%, etc.
    percentages = re.findall(r'\b\d{1,3}(?:\.\d{1,2})?\s*%', text)
    spans.extend(percentages)
    
    # Large numbers with commas: 620,000 or without: 620000
    large_numbers = re.findall(r'\b\d{1,3}(?:,\d{3})+\b|\b\d{4,}\b', text)
    spans.extend(large_numbers)
    
    # Currency amounts: $1.2M, €500K
    currency = re.findall(r'[$€£]\s*\d+(?:\.\d{1,2})?(?:\s*[MKB])?', text)
    spans.extend(currency)
    
    # Time durations: "3 hours", "24 minutes", "~3 hours"
    durations = re.findall(r'~?\s*\d+\s*(?:hour|minute|second|day|week|month)s?', text, re.IGNORECASE)
    spans.extend(durations)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_spans = []
    for span in spans:
        normalized = span.lower().strip()
        if normalized not in seen:
            seen.add(normalized)
            unique_spans.append(span)
    
    return unique_spans


def deduplicate_documents(documents: list[str]) -> list[str]:
    """
    Remove duplicate document chunks from list.
    
    Args:
        documents: List of document chunks
        
    Returns:
        List with duplicates removed (preserving order)
    """
    seen = set()
    unique_docs = []
    
    for doc in documents:
        normalized = normalize_text(doc)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_docs.append(doc)
    
    return unique_docs
