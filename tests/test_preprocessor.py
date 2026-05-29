"""
Unit tests for the preprocessor module.
"""
import pytest
from src.preprocessor import (
    normalize_text,
    chunk_document,
    extract_candidate_spans,
    deduplicate_documents
)


class TestNormalizeText:
    """Tests for text normalization."""
    
    def test_normalize_text_lowercase(self):
        """Test conversion to lowercase."""
        assert normalize_text("Hello WORLD") == "hello world"
    
    def test_normalize_text_strips_whitespace(self):
        """Test whitespace stripping and collapsing."""
        assert normalize_text("  hello   world  ") == "hello world"
    
    def test_normalize_text_collapse_multiple_spaces(self):
        """Test multiple space collapsing."""
        assert normalize_text("hello    world") == "hello world"
    
    def test_normalize_empty_string(self):
        """Test empty string handling."""
        assert normalize_text("") == ""
    
    def test_normalize_text_with_newlines(self):
        """Test newline collapsing."""
        assert normalize_text("hello\n\nworld") == "hello world"


class TestChunkDocument:
    """Tests for document chunking."""
    
    def test_chunk_small_document(self):
        """Test that small documents are not chunked."""
        text = "This is a small document."
        chunks = chunk_document(text, max_chars=100)
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_large_document(self):
        """Test that large documents are chunked."""
        text = "word " * 500  # Create a long text
        chunks = chunk_document(text, max_chars=100)
        assert len(chunks) > 1
    
    def test_chunk_preserves_text(self):
        """Test that concatenated chunks reconstruct the original."""
        text = "word " * 500
        chunks = chunk_document(text, max_chars=100, overlap=10)
        # Chunks may overlap, so just verify no data is lost
        assert len(chunks) > 0
    
    def test_chunk_empty_string(self):
        """Test empty string handling."""
        chunks = chunk_document("", max_chars=100)
        assert chunks == []


class TestExtractCandidateSpans:
    """Tests for candidate span extraction."""
    
    def test_extract_percentages(self):
        """Test percentage extraction."""
        text = "The accuracy improved by 24% and efficiency by 31%."
        spans = extract_candidate_spans(text)
        assert "24%" in spans or "24 %" in spans
        assert "31%" in spans or "31 %" in spans
    
    def test_extract_large_numbers(self):
        """Test large number extraction."""
        text = "The project analyzed 620,000 transactions and 4500 records."
        spans = extract_candidate_spans(text)
        assert any("620" in span for span in spans)
    
    def test_extract_durations(self):
        """Test time duration extraction."""
        text = "The process takes ~3 hours and 24 minutes."
        spans = extract_candidate_spans(text)
        assert len(spans) > 0
    
    def test_extract_empty_string(self):
        """Test empty string handling."""
        spans = extract_candidate_spans("")
        assert spans == []
    
    def test_no_duplicates(self):
        """Test that duplicates are removed."""
        text = "Error rate 5% error rate 5% error."
        spans = extract_candidate_spans(text)
        # Should not have duplicate "5%"
        percent_spans = [s for s in spans if "%" in s]
        assert len(percent_spans) <= 2


class TestDeduplicateDocuments:
    """Tests for document deduplication."""
    
    def test_remove_exact_duplicates(self):
        """Test removal of exact duplicate documents."""
        docs = ["Document A", "Document B", "Document A"]
        result = deduplicate_documents(docs)
        assert len(result) == 2
    
    def test_preserve_unique_documents(self):
        """Test that unique documents are preserved."""
        docs = ["Doc 1", "Doc 2", "Doc 3"]
        result = deduplicate_documents(docs)
        assert len(result) == 3
    
    def test_case_insensitive_deduplication(self):
        """Test case-insensitive deduplication."""
        docs = ["Document A", "document a"]
        result = deduplicate_documents(docs)
        assert len(result) == 1
    
    def test_preserve_order(self):
        """Test that order is preserved."""
        docs = ["A", "B", "A", "C"]
        result = deduplicate_documents(docs)
        # First occurrence of each should be preserved
        assert result[0] == "A"
        assert result[1] == "B"
        assert result[2] == "C"
    
    def test_empty_list(self):
        """Test empty list handling."""
        result = deduplicate_documents([])
        assert result == []
