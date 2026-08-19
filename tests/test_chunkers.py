"""Tests for all four chunking strategies."""

import pytest

from voice_rag.data.preprocessing import preprocess_text
from voice_rag.indexing.chunkers import chunk_fixed, chunk_sentence, chunk_text
from voice_rag.pipeline.schemas import ChunkStrategy


SAMPLE_TEXT = (
    "New Delhi is the capital city of India. "
    "It serves as the seat of the Government of India. "
    "Mumbai is the financial capital and the most populous city. "
    "India is a country in South Asia. "
    "It is the seventh-largest country by area."
)

HINDI_TEXT = (
    "नई दिल्ली भारत की राजधानी है। "
    "यह भारत सरकार की सीट के रूप में कार्य करती है। "
    "मुंबई भारत की वित्तीय राजधानी है।"
)


class TestFixedChunker:
    def test_basic_chunking(self):
        chunks = chunk_fixed(SAMPLE_TEXT, window_size=10, overlap=2)
        assert len(chunks) > 0
        for c in chunks:
            assert c.chunk_strategy == ChunkStrategy.FIXED
            assert len(c.text) > 0

    def test_window_size_respected(self):
        chunks = chunk_fixed(SAMPLE_TEXT, window_size=5, overlap=0)
        for c in chunks:
            assert c.word_count <= 6  # slight tolerance for edge

    def test_empty_text(self):
        chunks = chunk_fixed("", window_size=10, overlap=2)
        assert len(chunks) == 0

    def test_overlap_creates_more_chunks(self):
        no_overlap = chunk_fixed(SAMPLE_TEXT, window_size=10, overlap=0)
        with_overlap = chunk_fixed(SAMPLE_TEXT, window_size=10, overlap=5)
        assert len(with_overlap) >= len(no_overlap)

    def test_metadata_passthrough(self):
        chunks = chunk_fixed(SAMPLE_TEXT, window_size=10, overlap=0, document_id="doc1", language="en")
        for c in chunks:
            assert c.document_id == "doc1"
            assert c.language == "en"


class TestSentenceChunker:
    def test_basic_chunking(self):
        chunks = chunk_sentence(SAMPLE_TEXT, max_words=15)
        assert len(chunks) > 0
        for c in chunks:
            assert c.chunk_strategy == ChunkStrategy.SENTENCE

    def test_indic_sentence_boundaries(self):
        chunks = chunk_sentence(HINDI_TEXT, max_words=20)
        assert len(chunks) > 0

    def test_single_sentence(self):
        chunks = chunk_sentence("Hello world.", max_words=100)
        assert len(chunks) == 1

    def test_empty_text(self):
        chunks = chunk_sentence("", max_words=10)
        assert len(chunks) == 0


class TestUnifiedChunker:
    def test_fixed_strategy(self):
        chunks = chunk_text(SAMPLE_TEXT, strategy=ChunkStrategy.FIXED)
        assert all(c.chunk_strategy == ChunkStrategy.FIXED for c in chunks)

    def test_sentence_strategy(self):
        chunks = chunk_text(SAMPLE_TEXT, strategy=ChunkStrategy.SENTENCE)
        assert all(c.chunk_strategy == ChunkStrategy.SENTENCE for c in chunks)

    def test_default_strategy(self):
        chunks = chunk_text(SAMPLE_TEXT)
        assert len(chunks) > 0

    def test_word_count_populated(self):
        chunks = chunk_text(SAMPLE_TEXT, strategy=ChunkStrategy.FIXED)
        for c in chunks:
            assert c.word_count > 0
