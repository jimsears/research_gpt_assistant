"""
Document Processing Module for ResearchGPT Assistant

TODO: Implement the following functionality:
1. PDF text extraction and cleaning
2. Text preprocessing and chunking
3. Basic similarity search using TF-IDF
4. Document metadata extraction
"""

import os
import re
import logging
from typing import List, Dict, Tuple

import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class DocumentProcessor:
    def __init__(self, config):
        """
        Initialize Document Processor
        
        TODO: 
        1. Store configuration
        2. Initialize TF-IDF vectorizer
        3. Create empty document storage
        """
        self.config = config

        # Initialize TfidfVectorizer with appropriate parameters
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=20000,
            ngram_range=(1, 2),
            lowercase=True,
            strip_accents="ascii",
        )

        # Create document storage structure
        self.documents: Dict[str, Dict] = {}
        self.document_vectors = None
        self._all_chunks: List[Dict] = []  # {doc_id, text, chunk_idx}

        # Logging setup
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)

    # TODO: Implement PDF text extraction using PyPDF2
    # 1. Open PDF file
    # 2. Extract text from all pages
    # 3. Clean extracted text (remove extra whitespace, special characters)
    # 4. Return cleaned text
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        extracted_text_parts: List[str] = []
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text() or ""
                    except Exception:
                        page_text = ""
                    extracted_text_parts.append(page_text)
        except FileNotFoundError:
            self.logger.error(f"PDF not found: {pdf_path}")
            return ""
        except Exception as e:
            self.logger.error(f"Error reading PDF {pdf_path}: {str(e)}")
            return ""

        raw = "\n".join(extracted_text_parts)
        cleaned = self.preprocess_text(raw)
        self.logger.info(f"Extracted {len(cleaned)} characters from {os.path.basename(pdf_path)}")
        return cleaned

    # TODO: Implement text preprocessing
    # 1. Remove extra whitespace and newlines
    # 2. Fix common PDF extraction issues
    # 3. Remove special characters if needed
    # 4. Ensure text is properly formatted
    def preprocess_text(self, text: str) -> str:
        if not text:
            return ""

        # Remove soft hyphen and ligatures
        text = text.replace("\u00ad", "")
        text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")

        # Normalize whitespace
        text = re.sub(r"[\t ]+", " ", text)
        text = re.sub(r"\s*\n\s*", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        return text

    # TODO: Implement text chunking
    # 1. Use config chunk_size and overlap if not provided
    # 2. Split text into overlapping chunks
    # 3. Ensure chunks don't break in middle of sentences
    # 4. Return list of text chunks
    def chunk_text(self, text: str, chunk_size: int | None = None, overlap: int | None = None) -> List[str]:
        if chunk_size is None:
            chunk_size = int(getattr(self.config, "CHUNK_SIZE", 1000))
        if overlap is None:
            overlap = int(getattr(self.config, "OVERLAP", 100))

        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z(0-9)])", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: List[str] = []
        cur_chunk: List[str] = []
        cur_len = 0

        def flush():
            if cur_chunk:
                chunks.append(" ".join(cur_chunk).strip())

        for s in sentences:
            s_len = len(s)
            if cur_len + s_len + 1 <= chunk_size:
                cur_chunk.append(s)
                cur_len += s_len + 1
            else:
                flush()
                if overlap > 0 and chunks:
                    tail = chunks[-1][-overlap:]
                    cur_chunk = [tail, s]
                    cur_len = len(tail) + len(s)
                else:
                    cur_chunk = [s]
                    cur_len = s_len
        flush()

        chunks = [c for c in chunks if len(c) > 50]
        return chunks

    # TODO: Implement complete document processing pipeline
    # 1. Extract text from PDF
    # 2. Preprocess the text
    # 3. Create chunks
    # 4. Extract basic metadata (title, length, etc.)
    # 5. Store in document storage
    # 6. Return document ID
    def process_document(self, pdf_path: str) -> str:
        doc_id = os.path.basename(pdf_path).replace(".pdf", "")
        text = self.extract_text_from_pdf(pdf_path)

        if not text:
            self.documents[doc_id] = {
                "title": doc_id,
                "chunks": [],
                "metadata": {"path": pdf_path, "chars": 0, "chunks": 0},
            }
            return doc_id

        chunks = self.chunk_text(text)
        self.documents[doc_id] = {
            "title": doc_id,
            "chunks": chunks,
            "metadata": {"path": pdf_path, "chars": len(text), "chunks": len(chunks)},
        }
        return doc_id

    def process_all_in_dir(self, dir_path: str) -> List[str]:
        if not os.path.exists(dir_path):
            return []
        doc_ids: List[str] = []
        for name in os.listdir(dir_path):
            if name.lower().endswith(".pdf"):
                pdf_path = os.path.join(dir_path, name)
                doc_ids.append(self.process_document(pdf_path))
        return doc_ids

    # TODO: Build TF-IDF index
    # 1. Collect all text chunks from all documents
    # 2. Fit TF-IDF vectorizer on all chunks
    # 3. Transform chunks to vectors
    # 4. Store vectors for similarity search
    def build_search_index(self) -> None:
        self._all_chunks = []
        for doc_id, data in self.documents.items():
            for i, chunk in enumerate(data.get("chunks", [])):
                self._all_chunks.append({"doc_id": doc_id, "text": chunk, "chunk_idx": i})

        if not self._all_chunks:
            self.document_vectors = None
            self.logger.warning("No chunks to index. Did any PDFs yield text?")
            return

        corpus = [c["text"] for c in self._all_chunks]
        try:
            self.document_vectors = self.vectorizer.fit_transform(corpus)
        except ValueError as e:
            self.logger.error(f"Vectorizer failed: {str(e)}")
            self.document_vectors = None

    # TODO: Implement similarity search
    # 1. Transform query using fitted TF-IDF vectorizer
    # 2. Calculate cosine similarity with all chunks
    # 3. Return top_k most similar chunks with scores
    def find_similar_chunks(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        if self.document_vectors is None or getattr(self.document_vectors, "shape", (0,))[0] == 0:
            return []

        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.document_vectors)[0]
        if sims.size == 0:
            return []

        top_idx = sims.argsort()[::-1][:top_k]
        results: List[Tuple[str, float, str]] = []
        for idx in top_idx:
            meta = self._all_chunks[idx]
            results.append((meta["text"], float(sims[idx]), meta["doc_id"]))
        return results

    # TODO: Return dictionary with: number of documents, total chunks, average length, titles
    def get_document_stats(self) -> Dict:
        num_docs = len(self.documents)
        titles = [v.get("title", "") for v in self.documents.values()]
        total_chunks = sum(v["metadata"].get("chunks", 0) for v in self.documents.values()) if self.documents else 0
        total_chars = sum(v["metadata"].get("chars", 0) for v in self.documents.values()) if self.documents else 0
        avg_len = int(total_chars / num_docs) if num_docs else 0
        return {
            "num_documents": num_docs,
            "total_chunks": total_chunks,
            "avg_chars_per_document": avg_len,
            "titles": titles,
        }

