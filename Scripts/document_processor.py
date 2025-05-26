import os
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
import pandas as pd
import PyPDF2
from docx import Document
from pptx import Presentation
from bs4 import BeautifulSoup
import yaml
import json
import markdown2
import xmltodict
from PyRTF3.document import Document as RtfDocument
import pyarrow.parquet as pq
import sqlparse
import logging
import concurrent.futures
from tqdm import tqdm
import pytesseract
from PIL import Image
import pdf2image
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import ftfy
import hashlib
import tempfile
import shutil
from langdetect import detect
import re
import unicodedata
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for document processing"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    remove_urls: bool = True
    remove_emails: bool = True
    fix_unicode: bool = True
    normalize_whitespace: bool = True
    fix_punctuation: bool = True
    remove_brackets: bool = False
    fix_accents: bool = True
    fix_line_breaks: bool = True
    ocr_enabled: bool = True
    extract_images: bool = True
    dedup_threshold: float = 0.85
    language_detection: bool = True

class DocumentProcessor:
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.chunk_size = self.config.chunk_size
        self.chunk_overlap = self.config.chunk_overlap
        self.supported_extensions = {
            ".txt": self._process_text,
            ".pdf": self._process_pdf,
            ".docx": self._process_docx,
            ".pptx": self._process_pptx,
            ".xlsx": self._process_excel,
            ".csv": self._process_csv,
            ".parquet": self._process_parquet,
            ".sql": self._process_sql,
            ".json": self._process_json,
            ".yml": self._process_yaml,
            ".yaml": self._process_yaml,
            ".md": self._process_markdown,
            ".xml": self._process_xml,
            ".html": self._process_html,
            ".rtf": self._process_rtf,
            ".jpg": self._process_image,
            ".jpeg": self._process_image,
            ".png": self._process_image,
            ".tiff": self._process_image,
            ".bmp": self._process_image
        }
        
        # Initialize image processing if OCR is enabled
        if self.config.ocr_enabled:
            try:
                pytesseract.get_tesseract_version()
            except:
                logger.warning("Tesseract not found. OCR functionality will be disabled.")
                self.config.ocr_enabled = False

    def process_document(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Process a single document and return chunks with metadata."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = file_path.suffix.lower()
        if extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {extension}")

        try:
            # Process the document
            processor = self.supported_extensions[extension]
            content = processor(file_path)
            
            # Clean the text
            content = self._clean_text(content)
            
            # Extract metadata
            base_metadata = self._extract_metadata(content, file_path)
            
            # Split into chunks
            chunks = self._chunk_text(content)
            
            # Create chunk documents with metadata
            chunk_docs = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_hash': hashlib.md5(chunk.encode()).hexdigest()
                })
                chunk_docs.append({
                    'text': chunk,
                    'metadata': chunk_metadata
                })
            
            # Deduplicate chunks if enabled
            if len(chunk_docs) > 1:
                chunk_docs = self._deduplicate_chunks(chunk_docs)
            
            return chunk_docs
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            raise

    def process_directory(self, directory_path: Union[str, Path], max_workers: int = 4) -> List[Dict[str, Any]]:
        """Process all supported documents in a directory using parallel processing."""
        directory_path = Path(directory_path)
        if not directory_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory_path}")

        files_to_process = []
        for ext in self.supported_extensions.keys():
            files_to_process.extend(directory_path.glob(f"**/*{ext}"))

        all_chunks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self.process_document, file_path): file_path 
                            for file_path in files_to_process}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_file), 
                             total=len(files_to_process),
                             desc="Processing documents"):
                file_path = future_to_file[future]
                try:
                    chunks = future.result()
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")

        return all_chunks

    def _clean_text(self, text: str) -> str:
        """Apply comprehensive text cleaning"""
        if not text:
            return ""

        # Fix unicode issues
        if self.config.fix_unicode:
            text = ftfy.fix_text(text)
            text = unicodedata.normalize('NFKC', text)

        # Remove URLs
        if self.config.remove_urls:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)

        # Remove emails
        if self.config.remove_emails:
            text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', ' ', text)

        # Normalize whitespace
        if self.config.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()

        # Fix punctuation
        if self.config.fix_punctuation:
            text = re.sub(r'["""]', '"', text)
            text = re.sub(r'[''']', "'", text)
            text = re.sub(r'[‒–—―]', '-', text)

        # Fix line breaks
        if self.config.fix_line_breaks:
            text = re.sub(r'\r\n?|\n', '\n', text)
            text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        return text.strip()

    def _extract_metadata(self, text: str, file_path: Path) -> Dict[str, Any]:
        """Extract comprehensive metadata from text"""
        metadata = {
            'source': str(file_path),
            'file_type': file_path.suffix.lower(),
            'file_size': file_path.stat().st_size,
            'created_at': file_path.stat().st_ctime,
            'modified_at': file_path.stat().st_mtime,
            'hash': hashlib.md5(text.encode()).hexdigest(),
            'char_count': len(text),
            'word_count': len(text.split()),
            'file_name': file_path.name
        }

        # Detect language if enabled
        if self.config.language_detection and text.strip():
            try:
                metadata['language'] = detect(text)
            except:
                metadata['language'] = 'unknown'

        return metadata

    def _extract_images_from_pdf(self, pdf_path: Path) -> List[str]:
        """Extract and OCR images from PDF"""
        if not self.config.extract_images or not self.config.ocr_enabled:
            return []

        texts = []
        try:
            images = pdf2image.convert_from_path(str(pdf_path))
            for img in images:
                text = pytesseract.image_to_string(img)
                if text.strip():
                    texts.append(self._clean_text(text))
        except Exception as e:
            logger.warning(f"Error extracting images from PDF {pdf_path}: {str(e)}")

        return texts

    def _process_image(self, file_path: Path) -> str:
        """Process image files with OCR"""
        if not self.config.ocr_enabled:
            return ""

        try:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)
            return self._clean_text(text)
        except Exception as e:
            logger.warning(f"Error processing image {file_path}: {str(e)}")
            return ""

    def _deduplicate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove near-duplicate chunks using TF-IDF similarity"""
        if len(chunks) <= 1:
            return chunks

        texts = [chunk['text'] for chunk in chunks]
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()

        unique_chunks = []
        seen_indices = set()

        for i in range(len(chunks)):
            if i in seen_indices:
                continue

            unique_chunks.append(chunks[i])
            seen_indices.add(i)

            # Mark similar chunks as seen
            for j in range(i + 1, len(chunks)):
                if j not in seen_indices and similarity_matrix[i][j] > self.config.dedup_threshold:
                    seen_indices.add(j)

        return unique_chunks

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap while preserving semantic boundaries"""
        if not text:
            return []

        # Split into sentences first
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length

            # Handle very long sentences
            if sentence_length > self.chunk_size:
                words = sentence.split()
                current_chunk = []
                current_length = 0
                temp_chunk = []
                
                for word in words:
                    if current_length + len(word) <= self.chunk_size:
                        temp_chunk.append(word)
                        current_length += len(word) + 1
                    else:
                        if temp_chunk:
                            chunks.append(' '.join(temp_chunk))
                        temp_chunk = [word]
                        current_length = len(word)
                
                if temp_chunk:
                    current_chunk = temp_chunk
                    current_length = sum(len(word) + 1 for word in temp_chunk)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        # Add overlap between chunks
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped_chunks = []
            for i in range(len(chunks)):
                if i > 0:
                    words = chunks[i-1].split()
                    overlap_words = words[-self.chunk_overlap:]
                    chunks[i] = ' '.join(overlap_words + chunks[i].split())
                overlapped_chunks.append(chunks[i])
            chunks = overlapped_chunks

        return chunks

    def _process_text(self, file_path: Path) -> str:
        """Process plain text files."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _process_pdf(self, file_path: Path) -> str:
        """Process PDF files with OCR support for scanned documents"""
        text_content = []
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from each page
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(text)
                
                # If text extraction yielded little content, try OCR
                if self.config.ocr_enabled and len(''.join(text_content)) < 100:
                    logger.info(f"Attempting OCR on {file_path}")
                    ocr_texts = self._extract_images_from_pdf(file_path)
                    text_content.extend(ocr_texts)
                
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            raise
            
        return '\n\n'.join(text_content)

    def _ocr_pdf(self, file_path: Path) -> str:
        """Perform OCR on PDF file and return extracted text."""
        try:
            # Convert PDF to images
            images = pdf2image.convert_from_path(file_path)
            ocr_text = ""
            
            for image in images:
                # Perform OCR on each image
                text = pytesseract.image_to_string(image, lang='eng')
                ocr_text += text + "\n"
            
            return ocr_text
        except Exception as e:
            logger.error(f"Error during OCR processing: {e}")
            return ""

    def _process_docx(self, file_path: Path) -> str:
        """Process Word documents."""
        doc = Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])

    def _process_pptx(self, file_path: Path) -> str:
        """Process PowerPoint presentations."""
        prs = Presentation(file_path)
        text = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text.append(shape.text)
        return "\n".join(text)

    def _process_excel(self, file_path: Path) -> str:
        """Process Excel files."""
        df = pd.read_excel(file_path)
        return df.to_string()

    def _process_csv(self, file_path: Path) -> str:
        """Process CSV files."""
        df = pd.read_csv(file_path)
        return df.to_string()

    def _process_json(self, file_path: Path) -> str:
        """Process JSON files."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, indent=2)

    def _process_yaml(self, file_path: Path) -> str:
        """Process YAML files."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return yaml.dump(data)

    def _process_markdown(self, file_path: Path) -> str:
        """Process Markdown files."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return markdown2.markdown(text)

    def _process_xml(self, file_path: Path) -> str:
        """Process XML files."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = xmltodict.parse(f.read())
        return json.dumps(data, indent=2)

    def _process_html(self, file_path: Path) -> str:
        """Process HTML files."""
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "lxml")
        return soup.get_text()

    def _process_rtf(self, file_path: Path) -> str:
        """Process RTF files."""
        with open(file_path, "r", encoding="utf-8") as f:
            doc = RtfDocument(f)
        return str(doc)
        
    def _process_parquet(self, file_path: Path) -> str:
        """Process Parquet files."""
        try:
            # Read Parquet file
            table = pq.read_table(file_path)
            df = table.to_pandas()
            
            # Convert to string representation
            return df.to_string(index=True, max_rows=None, max_cols=None)
        except Exception as e:
            logger.error(f"Error processing Parquet file: {e}")
            raise

    def _process_sql(self, file_path: Path) -> str:
        """Process SQL files.
        
        This function:
        1. Reads SQL file content
        2. Formats and normalizes SQL queries
        3. Extracts comments and query structure
        4. Returns a formatted string with all SQL content
        """
        try:
            # Read SQL file
            with open(file_path, "r", encoding="utf-8") as f:
                sql_content = f.read()

            # Parse SQL content
            parsed = sqlparse.parse(sql_content)
            
            formatted_content = []
            
            for statement in parsed:
                # Format the statement
                formatted_sql = sqlparse.format(
                    str(statement),
                    keyword_case="upper",
                    identifier_case="lower",
                    reindent=True,
                    indent_width=4,
                    strip_comments=False
                )
                
                formatted_content.append(formatted_sql)
                
                # Extract and add comments separately
                comments = [token.value for token in statement.tokens if token.ttype is sqlparse.tokens.Comment]
                if comments:
                    formatted_content.extend(comments)
            
            return "\n\n".join(formatted_content)
            
        except Exception as e:
            logger.error(f"Error processing SQL file: {e}")
            raise
