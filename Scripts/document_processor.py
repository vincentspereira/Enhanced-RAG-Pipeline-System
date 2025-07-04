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
# logging.basicConfig(level=logging.INFO) # Potential reconfig if other modules set it up
logger = logging.getLogger(__name__)

# Attempt to import format handlers; fail gracefully if dependencies are missing for some
try:
    from . import format_handlers
except ImportError as e:
    logger.warning(f"Could not import format_handlers, some specific format processing might be unavailable: {e}")
    format_handlers = None # Ensure it exists even if import fails

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
            ".bmp": self._process_image,
            ".webp": self._process_image_webp_svg_ocr,
            ".svg": self._process_image_webp_svg_ocr,
        }
        
        # Add handlers from format_handlers.py if module was imported
        if format_handlers:
            self.supported_extensions.update({
                ".epub": format_handlers.process_epub,
                ".mobi": format_handlers.process_mobi, # Will gracefully fail if pandoc not present
                ".tex": format_handlers.process_latex,   # Will gracefully fail if pandoc not present
                ".rst": format_handlers.process_rst,    # Will gracefully fail if pandoc not present
                ".adoc": format_handlers.process_asciidoc, # Will gracefully fail if pandoc not present
                ".asciidoc": format_handlers.process_asciidoc,
                ".ipynb": format_handlers.process_ipynb,
                # Archives - these will be handled specially to trigger recursive processing
                ".zip": self._process_archive,
                ".tar": self._process_archive,
                ".gz": self._process_archive, # Often .tar.gz
                ".bz2": self._process_archive, # Often .tar.bz2
                ".xz": self._process_archive, # Often .tar.xz
                # Placeholders for other complex formats
                ".odp": format_handlers.process_odp,
                ".key": format_handlers.process_key,
                ".ods": format_handlers.process_ods,
                ".numbers": format_handlers.process_numbers,
                # Confluence might be .html or .xml, or a specific export type
                # For now, let's assume a .confluence_export extension for placeholder
                ".confluence_export": format_handlers.process_confluence_export,
                ".eml": format_handlers.process_eml,
                ".msg": format_handlers.process_msg,
                # CAD: .dxf, .dwg etc. - using a generic placeholder for now
                ".dxf": format_handlers.process_cad_metadata, # Example CAD extension
                # Audio/Video
                ".mp4": self._process_audio_video_placeholder,
                ".avi": self._process_audio_video_placeholder,
                ".mov": self._process_audio_video_placeholder,
                ".mp3": self._process_audio_video_placeholder,
                ".wav": self._process_audio_video_placeholder,
                ".flac": self._process_audio_video_placeholder,
            })

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
        # Basic file validation (e.g., non-empty)
        if file_path.stat().st_size == 0:
            logger.warning(f"File {file_path} is empty. Skipping.")
            return []

        # Placeholder for more advanced content validation (corruption/malicious)
        if self._is_file_potentially_corrupted_or_malicious(file_path):
             logger.warning(f"File {file_path} skipped due to potential corruption or malicious content flag.")
             return []

        if extension not in self.supported_extensions:
            # Try to detect type using python-magic if available
            try:
                import magic
                detected_mime = magic.from_file(str(file_path), mime=True)
                logger.info(f"File {file_path} has extension '{extension}', detected MIME: {detected_mime}")
                # TODO: Add logic to map detected_mime to a handler if different from extension
            except ImportError:
                logger.debug("python-magic not installed, relying on extension.")
            except Exception as e:
                logger.warning(f"python-magic failed for {file_path}: {e}")

            # If still not found after magic check (or magic not available)
            if extension not in self.supported_extensions:
                 logger.warning(f"Unsupported file type based on extension: {extension} for file {file_path}. Skipping.")
                 return []


        all_extracted_texts = []

        # Handle archives by extracting and then processing their contents
        if extension in [".zip", ".tar", ".gz", ".bz2", ".xz"]: # .rar, .7z would need their libraries
            if format_handlers: # Ensure format_handlers module is available
                temp_dir_path_obj = self._create_temp_dir_for_extraction()
                try:
                    extracted_files = format_handlers.process_archive(str(file_path), lambda: str(temp_dir_path_obj))
                    logger.info(f"Archive {file_path}: Extracted {len(extracted_files)} files to {temp_dir_path_obj}.")
                    for extracted_file_str_path in extracted_files:
                        # Recursively process extracted files
                        # Note: This recursive call creates new chunks lists.
                        # We are collecting text content here to be processed as one "document" from the archive source.
                        # Alternatively, each file in archive could be its own set of chunks.
                        # For now, let's aggregate text from supported files within the archive.
                        extracted_file_path = Path(extracted_file_str_path)
                        sub_ext = extracted_file_path.suffix.lower()
                        if sub_ext in self.supported_extensions and sub_ext not in [".zip", ".tar", ".gz", ".bz2", ".xz"]: # Avoid re-processing archives this way
                             try:
                                processor = self.supported_extensions[sub_ext]
                                text_content = processor(extracted_file_path)
                                if text_content and isinstance(text_content, str):
                                    all_extracted_texts.append(text_content)
                             except Exception as e_sub:
                                logger.error(f"Error processing extracted file {extracted_file_path} from archive {file_path}: {e_sub}")
                        elif sub_ext in [".zip", ".tar", ".gz", ".bz2", ".xz"]:
                             logger.info(f"Found nested archive {extracted_file_path}, recursively processing its text content.")
                             # This will return a list of chunk dicts, we need raw text here for aggregation.
                             # Simpler approach: Process nested archive and get its text.
                             # This requires process_document to return raw text if called in a specific mode,
                             # or handle chunk dictionaries appropriately.
                             # For now, let's assume _process_archive itself should handle recursion internally
                             # if it wants to aggregate all text. The current format_handlers.process_archive
                             # returns file paths, so this structure is for processing those paths.
                             # To aggregate all text from an archive (including nested ones) into one "document":
                             nested_archive_texts = self.process_document_to_text(extracted_file_path) # New helper needed
                             if nested_archive_texts:
                                all_extracted_texts.append(nested_archive_texts)

                finally:
                    self._cleanup_temp_dir(temp_dir_path_obj)

                content = "\n\n--- File Separator ---\n\n".join(all_extracted_texts)
            else:
                logger.warning(f"format_handlers module not available, cannot process archive {file_path}")
                return [] # Or raise error
        else:
            # Process single document
            try:
                processor = self.supported_extensions[extension]
                content = processor(file_path) # This should return a string
            except Exception as e:
                 logger.error(f"Error processing {file_path} with handler for {extension}: {e}")
                 raise # Re-raise to be caught by the outer try-except

        if not isinstance(content, str):
            logger.error(f"Processor for {extension} did not return a string for {file_path}. Got {type(content)}. Skipping.")
            return []
            
        # Clean the text
        cleaned_content = self._clean_text(content)
        if not cleaned_content.strip():
            logger.info(f"No content extracted or content became empty after cleaning for {file_path}.")
            return []
            
            # Extract metadata
            base_metadata = self._extract_metadata(cleaned_content, file_path) # Use cleaned_content for metadata
            
            # Split into chunks
            chunks = self._chunk_text(cleaned_content) # Use cleaned_content for chunking
            
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
            logger.error(f"Error during OCR processing for {file_path}: {e}")
            return ""

    def _is_file_potentially_corrupted_or_malicious(self, file_path: Path) -> bool:
        """
        Basic placeholder for content validation.
        Checks for zero-byte files or excessively large files as simple heuristics.
        A real implementation would involve more sophisticated checks, possibly
        integrating with tools like ClamAV for malware or format-specific validators.
        """
        try:
            size = file_path.stat().st_size
            if size == 0:
                logger.warning(f"File {file_path} is zero bytes (potentially corrupted or empty).")
                return True # Considered problematic

            # Example: Flag files larger than 1GB as potentially problematic for auto-processing
            # This limit should be configurable.
            MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024 * 1024 # 1 GB
            if size > MAX_FILE_SIZE_BYTES:
                logger.warning(f"File {file_path} is very large ({size} bytes). May be corrupted or unsuitable for typical processing.")
                return True # Flagging large files

            # Placeholder for actual malware scanning integration
            # if format_handlers and hasattr(format_handlers, 'scan_for_malware'):
            #     if format_handlers.scan_for_malware(str(file_path)):
            #         logger.warning(f"Malware scan placeholder: flagged {file_path}.")
            #         return True

            # Add more checks here, e.g., magic number validation against extension
            # import magic
            # try:
            #     mime_type = magic.from_file(str(file_path), mime=True)
            #     # Compare mime_type with expected based on extension
            # except Exception:
            #     pass # magic might not be available or fail

        except Exception as e:
            logger.error(f"Error during file validation for {file_path}: {e}")
            return True # Treat as problematic if validation fails

        return False # Default to not corrupted/malicious

    def _create_temp_dir_for_extraction(self) -> Path:
        """Creates a temporary directory for archive extraction."""
        # Use a main temporary directory for all extractions from this processor instance
        # to simplify cleanup if the processor is long-lived.
        # Or create a new one each time if preferred.
        if not hasattr(self, '_main_temp_dir') or not self._main_temp_dir.exists():
             # Create a general temp directory for this DocumentProcessor instance
            self._main_temp_dir = Path(tempfile.mkdtemp(prefix="docproc_"))
            logger.info(f"Created main temporary directory for extractions: {self._main_temp_dir}")

        # Create a unique subdirectory for each archive extraction call
        extraction_path = Path(tempfile.mkdtemp(dir=self._main_temp_dir))
        return extraction_path

    def _cleanup_temp_dir(self, temp_dir_path: Optional[Path] = None):
        """Cleans up a specific temporary directory, or the main one if no path is given."""
        if temp_dir_path and temp_dir_path.exists():
            try:
                shutil.rmtree(temp_dir_path)
                logger.debug(f"Cleaned up temporary directory: {temp_dir_path}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary directory {temp_dir_path}: {e}")
        elif hasattr(self, '_main_temp_dir') and self._main_temp_dir.exists():
            try:
                shutil.rmtree(self._main_temp_dir)
                logger.info(f"Cleaned up main temporary extraction directory: {self._main_temp_dir}")
                delattr(self, '_main_temp_dir')
            except Exception as e:
                logger.error(f"Error cleaning up main temporary directory {self._main_temp_dir}: {e}")

    def __del__(self):
        """Ensure main temporary directory is cleaned up when the processor is deleted."""
        self._cleanup_temp_dir()


    def process_document_to_text(self, file_path: Path) -> str:
        """
        Helper function to process a single document (including nested archives)
        and return all its text content as a single string.
        Used by the archive processing logic.
        """
        logger.debug(f"Recursively processing {file_path} to get its text content.")
        extension = file_path.suffix.lower()
        all_texts = []

        if extension in [".zip", ".tar", ".gz", ".bz2", ".xz"]:
            if format_handlers:
                temp_dir_path_obj = self._create_temp_dir_for_extraction()
                try:
                    extracted_files = format_handlers.process_archive(str(file_path), lambda: str(temp_dir_path_obj))
                    for extracted_file_str_path in extracted_files:
                        extracted_file_path_obj = Path(extracted_file_str_path)
                        # Recursive call to get text from these files
                        all_texts.append(self.process_document_to_text(extracted_file_path_obj))
                finally:
                    self._cleanup_temp_dir(temp_dir_path_obj) # Clean up specific temp dir
            else:
                logger.warning(f"format_handlers not available, cannot extract text from archive {file_path}")
        elif extension in self.supported_extensions:
            try:
                processor = self.supported_extensions[extension]
                text = processor(file_path)
                if text and isinstance(text, str):
                    all_texts.append(text)
            except Exception as e:
                logger.error(f"Error processing sub-file {file_path} for text: {e}")
        else:
            logger.warning(f"Unsupported file type {extension} in process_document_to_text for {file_path}")

        return "\n\n--- File Separator ---\n\n".join(filter(None, all_texts))


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
        """Process Excel files, concatenating text from all sheets."""
        try:
            xls = pd.ExcelFile(file_path)
            all_text = []
            for sheet_name in xls.sheet_names:
                df = xls.parse(sheet_name)
                # Convert entire sheet to string, trying to preserve some structure
                sheet_text = f"Sheet: {sheet_name}\n{df.to_string(index=True, na_rep='NaN')}\n\n"
                all_text.append(sheet_text)
            return "".join(all_text)
        except Exception as e:
            logger.error(f"Error processing Excel file {file_path}: {e}")
            return "" # Return empty string on error

    def _process_csv(self, file_path: Path) -> str:
        """Process CSV files."""
        # TODO: Add encoding detection for robust CSV handling
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='latin1') # Common fallback
            except Exception as e:
                logger.error(f"Error processing CSV {file_path} even with latin1 encoding: {e}")
                return ""
        except Exception as e:
            logger.error(f"Error processing CSV {file_path}: {e}")
            return ""
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
                logger.error(f"Error processing Parquet file {file_path}: {e}")
                return "" # Return empty string on error

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
                
                # Extract and add comments separately if not already part of formatted_sql by default
                # (sqlparse.format with strip_comments=False should keep them)
                # comments = [token.value for token in statement.tokens if token.ttype is sqlparse.tokens.Comment]
                # if comments:
                #    formatted_content.extend(comments)
            
            return "\n\n".join(formatted_content)
            
        except Exception as e:
            logger.error(f"Error processing SQL file {file_path}: {e}")
            return "" # Return empty string on error


    # --- Placeholder methods for new formats / features ---
    def _process_image_webp_svg_ocr(self, file_path: Path) -> str:
        """Placeholder handler for WEBP/SVG OCR, delegates to format_handlers if available."""
        if format_handlers and hasattr(format_handlers, 'process_webp_svg_ocr'):
            return format_handlers.process_webp_svg_ocr(str(file_path))
        logger.warning(f"WEBP/SVG OCR handler not fully available for {file_path}.")
        return self._process_image(file_path) # Fallback to generic image OCR if Pillow can open it

    def _process_audio_video_placeholder(self, file_path: Path) -> str:
        """Placeholder for audio/video processing."""
        if format_handlers and hasattr(format_handlers, 'transcribe_audio_video'):
            text = format_handlers.transcribe_audio_video(str(file_path))
            # Placeholder for speaker identification - would append to text or add to metadata
            # speaker_info = format_handlers.identify_speakers(str(file_path))
            # text += f"\n\n{speaker_info}"
            return text
        logger.warning(f"Audio/Video processing handler not available for {file_path}.")
        return f"Audio/Video content from {file_path.name} (Placeholder)"

    def _process_archive(self, file_path: Path) -> str:
        """
        Wrapper for archive processing. This method is called by the dispatcher.
        It uses process_document_to_text to get all text from an archive.
        """
        logger.info(f"Starting processing of archive: {file_path}")
        # process_document_to_text will handle the extraction and recursive processing.
        # It returns a single string of all aggregated text.
        return self.process_document_to_text(file_path)

    # --- Future PDF enhancements (placeholders in format_handlers) ---
    def _enhance_pdf_processing(self, file_path: Path, existing_text: str) -> str:
        """
        Placeholder to demonstrate where table and chart extraction for PDFs would be added.
        This would likely modify the metadata or append structured data.
        For now, it just returns the existing text.
        """
        all_content = [existing_text]
        if format_handlers:
            if hasattr(format_handlers, 'extract_tables_from_pdf'):
                table_data = format_handlers.extract_tables_from_pdf(str(file_path))
                if table_data:
                    all_content.append("\n\n--- Extracted Tables ---\n" + table_data)
            if hasattr(format_handlers, 'recognize_charts_from_pdf'):
                chart_data = format_handlers.recognize_charts_from_pdf(str(file_path))
                if chart_data:
                    all_content.append("\n\n--- Extracted Chart Data ---\n" + chart_data)
        return "\n".join(all_content)

    # In _process_pdf, after getting text_content, you could call:
    # full_pdf_text = "\n\n".join(text_content)
    # content_with_extras = self._enhance_pdf_processing(file_path, full_pdf_text)
    # return content_with_extras
    # (This change is not made yet to keep the PR smaller, but indicates intent)
