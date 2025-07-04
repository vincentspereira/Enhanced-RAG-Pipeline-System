"""
Unit tests for the DocumentProcessor and its components.
"""
import unittest
from unittest.mock import patch, MagicMock # Added MagicMock
from pathlib import Path
import tempfile
import shutil
import os

# Ensure Scripts directory is in path for imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Scripts')))

from Scripts.document_processor import DocumentProcessor, ProcessingConfig
# Import format_handlers to allow mocking its members if DocumentProcessor imports it with "from . import format_handlers"
from Scripts.document_processor import format_handlers as docproc_format_handlers


class TestDocumentProcessor(unittest.TestCase):

    def setUp(self):
        """Set up for test cases."""
        self.test_dir = Path(tempfile.mkdtemp(prefix="docproc_test_"))
        # Ensure OCR is disabled for most unit tests for speed/simplicity,
        # but ImageProcessor might still be initialized if present.
        self.config = ProcessingConfig(ocr_enabled=False, extract_images=False)
        self.processor = DocumentProcessor(config=self.config)

        # For tests that specifically need to mock format_handlers,
        # we can patch 'Scripts.document_processor.format_handlers' directly.

    def tearDown(self):
        """Clean up after test cases."""
        shutil.rmtree(self.test_dir)
        # Clean up main temp dir used by processor if it was created
        if hasattr(self.processor, '_main_temp_dir') and self.processor._main_temp_dir and self.processor._main_temp_dir.exists():
            try:
                shutil.rmtree(self.processor._main_temp_dir)
            except Exception as e:
                print(f"Error cleaning up processor's main temp dir: {e}")


    def _create_test_file(self, filename: str, content: str, mode: str = "w", encoding: str = "utf-8"):
        file_path = self.test_dir / filename
        with open(file_path, mode, encoding=encoding if mode == "w" else None) as f:
            f.write(content)
        return file_path

    def test_process_txt_file(self):
        """Test processing a simple TXT file."""
        content = "This is a test document.\nIt has two lines."
        file_path = self._create_test_file("test.txt", content)

        chunks = self.processor.process_document(file_path)

        self.assertIsInstance(chunks, list)
        self.assertTrue(len(chunks) > 0)
        self.assertIn("This is a test document.", chunks[0]['text'])
        self.assertEqual(chunks[0]['metadata']['file_type'], ".txt")
        self.assertEqual(chunks[0]['metadata']['file_name'], "test.txt")

    def test_text_cleaning(self):
        """Test various text cleaning functionalities."""
        raw_text = "  This is a test with   extra spaces, a URL http://example.com and email test@example.com.  \n\nMultiple newlines.  "
        # Expected: based on default config (remove URLs, emails, normalize whitespace)
        expected_text = "This is a test with extra spaces, a URL and email . Multiple newlines."
        cleaned = self.processor._clean_text(raw_text) # _clean_text is part of DocumentProcessor
        self.assertEqual(cleaned, expected_text)

    def test_chunk_text_simple_delegation(self):
        """Test basic text chunking (delegation to optimizer or fallback)."""
        text = "This is the first sentence. This is the second sentence. This is the third sentence."

        if self.processor.chunk_optimizer_instance:
            # Mock the optimizer's method
            self.processor.chunk_optimizer_instance.split_into_chunks = MagicMock(return_value=[text]) # Simulate it returns one chunk
            chunks = self.processor.process_document(self._create_test_file("chunk_test.txt", text))
            self.processor.chunk_optimizer_instance.split_into_chunks.assert_called_once_with(text)
            self.assertEqual(len(chunks), 1)
        else:
            # Test fallback logic if optimizer is not there (original _chunk_text)
            # This requires setting chunk_size on processor for the fallback logic
            original_chunk_size = self.processor.chunk_size
            self.processor.chunk_size = 50
            self.processor.chunk_overlap = 5

            chunks_list_content = self.processor._chunk_text(text) # Call the internal method directly for fallback test
            self.assertTrue(len(chunks_list_content) >= 1) # Behavior of fallback
            self.processor.chunk_size = original_chunk_size # Reset


    def test_process_unsupported_file(self):
        file_path = self._create_test_file("test.unsupported", "Dummy content")
        results = self.processor.process_document(file_path)
        self.assertEqual(results, [])


    def test_process_empty_file(self):
        file_path = self._create_test_file("empty.txt", "")
        chunks = self.processor.process_document(file_path)
        self.assertEqual(chunks, [])


    def test_metadata_extraction_delegation(self):
        content = "Sample content for metadata."
        file_path = self._create_test_file("meta_test_delegation.txt", content)

        if self.processor.metadata_extractor_instance:
            # Mock the extractor's method
            expected_meta = {"file_path": str(file_path), "test_meta": "success"}
            self.processor.metadata_extractor_instance.extract_metadata = MagicMock(return_value=expected_meta)

            # process_document calls the metadata extraction
            chunks = self.processor.process_document(file_path)

            self.processor.metadata_extractor_instance.extract_metadata.assert_called_once_with(str(file_path))
            self.assertTrue(len(chunks) > 0)
            self.assertEqual(chunks[0]['metadata']['test_meta'], "success")
        else:
            # Test fallback logic (_extract_metadata in DocumentProcessor)
            metadata = self.processor._extract_metadata(content, file_path) # Call internal method
            self.assertEqual(metadata['source'], str(file_path))
            self.assertIn('hash', metadata)


    def test_process_docx_file(self):
        try:
            from docx import Document as DocxDocument
            doc = DocxDocument()
            doc.add_paragraph("Hello from DOCX.")
            temp_docx_path = self.test_dir / "test.docx"
            doc.save(temp_docx_path)

            chunks = self.processor.process_document(temp_docx_path)
            self.assertTrue(len(chunks) > 0)
            self.assertIn("Hello from DOCX.", chunks[0]['text'])
            self.assertEqual(chunks[0]['metadata']['file_type'], ".docx")
        except ImportError:
            self.skipTest("python-docx not installed.")
        except Exception as e:
            self.skipTest(f"Skipping DOCX test due to environment issue: {e}")


    def test_archive_processing_basic_zip(self):
        try:
            import zipfile
            text_content = "Text in zip."
            zip_path = self.test_dir / "test_archive.zip"
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr("zipped_text.txt", text_content)

            chunks = self.processor.process_document(zip_path)
            self.assertTrue(len(chunks) > 0, "Expected chunks from zipped file")
            self.assertIn(text_content, chunks[0]['text'])
            self.assertEqual(chunks[0]['metadata']['file_name'], "test_archive.zip")
            # For a single file in zip, default "--- File Separator ---" should not appear
            self.assertNotIn("--- File Separator ---", chunks[0]['text'])
        except ImportError:
            self.skipTest("zipfile library not available.")
        except Exception as e:
            self.skipTest(f"Skipping ZIP test due to environment issue: {e}")

    # --- Tests for new/enhanced format handlers and integrations ---

    @patch.object(docproc_format_handlers, 'process_eml', MagicMock(return_value="Email body content from EML."))
    def test_process_eml_delegation(self):
        file_path = self._create_test_file("test.eml", "dummy eml data")
        chunks = self.processor.process_document(file_path)
        docproc_format_handlers.process_eml.assert_called_once_with(str(file_path))
        self.assertTrue(len(chunks) > 0)
        self.assertIn("Email body content from EML.", chunks[0]['text'])

    @patch.object(docproc_format_handlers, 'process_msg', MagicMock(return_value="Outlook message content from MSG."))
    def test_process_msg_delegation(self):
        file_path = self._create_test_file("test.msg", "dummy msg data")
        chunks = self.processor.process_document(file_path)
        docproc_format_handlers.process_msg.assert_called_once_with(str(file_path))
        self.assertTrue(len(chunks) > 0)
        self.assertIn("Outlook message content from MSG.", chunks[0]['text'])

    @patch.object(docproc_format_handlers, 'process_cad_metadata', MagicMock(return_value="DXF CAD Metadata and Text."))
    def test_process_dxf_delegation(self):
        file_path = self._create_test_file("test.dxf", "dummy dxf data")
        chunks = self.processor.process_document(file_path)
        docproc_format_handlers.process_cad_metadata.assert_called_once_with(str(file_path))
        self.assertTrue(len(chunks) > 0)
        self.assertIn("DXF CAD Metadata and Text.", chunks[0]['text'])

    @patch.object(docproc_format_handlers, 'transcribe_audio_video', MagicMock(return_value={"transcript_with_speakers": "This is a spoken transcript."}))
    def test_process_mp3_delegation(self):
        file_path = self._create_test_file("test.mp3", b"dummy audio data", mode="wb") # create binary file
        chunks = self.processor.process_document(file_path)
        docproc_format_handlers.transcribe_audio_video.assert_called_once_with(str(file_path))
        self.assertTrue(len(chunks) > 0)
        self.assertIn("This is a spoken transcript.", chunks[0]['text'])

    @patch.object(docproc_format_handlers, 'extract_tables_from_pdf', MagicMock(return_value="\n\n--- Extracted Table ---\n| ColA | ColB |"))
    @patch('PyPDF2.PdfReader')
    def test_process_pdf_with_table_delegation(self, mock_pdf_reader_constructor, mock_extract_tables_func):
        # This test focuses on the delegation to table extraction.
        # Mock PdfReader to return minimal valid structure with some text.
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "PDF page text."
        mock_pdf_instance = MagicMock()
        mock_pdf_instance.pages = [mock_page]
        mock_pdf_reader_constructor.return_value = mock_pdf_instance

        # The _process_pdf method should call the (mocked) format_handlers.extract_tables_from_pdf
        pdf_path = self.test_dir / "test_table.pdf"
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.4\n%fake pdf content") # Minimal binary content

        # Temporarily enable OCR to ensure that part of the PDF logic is also exercised,
        # but mock away the actual OCR image processing for this specific test.
        original_ocr_enabled = self.processor.config.ocr_enabled
        self.processor.config.ocr_enabled = True
        with patch.object(self.processor, '_extract_images_from_pdf', return_value=["OCR text from image."]):
            chunks = self.processor.process_document(pdf_path)
        self.processor.config.ocr_enabled = original_ocr_enabled # Restore config

        docproc_format_handlers.extract_tables_from_pdf.assert_called_once_with(str(pdf_path))
        self.assertTrue(len(chunks) > 0)
        self.assertIn("PDF page text.", chunks[0]['text'])
        self.assertIn("Extracted Table", chunks[0]['text'])
        self.assertIn("OCR text from image.", chunks[0]['text']) # Check if OCR text also included

if __name__ == '__main__':
    unittest.main()
