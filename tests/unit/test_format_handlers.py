import unittest
from unittest.mock import patch, MagicMock
import zipfile
import io
import os

# Ensure Scripts directory is in path for imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Scripts')))

from Scripts.document_processor import format_handlers # Import the module to test

class TestFormatHandlers(unittest.TestCase):

    def _create_mock_zip_file_bytes(self, file_contents: dict) -> bytes:
        """Helper to create a mock zip file in memory (as bytes)."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for filename, content in file_contents.items():
                if isinstance(content, str):
                    content_bytes = content.encode('utf-8')
                else: # assume bytes
                    content_bytes = content
                zf.writestr(filename, content_bytes)
        return zip_buffer.getvalue()

    def test_process_confluence_export_zip_with_html(self):
        """Test processing a Confluence export ZIP with valid HTML files."""
        html_content1 = "<html><body><div id='main-content'>Page 1 Content here.</div></body></html>"
        html_content2 = "<html><body><div class='wiki-content'>Page 2 has <b>bold</b> text.</div></body></html>"
        attachment_content = "some binary data"

        mock_zip_bytes = self._create_mock_zip_file_bytes({
            "page1.html": html_content1,
            "pages/page2.html": html_content2,
            "attachments/image.png": attachment_content,
            "styles/main.css": "body {color: blue;}"
        })

        # Create a temporary file path for the test
        temp_dir = tempfile.TemporaryDirectory()
        mock_zip_path = os.path.join(temp_dir.name, "confluence_export.zip")
        with open(mock_zip_path, "wb") as f:
            f.write(mock_zip_bytes)

        extracted_text = format_handlers.process_confluence_export(mock_zip_path)

        self.assertIn("Page 1 Content here.", extracted_text)
        self.assertIn("Page 2 has bold text.", extracted_text)
        self.assertNotIn("some binary data", extracted_text) # Attachments should not be text processed
        self.assertNotIn("body {color: blue;}", extracted_text) # Styles should be ignored
        self.assertIn("--- Page: page1.html ---", extracted_text)
        self.assertIn("--- Page: pages/page2.html ---", extracted_text)

        temp_dir.cleanup()

    def test_process_confluence_export_zip_no_html(self):
        """Test Confluence ZIP with no processable HTML files."""
        mock_zip_bytes = self._create_mock_zip_file_bytes({
            "attachments/image.png": "binary",
            "styles/main.css": "css"
        })
        temp_dir = tempfile.TemporaryDirectory()
        mock_zip_path = os.path.join(temp_dir.name, "no_html.zip")
        with open(mock_zip_path, "wb") as f:
            f.write(mock_zip_bytes)

        extracted_text = format_handlers.process_confluence_export(mock_zip_path)
        self.assertEqual(extracted_text, "No processable HTML content found in Confluence ZIP.")
        temp_dir.cleanup()

    def test_process_confluence_export_not_a_zip(self):
        """Test Confluence handler with a non-ZIP file (should fallback to HTML parse)."""
        html_content = "<html><body>Simple HTML Page fallback content.</body></html>"
        temp_dir = tempfile.TemporaryDirectory()
        mock_html_path = os.path.join(temp_dir.name, "not_a_zip.html")
        with open(mock_html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        extracted_text = format_handlers.process_confluence_export(mock_html_path)
        self.assertIn("Simple HTML Page fallback content.", extracted_text)
        temp_dir.cleanup()

    def test_process_confluence_export_corrupted_zip(self):
        """Test Confluence handler with a corrupted ZIP file."""
        temp_dir = tempfile.TemporaryDirectory()
        mock_corrupted_zip_path = os.path.join(temp_dir.name, "corrupted.zip")
        with open(mock_corrupted_zip_path, "wb") as f:
            f.write(b"This is not a valid zip file content.")

        extracted_text = format_handlers.process_confluence_export(mock_corrupted_zip_path)
        self.assertIn("Error: Corrupted or invalid ZIP file", extracted_text)
        temp_dir.cleanup()

    # TODO: Add tests for other format handlers like EML, MSG, CAD once their implementations are stable.

    # --- Tests for Malware Scanning Mockup ---
    def test_scan_for_malware_mockup_safe_extensions(self):
        self.assertFalse(format_handlers.scan_for_malware("document.pdf"))
        self.assertFalse(format_handlers.scan_for_malware("image.png"))
        self.assertFalse(format_handlers.scan_for_malware("archive.zip_content.txt", within_archive=True)) # .txt is safe
        self.assertFalse(format_handlers.scan_for_malware("archive.zip_content.docx", within_archive=True))

    def test_scan_for_malware_mockup_risky_general(self):
        self.assertTrue(format_handlers.scan_for_malware("suspicious.scr"))
        self.assertTrue(format_handlers.scan_for_malware("another.pif", within_archive=False))
        self.assertTrue(format_handlers.scan_for_malware("command.com", within_archive=True))


    def test_scan_for_malware_mockup_risky_in_archive(self):
        self.assertTrue(format_handlers.scan_for_malware("myapp.exe", within_archive=True))
        self.assertTrue(format_handlers.scan_for_malware("script.vbs", within_archive=True))
        self.assertTrue(format_handlers.scan_for_malware("exploit.js", within_archive=True))
        # .exe might be considered safe if not in archive context by this mock's logic
        self.assertFalse(format_handlers.scan_for_malware("myapp.exe", within_archive=False))

    def test_scan_for_malware_case_insensitivity(self):
        self.assertTrue(format_handlers.scan_for_malware("MYAPP.EXE", within_archive=True))
        self.assertTrue(format_handlers.scan_for_malware("SUSPICIOUS.SCR", within_archive=False))


    # --- Tests for Google API format handlers (mocked) ---
    def test_process_google_slides_mocked(self):
        """Test mocked Google Slides processing."""
        # Simulate the case where an API client might be passed (though it's not used by the mock logic yet)
        mock_api_client = MagicMock()

        # Test with a known mock ID
        text_content = format_handlers.process_google_slides("sample_slide_id_1", mock_api_client=mock_api_client)
        self.assertIn("Slide 1 Title", text_content)
        self.assertIn("Slide 2 Bullet 1", text_content)

        # Test with an empty mock ID
        text_empty = format_handlers.process_google_slides("empty_slide_id", mock_api_client=mock_api_client)
        self.assertEqual(text_empty, "")

        # Test with an unknown mock ID
        text_unknown = format_handlers.process_google_slides("unknown_id", mock_api_client=mock_api_client)
        self.assertIn("Unknown Google Slide content", text_unknown)

        # Test with no API client (should return placeholder)
        text_no_client = format_handlers.process_google_slides("any_id", mock_api_client=None)
        self.assertIn("API client missing - placeholder", text_no_client)

    def test_process_google_sheets_mocked(self):
        """Test mocked Google Sheets processing."""
        mock_api_client = MagicMock()

        # Test with a known mock ID
        csv_data = format_handlers.process_google_sheets("sample_sheet_id_1", mock_api_client=mock_api_client)
        self.assertIn("Header1,Header2,Header3", csv_data)
        self.assertIn("Val1A,Val1B,Val1C", csv_data)

        # Test with an empty mock ID
        csv_empty = format_handlers.process_google_sheets("empty_sheet_id", mock_api_client=mock_api_client)
        self.assertEqual(csv_empty, "")

        # Test with an unknown mock ID
        csv_unknown = format_handlers.process_google_sheets("unknown_id", mock_api_client=mock_api_client)
        self.assertIn("Unknown Google Sheet data", csv_unknown)

        # Test with no API client (should return placeholder)
        csv_no_client = format_handlers.process_google_sheets("any_id", mock_api_client=None)
        self.assertIn("API client missing - placeholder", csv_no_client)


if __name__ == '__main__':
    unittest.main()
```
