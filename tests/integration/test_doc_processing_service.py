import pytest
import httpx
import os
import logging
from unittest.mock import MagicMock, patch, ANY
import io
from typing import Dict, Any

# Configure logger for tests
logger = logging.getLogger(__name__)

# Base URL for the Document Processing Service
DOC_PROCESSING_SERVICE_BASE_URL = os.getenv("TEST_DOC_PROC_SERVICE_URL", "http://localhost:8002")
# API Key for testing (if the gateway was in front, but we test service directly here)
# For direct service test, API key is not used by the service itself.
# If testing via gateway: API_KEY = os.getenv("TEST_API_KEY", "testkey1")

# Helper to create a dummy file for upload
def create_dummy_file(filename: str, content: str, encoding: str = 'utf-8') -> io.BytesIO:
    return io.BytesIO(content.encode(encoding))

@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"

# Mock QdrantClient and SentenceTransformer at the service level for all tests in this module
# This assumes these are attributes of the service instance or globally accessible in its module.
# If they are instantiated per request, mocking needs to happen differently (e.g. patching the class).
# For this test, we assume they are initialized at service startup and can be patched.

@pytest.fixture(autouse=True)
def mock_qdrant_and_embedding(mocker):
    """Mocks QdrantClient and SentenceTransformer used by the service."""
    mock_qdrant = MagicMock()
    mock_qdrant.upsert.return_value = None # or some success indicator if needed

    mock_embedder = MagicMock()
    # Simulate encode returning a list of floats (embedding) of a certain dimension
    # The actual dimension doesn't matter much for this mock as long as it's a list.
    mock_embedder.encode.return_value = [[0.1, 0.2, 0.3]] # Batch encoding returns list of lists
    # If service calls encode on single strings:
    # mock_embedder.encode.return_value = [0.1, 0.2, 0.3] # Single encoding
    # Let's assume it's called with a single string for now based on doc_processing_service logic
    mock_embedder.encode.return_value = [0.1, 0.2, 0.3]
    mock_embedder.get_sentence_embedding_dimension.return_value = 3 # Match the dummy embedding

    # Patch where these are instantiated or accessed in doc_processing_service.py
    # Adjust path according to actual usage in the service.
    mocker.patch('Scripts.services.doc_processing_service.qdrant_client', mock_qdrant)
    mocker.patch('Scripts.services.doc_processing_service.embedding_model', mock_embedder)

    return mock_qdrant, mock_embedder


@pytest.mark.asyncio
async def process_file_test_helper(
    filename: str,
    file_content: str,
    metadata: Dict[str, Any],
    expected_status_code: int,
    expected_substring_in_message: str,
    expected_status_in_response: str,
    mock_qdrant_and_embedding_fixture # To ensure mocks are active
):
    mock_qdrant, mock_embedder = mock_qdrant_and_embedding_fixture
    url = f"{DOC_PROCESSING_SERVICE_BASE_URL}/process_document"

    files = {'file': (filename, create_dummy_file(filename, file_content), 'application/octet-stream')}
    data = {'metadata_json': json.dumps(metadata)}

    async with httpx.AsyncClient(timeout=20.0) as client:
        logger.info(f"Testing file: {filename}, metadata: {metadata}")
        response = await client.post(url, files=files, data=data)

    logger.info(f"Response for {filename}: {response.status_code}, {response.text}")
    assert response.status_code == expected_status_code
    response_json = response.json()
    assert expected_substring_in_message in response_json.get("message", "")
    assert response_json.get("status") == expected_status_in_response

    if expected_status_in_response == "indexed":
        mock_embedder.encode.assert_called_once() # Or called with specific text if checking that
        mock_qdrant.upsert.assert_called_once()

        # Verify payload structure passed to Qdrant
        args, kwargs = mock_qdrant.upsert.call_args
        assert 'points' in kwargs
        points = kwargs['points']
        assert len(points) == 1
        point = points[0]
        assert "text" in point.payload
        assert file_content.strip() in point.payload["text"] # Check if extracted text is part of payload
        assert "metadata" in point.payload
        assert point.payload["metadata"]["source"] == metadata["source"]
        assert "keywords" in point.payload["metadata"]
        assert "term_frequencies" in point.payload["metadata"]
        assert "original_filename" in point.payload["metadata"]
        assert point.payload["metadata"]["original_filename"] == filename
        if metadata.get("document_id"):
            assert point.id == metadata["document_id"]
        else:
            assert isinstance(point.id, str) # UUID4 string
    else: # Not indexed
        mock_embedder.encode.assert_not_called()
        mock_qdrant.upsert.assert_not_called()

    # Reset mocks for the next test case if they are stateful from call counts
    mock_embedder.reset_mock()
    mock_qdrant.reset_mock()


# Parameterized test cases
test_data_success = [
    ("test.txt", "Simple text content for TXT.", {"source": "txt_test", "document_id": "txt-001"}, 200, "indexed successfully", "indexed"),
    ("test.pdf", "PDF content extracted.", {"source": "pdf_test", "document_id": "pdf-001"}, 200, "indexed successfully", "indexed"),
    ("test.docx", "DOCX content extracted.", {"source": "docx_test", "document_id": "docx-001"}, 200, "indexed successfully", "indexed"),
    ("test.pptx", "PPTX content extracted.", {"source": "pptx_test", "document_id": "pptx-001"}, 200, "indexed successfully", "indexed"),
    ("test.xlsx", "XLSX cell1 data. Cell2 data.", {"source": "xlsx_test", "document_id": "xlsx-001"}, 200, "indexed successfully", "indexed"),
]

# Need to mock the actual file reading part for PDF, DOCX, PPTX, XLSX if not sending raw text
# For this test, we send simple string content and rely on the filename extension for routing in the service.
# The service's extraction logic is what's being implicitly tested.
# To truly test extraction, the `file_content` here would be binary content of a real sample file.
# For now, this tests the routing and that the mock `text_to_embed` (which would be the extracted text) gets processed.
# Let's adjust the helper to actually pass "binary" content for those that need it, and mock extraction for unit,
# but for this integration test, we assume the service's extraction works and it will call encode.

# Re-think: For integration test, we *should* send actual file content if possible.
# However, mocking Qdrant/Embedding means we are not testing *their* interaction with varied text.
# We are testing: can the service *receive* the file, *route* based on type, *attempt* extraction,
# and then *call* the (mocked) embedder and Qdrant client with *some* text.

# For simplicity, the `file_content` will be used as the "extracted text" by the mocks.
# The service itself will try to extract, but our mocks for embedder/qdrant won't care about the *actual* extracted text.
# This is a limitation of not having true end-to-end with real dependencies.

@pytest.mark.parametrize("filename, file_content, metadata, code, msg_substr, status", test_data_success)
@pytest.mark.asyncio
async def test_process_document_supported_types(
    filename, file_content, metadata, code, msg_substr, status,
    mock_qdrant_and_embedding_fixture, mocker # mocker for patching open if using real files
):
    # If we were using real files, we'd mock 'fitz.open', 'DocxDocument', 'Presentation', 'load_workbook'
    # to return objects that yield `file_content` when their text extraction methods are called.
    # Example for fitz (PDF):
    if filename.endswith(".pdf"):
        mock_pdf_page = MagicMock()
        mock_pdf_page.get_text.return_value = file_content
        mock_pdf_doc = MagicMock()
        mock_pdf_doc.load_page.return_value = mock_pdf_page
        mock_pdf_doc.__len__.return_value = 1 # Number of pages
        mocker.patch('fitz.open', return_value=mock_pdf_doc)
    elif filename.endswith(".docx"):
        mock_docx_doc = MagicMock()
        mock_para = MagicMock()
        mock_para.text = file_content # Simplification, real docx has list of paras
        mock_docx_doc.paragraphs = [mock_para]
        mocker.patch('Scripts.services.doc_processing_service.DocxDocument', return_value=mock_docx_doc) # Patch where it's used
    elif filename.endswith(".pptx"):
        mock_pptx_pres = MagicMock()
        mock_slide = MagicMock()
        mock_shape = MagicMock()
        mock_text_frame = MagicMock()
        mock_paragraph = MagicMock()
        mock_run = MagicMock()
        mock_run.text = file_content # Simplified
        mock_paragraph.runs = [mock_run]
        mock_text_frame.paragraphs = [mock_paragraph]
        mock_shape.text_frame = mock_text_frame
        mock_shape.has_text_frame = True
        mock_slide.shapes = [mock_shape]
        mock_slide.has_notes_slide = False # Simplify: no notes
        mock_pptx_pres.slides = [mock_slide]
        mocker.patch('Scripts.services.doc_processing_service.Presentation', return_value=mock_pptx_pres)
    elif filename.endswith(".xlsx"):
        mock_xlsx_wb = MagicMock()
        mock_sheet = MagicMock()
        mock_cell1 = MagicMock()
        mock_cell1.value = file_content.split('.')[0] # "XLSX cell1 data"
        mock_cell2 = MagicMock()
        mock_cell2.value = file_content.split('.')[1].strip() if '.' in file_content else "" # "Cell2 data"
        mock_sheet.iter_rows.return_value = iter([ (mock_cell1, mock_cell2) ]) # Simulate one row with two cells
        mock_xlsx_wb.sheetnames = ["Sheet1"]
        mock_xlsx_wb.__getitem__.return_value = mock_sheet # workbook['Sheet1']
        mocker.patch('Scripts.services.doc_processing_service.load_workbook', return_value=mock_xlsx_wb)


    await process_file_test_helper(filename, file_content, metadata, code, msg_substr, status, mock_qdrant_and_embedding_fixture)


@pytest.mark.asyncio
async def test_process_document_unsupported_type(mock_qdrant_and_embedding_fixture):
    await process_file_test_helper(
        "test.unsupported", "content", {"source": "unsupported_test"},
        200, # Service currently returns 200 for unsupported but logs warning
        "type not supported for content extraction",
        "received_metadata_only",
        mock_qdrant_and_embedding_fixture
    )

@pytest.mark.asyncio
async def test_process_document_no_file_or_content(mock_qdrant_and_embedding_fixture):
    # This test sends neither 'file' nor 'document_content'
    mock_qdrant, mock_embedder = mock_qdrant_and_embedding_fixture
    url = f"{DOC_PROCESSING_SERVICE_BASE_URL}/process_document"
    data = {'metadata_json': json.dumps({"source": "no_content_test"})} # Only metadata

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(url, data=data) # No files part

    assert response.status_code == 400 # Expecting Bad Request
    response_json = response.json()
    assert "must be provided for processing" in response_json.get("detail", "")
    mock_embedder.encode.assert_not_called()
    mock_qdrant.upsert.assert_not_called()

@pytest.mark.asyncio
async def test_process_document_invalid_metadata_json(mock_qdrant_and_embedding_fixture):
    mock_qdrant, mock_embedder = mock_qdrant_and_embedding_fixture
    url = f"{DOC_PROCESSING_SERVICE_BASE_URL}/process_document"

    # Create a dummy text file for this test
    filename = "valid_file.txt"
    file_content = "This content is valid."
    files = {'file': (filename, create_dummy_file(filename, file_content), 'text/plain')}
    data = {'metadata_json': 'this is not valid json'} # Invalid JSON

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(url, files=files, data=data)

    assert response.status_code == 400
    response_json = response.json()
    assert "Invalid JSON format for metadata" in response_json.get("detail", "")
    mock_embedder.encode.assert_not_called()
    mock_qdrant.upsert.assert_not_called()

# To run: pytest tests/integration/test_doc_processing_service.py
# Ensure Doc Processing Service is running at TEST_DOC_PROC_SERVICE_URL (default http://localhost:8002)
# The mocks for Qdrant and SentenceTransformer mean no actual Qdrant/model loading is needed for the service *during these tests*.
# However, the service must be able to start up (i.e., its own Python dependencies must be met).
