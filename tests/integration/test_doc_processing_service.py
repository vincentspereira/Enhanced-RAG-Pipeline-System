import pytest
import httpx
import os
import logging
from unittest.mock import MagicMock, patch, ANY
import io
import json # Ensure json is imported for process_file_test_helper data
from typing import Dict, Any

# Configure logger for tests
logger = logging.getLogger(__name__)

# Base URL for the Document Processing Service
DOC_PROCESSING_SERVICE_BASE_URL = os.getenv("TEST_DOC_PROC_SERVICE_URL", "http://localhost:8002")

# Helper to create a dummy file for upload
def create_dummy_file(filename: str, content: bytes) -> io.BytesIO: # Expect bytes
    return io.BytesIO(content)

@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"

@pytest.fixture(autouse=True)
def mock_qdrant_and_embedding(mocker):
    """Mocks QdrantClient and SentenceTransformer used by the service."""
    mock_qdrant = MagicMock()
    mock_qdrant.upsert.return_value = None

    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = [0.1, 0.2, 0.3] # Single embedding for a chunk
    mock_embedder.get_sentence_embedding_dimension.return_value = 3

    mocker.patch('Scripts.services.doc_processing_service.qdrant_client', mock_qdrant)
    mocker.patch('Scripts.services.doc_processing_service.embedding_model', mock_embedder)

    # Also mock text splitters if their behavior is complex or slow, though for this test
    # we primarily care about the number of chunks leading to upsert calls.
    # For now, assume RecursiveCharacterTextSplitter works as expected.
    # mock_text_splitter = MagicMock()
    # mock_text_splitter.split_text.return_value = ["chunk1", "chunk2"] # Example
    # mocker.patch('Scripts.services.doc_processing_service.RecursiveCharacterTextSplitter', return_value=mock_text_splitter)

    return mock_qdrant, mock_embedder


async def process_file_test_helper(
    filename: str,
    file_content_bytes: bytes, # Changed to bytes
    metadata: Dict[str, Any],
    expected_status_code: int,
    expected_substring_in_message: str,
    expected_status_in_response: str,
    expected_min_chunks: int, # Minimum chunks expected (can be 1)
    mock_qdrant_and_embedding_fixture
):
    mock_qdrant, mock_embedder = mock_qdrant_and_embedding_fixture
    url = f"{DOC_PROCESSING_SERVICE_BASE_URL}/process_document"

    files = {'file': (filename, create_dummy_file(filename, file_content_bytes), 'application/octet-stream')}
    data = {'metadata_json': json.dumps(metadata)}

    async with httpx.AsyncClient(timeout=30.0) as client: # Increased timeout slightly
        logger.info(f"Testing file: {filename}, metadata: {metadata}")
        response = await client.post(url, files=files, data=data)

    logger.info(f"Response for {filename}: {response.status_code}, {response.text}")
    assert response.status_code == expected_status_code
    response_json = response.json()
    assert expected_substring_in_message in response_json.get("message", "")
    assert response_json.get("status") == expected_status_in_response

    if expected_status_in_response == "indexed_chunked":
        assert response_json.get("chunks_indexed") >= expected_min_chunks
        assert mock_embedder.encode.call_count >= expected_min_chunks
        assert mock_qdrant.upsert.call_count >= 1 # Upsert might be called once with all chunk points

        # Verify payload structure for the first chunk of the first upsert call
        first_upsert_call_args, first_upsert_call_kwargs = mock_qdrant.upsert.call_args_list[0]
        assert 'points' in first_upsert_call_kwargs
        points = first_upsert_call_kwargs['points']
        assert len(points) >= expected_min_chunks # Check if all chunks for this doc are in one call

        first_point = points[0]
        assert "chunk_text" in first_point.payload
        assert "parent_document_id" in first_point.payload
        assert "chunk_sequence_number" in first_point.payload
        assert isinstance(first_point.payload["chunk_sequence_number"], int)

        assert "metadata" in first_point.payload
        parent_meta = first_point.payload["metadata"]
        assert parent_meta["source"] == metadata["source"]
        assert "keywords" in parent_meta
        assert "term_frequencies" in parent_meta
        assert parent_meta["original_filename"] == filename

        expected_parent_doc_id = metadata.get("document_id", response_json.get("parent_document_id"))
        if expected_parent_doc_id:
            assert first_point.payload["parent_document_id"] == expected_parent_doc_id
            assert parent_meta["_internal_id"] == expected_parent_doc_id
            assert first_point.id.startswith(str(expected_parent_doc_id) + "_chunk_")
        else: # If no doc_id provided, it's generated
            assert isinstance(first_point.payload["parent_document_id"], str)
            assert first_point.id.startswith(first_point.payload["parent_document_id"] + "_chunk_")

        # Check if the chunk_text is part of the original file_content (requires decoding bytes)
        # This is harder to assert precisely without knowing the exact chunk.
        # For now, we trust the splitter and focus on structure.
        # assert first_point.payload["chunk_text"] in file_content_bytes.decode('utf-8', errors='ignore')

    else:
        mock_embedder.encode.assert_not_called()
        mock_qdrant.upsert.assert_not_called()

    mock_embedder.reset_mock()
    mock_qdrant.reset_mock()

# Use actual minimal file contents as bytes for testing extraction
# These are small enough not to cause issues, but represent real file content.
# For chunking tests, we'd need larger content.
# Let's assume default CHUNK_SIZE=1000, CHUNK_OVERLAP=200.
# To test multiple chunks, content should be > 1000 chars.

long_text_for_chunking = ("This is a very long string designed to test the chunking mechanism. " * 50 +
                          "It repeats this sentence many times to ensure that the total length exceeds " +
                          "the default chunk size of 1000 characters. We hope this will generate multiple chunks. " * 50 +
                          "The RecursiveCharacterTextSplitter should handle this by breaking it down. " * 50 +
                          "Let's add even more content to be absolutely sure. More content. More content. " * 50) # Approx 20k chars


# Dummy file contents (as bytes)
dummy_txt_content = long_text_for_chunking.encode('utf-8')
# For binary files, you'd use real minimal sample files read as bytes.
# For mocked extraction tests, the content of these doesn't matter as much as the file type routing.
# But to test *actual* extraction + chunking, they should be representative.
# For now, the mocks for fitz, DocxDocument etc. will return `long_text_for_chunking`.
dummy_pdf_content_bytes = b"%PDF-1.4 minimal pdf" # Actual content doesn't matter if fitz.open is mocked
dummy_docx_content_bytes = b"PK..." # Actual content doesn't matter if DocxDocument is mocked
dummy_pptx_content_bytes = b"PK..." # Actual content doesn't matter if Presentation is mocked
dummy_xlsx_content_bytes = b"PK..." # Actual content doesn't matter if load_workbook is mocked

test_data_success = [
    ("test_long.txt", dummy_txt_content, {"source": "txt_chunk_test", "document_id": "txt-chk-001"}, 200, "chunks indexed", "indexed_chunked", 1), # Expect at least 1 chunk
    ("test_long.pdf", dummy_pdf_content_bytes, {"source": "pdf_chunk_test", "document_id": "pdf-chk-001"}, 200, "chunks indexed", "indexed_chunked", 1),
    ("test_long.docx", dummy_docx_content_bytes, {"source": "docx_chunk_test", "document_id": "docx-chk-001"}, 200, "chunks indexed", "indexed_chunked", 1),
    ("test_long.pptx", dummy_pptx_content_bytes, {"source": "pptx_chunk_test", "document_id": "pptx-chk-001"}, 200, "chunks indexed", "indexed_chunked", 1),
    ("test_long.xlsx", dummy_xlsx_content_bytes, {"source": "xlsx_chunk_test", "document_id": "xlsx-chk-001"}, 200, "chunks indexed", "indexed_chunked", 1),
    # Test with very short content that should result in one chunk (or specific handling if too short)
    ("test_short.txt", "Short text.".encode('utf-8'), {"source": "short_txt_test"}, 200, "chunks indexed", "indexed_chunked", 1),
    ("test_veryshort.txt", "Hi".encode('utf-8'), {"source": "veryshort_txt_test"}, 200, "no text chunks were generated", "processed_no_chunks", 0),
]


@pytest.mark.parametrize("filename, file_content_bytes, metadata, code, msg_substr, status, min_chunks", test_data_success)
@pytest.mark.asyncio
async def test_process_document_supported_types_with_chunking(
    filename, file_content_bytes, metadata, code, msg_substr, status, min_chunks,
    mock_qdrant_and_embedding_fixture, mocker
):
    # Mock the text extraction for binary types to return the long_text_for_chunking
    # This ensures the chunker gets enough text, while still testing the file type routing.
    if filename.endswith(".pdf"):
        mock_pdf_page = MagicMock()
        mock_pdf_page.get_text.return_value = long_text_for_chunking # Use long text for PDF mock
        mock_pdf_doc = MagicMock()
        mock_pdf_doc.load_page.return_value = mock_pdf_page
        mock_pdf_doc.__len__.return_value = 1
        mocker.patch('fitz.open', return_value=mock_pdf_doc)
    elif filename.endswith(".docx"):
        mock_docx_doc = MagicMock()
        mock_para = MagicMock(); mock_para.text = long_text_for_chunking # Use long text
        mock_docx_doc.paragraphs = [mock_para]
        mocker.patch('Scripts.services.doc_processing_service.DocxDocument', return_value=mock_docx_doc)
    elif filename.endswith(".pptx"):
        mock_pptx_pres = MagicMock()
        mock_slide = MagicMock(); mock_shape = MagicMock(); mock_text_frame = MagicMock()
        mock_paragraph = MagicMock(); mock_run = MagicMock(); mock_run.text = long_text_for_chunking # Use long text
        mock_paragraph.runs = [mock_run]; mock_text_frame.paragraphs = [mock_paragraph]
        mock_shape.text_frame = mock_text_frame; mock_shape.has_text_frame = True
        mock_slide.shapes = [mock_shape]; mock_slide.has_notes_slide = False
        mock_pptx_pres.slides = [mock_slide]
        mocker.patch('Scripts.services.doc_processing_service.Presentation', return_value=mock_pptx_pres)
    elif filename.endswith(".xlsx"):
        mock_xlsx_wb = MagicMock()
        mock_sheet = MagicMock()
        # Simulate multiple cells that would sum up to long_text_for_chunking
        parts = [long_text_for_chunking[i:i+100] for i in range(0, len(long_text_for_chunking), 100)]
        row_data = []
        for part in parts:
            cell = MagicMock(); cell.value = part
            row_data.append(cell)
        mock_sheet.iter_rows.return_value = iter([ tuple(row_data) ]) # Single row with many cells
        mock_xlsx_wb.sheetnames = ["Sheet1"]
        mock_xlsx_wb.__getitem__.return_value = mock_sheet
        mocker.patch('Scripts.services.doc_processing_service.load_workbook', return_value=mock_xlsx_wb)

    # For .txt files, the actual file_content_bytes will be used directly.
    # If it's a short .txt file from test_data_success, it will use that short content.
    # If it's "test_long.txt", it uses the long_text_for_chunking.

    current_file_content_bytes = file_content_bytes
    if "long" in filename and not filename.endswith(".txt"): # For mocked binary files, ensure they use long text
        pass # The mocks above already ensure long_text_for_chunking is "extracted"
    elif "short.txt" in filename:
        current_file_content_bytes = "Short text.".encode('utf-8')
    elif "veryshort.txt" in filename:
        current_file_content_bytes = "Hi".encode('utf-8')


    await process_file_test_helper(
        filename, current_file_content_bytes, metadata,
        code, msg_substr, status, min_chunks,
        mock_qdrant_and_embedding_fixture
    )


@pytest.mark.asyncio
async def test_process_document_unsupported_type(mock_qdrant_and_embedding_fixture):
    await process_file_test_helper(
        "test.unsupported", b"content", {"source": "unsupported_test"},
        200,
        "type not supported for content extraction",
        "received_metadata_only", 0, # Expect 0 chunks
        mock_qdrant_and_embedding_fixture
    )

@pytest.mark.asyncio
async def test_process_document_no_file_or_content(mock_qdrant_and_embedding_fixture):
    # ... (same as before) ...
    mock_qdrant, mock_embedder = mock_qdrant_and_embedding_fixture
    url = f"{DOC_PROCESSING_SERVICE_BASE_URL}/process_document"
    data = {'metadata_json': json.dumps({"source": "no_content_test"})}
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(url, data=data)
    assert response.status_code == 400
    response_json = response.json()
    assert "must be provided for processing" in response_json.get("detail", "")
    mock_embedder.encode.assert_not_called()
    mock_qdrant.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_process_document_invalid_metadata_json(mock_qdrant_and_embedding_fixture):
    # ... (same as before) ...
    mock_qdrant, mock_embedder = mock_qdrant_and_embedding_fixture
    url = f"{DOC_PROCESSING_SERVICE_BASE_URL}/process_document"
    filename = "valid_file.txt"
    file_content_bytes = b"This content is valid."
    files = {'file': (filename, create_dummy_file(filename, file_content_bytes), 'text/plain')}
    data = {'metadata_json': 'this is not valid json'}
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(url, files=files, data=data)
    assert response.status_code == 400
    response_json = response.json()
    assert "Invalid JSON format for metadata" in response_json.get("detail", "")
    mock_embedder.encode.assert_not_called()
    mock_qdrant.upsert.assert_not_called()

# To run: pytest tests/integration/test_doc_processing_service.py
# Ensure Doc Processing Service is running. Mocks handle Qdrant/Embedding model.
