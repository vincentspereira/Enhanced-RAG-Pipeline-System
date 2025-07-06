"""
Specialized format handlers for the DocumentProcessor.
This module will contain functions to process file types that require
more complex parsing or external libraries.
"""
import logging
import os
import zipfile
import tarfile
# import rarfile # Requires unrar package
# import py7zr # Requires py7zr package
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import pypandoc # Requires pandoc installation
import nbformat # For Jupyter notebooks
from email import policy
from email.parser import BytesParser
import extract_msg # For .msg files
from pathlib import Path # Added for Path object usage
# from odf import text, teletype # For ODP, ODS
# from key_reader import KeyReader # For Apple Keynote (hypothetical)
# from numbers_parser import Document as NDocument # For Apple Numbers (hypothetical)


logger = logging.getLogger(__name__)

# Placeholder for malware scanning - enhanced mockup
def scan_for_malware(file_path: str, within_archive: bool = True) -> bool:
    """
    Simulates basic malware scanning by checking for risky file extensions.
    In a real system, this would integrate a proper scanner like ClamAV.
    If 'within_archive' is True, it's more stringent with executable-like extensions.
    """
    filename = os.path.basename(file_path).lower()
    risky_extensions_general = {'.scr', '.pif', '.com'} # Generally suspicious anywhere
    risky_extensions_in_archive = {'.exe', '.vbs', '.js', '.bat', '.cmd', '.ps1', '.jar', '.dll'} # More suspicious inside archives

    extension = os.path.splitext(filename)[1]

    if extension in risky_extensions_general:
        logger.warning(f"Malware Scan Mockup: File '{filename}' has a generally risky extension '{extension}'. Flagged.")
        return True

    if within_archive and extension in risky_extensions_in_archive:
        logger.warning(f"Malware Scan Mockup: File '{filename}' (in archive context) has risky extension '{extension}'. Flagged.")
        return True

    # Add a simple content check placeholder (very basic, not real detection)
    # try:
    #     with open(file_path, 'rb') as f:
    #         content_sample = f.read(1024) # Read first 1KB
    #         if b"MZ" in content_sample and extension not in {".exe", ".dll"}: # MZ is start of .exe
    #             logger.warning(f"Malware Scan Mockup: File '{filename}' contains PE header but is not .exe/.dll. Suspicious.")
    #             return True
    # except Exception:
    #     pass # Ignore read errors for this mock

    logger.debug(f"Malware Scan Mockup: File '{filename}' passed basic extension check.")
    return False # Assume no malware for this mockup

def process_epub(file_path: str) -> str:
    """Processes EPUB files."""
    try:
        book = epub.read_epub(file_path)
        content = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            content.append(soup.get_text())
        return "\n".join(content)
    except Exception as e:
        logger.error(f"Error processing EPUB {file_path}: {e}")
        return ""

def process_mobi(file_path: str) -> str:
    """Processes MOBI files. Often requires conversion, e.g., via Calibre or pandoc."""
    logger.warning(f"Direct MOBI processing is complex. Attempting conversion with pandoc for {file_path}.")
    try:
        # Pandoc can convert MOBI to plain text
        output_text = pypandoc.convert_file(file_path, 'plain', format='mobi')
        return output_text
    except Exception as e:
        logger.error(f"Error processing MOBI {file_path} with pandoc: {e}. Consider Calibre for robust conversion.")
        return ""

def process_latex(file_path: str) -> str:
    """Processes LaTeX files (.tex) using pandoc."""
    logger.info(f"Processing LaTeX file {file_path} using pandoc.")
    try:
        # Convert LaTeX to plain text
        output_text = pypandoc.convert_file(file_path, 'plain', format='latex')
        return output_text
    except Exception as e:
        logger.error(f"Error processing LaTeX file {file_path} with pandoc: {e}")
        return ""

def process_rst(file_path: str) -> str:
    """Processes reStructuredText files (.rst) using pandoc."""
    logger.info(f"Processing reStructuredText file {file_path} using pandoc.")
    try:
        output_text = pypandoc.convert_file(file_path, 'plain', format='rst')
        return output_text
    except Exception as e:
        logger.error(f"Error processing reStructuredText file {file_path} with pandoc: {e}")
        return ""

def process_asciidoc(file_path: str) -> str:
    """Processes AsciiDoc files (.adoc, .asciidoc) using pandoc."""
    logger.info(f"Processing AsciiDoc file {file_path} using pandoc.")
    try:
        output_text = pypandoc.convert_file(file_path, 'plain', format='asciidoc')
        return output_text
    except Exception as e:
        logger.error(f"Error processing AsciiDoc file {file_path} with pandoc: {e}")
        return ""

def process_ipynb(file_path: str) -> str:
    """Processes Jupyter Notebook files (.ipynb)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        content = []
        for cell in notebook.cells:
            if cell.cell_type == 'markdown':
                content.append(cell.source)
            elif cell.cell_type == 'code':
                # Include code as text, could also consider outputs if relevant
                content.append("# Code Cell:\n" + cell.source)
                # For outputs:
                # for output in cell.outputs:
                #     if output.output_type == 'stream':
                #         content.append("# Output:\n" + output.text)
                #     elif 'data' in output and 'text/plain' in output.data:
                #         content.append("# Output:\n" + output.data['text/plain'])
        return "\n\n".join(content)
    except Exception as e:
        logger.error(f"Error processing Jupyter Notebook {file_path}: {e}")
        return ""

def process_archive(file_path: str, extract_to_temp_dir_fn: callable) -> List[str]:
    """
    Processes archive files (ZIP, TAR) by extracting their contents.
    RAR and 7Z would require external libraries and unrar/7zip command line tools.
    Returns a list of paths to extracted files.
    """
    file_path_obj = Path(file_path)
    ext = file_path_obj.suffix.lower()
    extracted_files = []

    # Create a unique temporary directory for extraction
    temp_extraction_path = extract_to_temp_dir_fn()

    try:
        if ext == ".zip":
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_extraction_path)
            logger.info(f"Extracted ZIP archive {file_path} to {temp_extraction_path}")
        elif ext in [".tar", ".gz", ".bz2", ".xz"]: # Handling .tar.gz, .tar.bz2 etc.
            # tarfile can often handle compressed tars automatically
            mode = 'r:*' # Auto-detect compression
            if ext == ".tar": mode = 'r:'

            with tarfile.open(file_path, mode) as tar_ref:
                tar_ref.extractall(temp_extraction_path)
            logger.info(f"Extracted TAR archive {file_path} to {temp_extraction_path}")
        # elif ext == ".rar":
        #     # Requires rarfile and unrar utility
        #     # with rarfile.RarFile(file_path, 'r') as rar_ref:
        #     #     rar_ref.extractall(temp_extraction_path)
        #     # logger.info(f"Extracted RAR archive {file_path} to {temp_extraction_path}")
        #     logger.warning("RAR processing requires 'rarfile' and 'unrar'. Placeholder.")
        #     pass
        # elif ext == ".7z":
        #     # Requires py7zr or sevenzip command line
        #     # with py7zr.SevenZipFile(file_path, mode='r') as z_ref:
        #     #     z_ref.extractall(path=temp_extraction_path)
        #     # logger.info(f"Extracted 7Z archive {file_path} to {temp_extraction_path}")
        #     logger.warning("7Z processing requires 'py7zr' or '7z' utility. Placeholder.")
        #     pass
        else:
            logger.warning(f"Unsupported archive type for direct processing: {ext}")
            return []

        for root, _, files in os.walk(temp_extraction_path):
            for file in files:
                full_file_path = os.path.join(root, file)
                if not scan_for_malware(full_file_path): # Basic malware scan placeholder
                    extracted_files.append(full_file_path)
                else:
                    logger.warning(f"Potential malware detected in {full_file_path}. Skipping.")

        return extracted_files # Return paths of extracted files for further processing by main processor

    except Exception as e:
        logger.error(f"Error processing archive {file_path}: {e}")
        # Clean up temp dir on error before returning
        # shutil.rmtree(temp_extraction_path) # Be careful with recursive delete
        return []


# --- Placeholders for formats requiring APIs or more complex libraries ---

def process_google_slides(file_identifier: str, mock_api_client: Optional[Any] = None) -> str:
    """
    Processes Google Slides using a (mocked) API client.
    'file_identifier' is expected to be a Google Slides File ID.
    'mock_api_client' would be a real Google API client service object in production.
    """
    logger.info(f"Attempting to process Google Slide with ID: {file_identifier}")
    if mock_api_client is None:
        logger.warning("No Google API client provided for Slides. Returning placeholder text.")
        return f"Google Slide content for ID {file_identifier} (API client missing - placeholder)."

    try:
        # --- Mocked API Interaction ---
        # In a real scenario:
        # presentation = mock_api_client.presentations().get(presentationId=file_identifier).execute()
        # slides = presentation.get('slides', [])
        # text_elements = []
        # for slide in slides:
        #     for pageElement in slide.get('pageElements', []):
        #         if 'shape' in pageElement and 'text' in pageElement['shape']:
        #             for textRun in pageElement['shape']['text'].get('textElements', []):
        #                 if 'textRun' in textRun and 'content' in textRun['textRun']:
        #                     text_elements.append(textRun['textRun']['content'])
        #         # Could also extract from notesPage: slide.get('notesPage', {}).get(... )
        # return "\n".join(text_elements)
        # --- End of Real Scenario ---

        # Mocked response based on file_id for testing:
        if file_identifier == "sample_slide_id_1":
            logger.info(f"Processing Google Slide (mocked): {file_identifier}")
            return "Slide 1 Title\nSlide 1 Body Text.\nNotes: Important note 1.\n\nSlide 2 Title\nSlide 2 Bullet 1\nSlide 2 Bullet 2."
        elif file_identifier == "empty_slide_id":
            logger.info(f"Processing empty Google Slide (mocked): {file_identifier}")
            return ""
        else:
            logger.warning(f"Unknown Google Slide ID for mock processing: {file_identifier}")
            return f"Unknown Google Slide content for ID {file_identifier} (mocked)."

    except Exception as e:
        logger.error(f"Error processing Google Slide ID {file_identifier} (even with mock): {e}")
        return f"Error processing Google Slide ID {file_identifier}."

def process_google_sheets(file_identifier: str, mock_api_client: Optional[Any] = None) -> str:
    """
    Processes Google Sheets using a (mocked) API client.
    'file_identifier' is expected to be a Google Sheets File ID.
    'mock_api_client' would be a real Google API client service object in production.
    Returns a string representation (e.g., CSV-like) of the sheet data.
    """
    logger.info(f"Attempting to process Google Sheet with ID: {file_identifier}")
    if mock_api_client is None:
        logger.warning("No Google API client provided for Sheets. Returning placeholder text.")
        return f"Google Sheet data for ID {file_identifier} (API client missing - placeholder)."

    try:
        # --- Mocked API Interaction ---
        # In a real scenario:
        # result = mock_api_client.spreadsheets().values().get(spreadsheetId=file_identifier, range="Sheet1").execute() # Example: get Sheet1
        # values = result.get('values', [])
        # csv_representation = "\n".join([",".join(map(str, row)) for row in values])
        # return csv_representation
        # --- End of Real Scenario ---

        # Mocked response:
        if file_identifier == "sample_sheet_id_1":
            logger.info(f"Processing Google Sheet (mocked): {file_identifier}")
            return "Header1,Header2,Header3\nVal1A,Val1B,Val1C\nVal2A,Val2B,Val2C"
        elif file_identifier == "empty_sheet_id":
            logger.info(f"Processing empty Google Sheet (mocked): {file_identifier}")
            return ""
        else:
            logger.warning(f"Unknown Google Sheet ID for mock processing: {file_identifier}")
            return f"Unknown Google Sheet data for ID {file_identifier} (mocked)."

    except Exception as e:
        logger.error(f"Error processing Google Sheet ID {file_identifier} (even with mock): {e}")
        return f"Error processing Google Sheet ID {file_identifier}."

def process_odp(file_path: str) -> str:
    """Processes ODP (OpenDocument Presentation) files."""
    logger.warning(f"ODP processing for {file_path} is a placeholder. Consider using pandoc or specific libraries.")
    # try:
    #     textdoc = text.load(file_path)
    #     all_paras = textdoc.getElementsByType(teletype.P)
    #     return "\n".join(teletype.extractText(para) for para in all_paras)
    # except Exception as e:
    #     logger.error(f"Error processing ODP {file_path}: {e}")
    return f"Text from ODP file {os.path.basename(file_path)} (Placeholder - requires python-odf)"

def process_key(file_path: str) -> str:
    """Processes KEY (Apple Keynote) files."""
    logger.warning(f"KEY (Apple Keynote) processing for {file_path} is a placeholder. Requires specialized libraries like 'keynote-parser'.")
    # Example:
    # from keynote_parser import Keynote
    # pres = Keynote(file_path)
    # return "\n".join(slide.text for slide in pres.slides)
    return f"Text from Apple Keynote file {os.path.basename(file_path)} (Placeholder)"

def process_ods(file_path: str) -> str:
    """Processes ODS (OpenDocument Spreadsheet) files."""
    logger.warning(f"ODS processing for {file_path} is a placeholder. Consider using pandas with odfpy engine or specific libraries.")
    # Example:
    # import pandas as pd
    # df = pd.read_excel(file_path, engine='odf')
    # return df.to_string()
    return f"Data from ODS file {os.path.basename(file_path)} (Placeholder - requires pandas & odfpy)"

def process_numbers(file_path: str) -> str:
    """Processes NUMBERS (Apple Numbers) files."""
    logger.warning(f"NUMBERS (Apple Numbers) processing for {file_path} is a placeholder. Requires specialized libraries like 'numbers-parser'.")
    # Example:
    # from numbers_parser import Document
    # doc = Document(file_path)
    # sheets = doc.sheets
    # content = []
    # for sheet in sheets:
    #     tables = sheet.tables
    #     for table in tables:
    #         # crude extraction
    #         for row in table.rows(values_only=True):
    #             content.append(", ".join(map(str, filter(None, row))))
    # return "\n".join(content)
    return f"Data from Apple Numbers file {os.path.basename(file_path)} (Placeholder)"

def process_confluence_export(file_path: str) -> str:
    """
    Processes Confluence exports, expecting a ZIP file containing HTML pages.
    Extracts text from HTML files found within the root or 'pages' directory of the ZIP.
    """
    logger.info(f"Processing Confluence export: {file_path}")
    if not zipfile.is_zipfile(file_path):
        logger.warning(f"{file_path} is not a ZIP file. Confluence export handler expects a ZIP. Attempting direct HTML parse as fallback.")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser") # Use html.parser for broader compatibility
            return soup.get_text(separator='\n', strip=True)
        except Exception as e:
            logger.error(f"Failed to parse {file_path} as direct HTML after non-ZIP check: {e}")
            return f"Error: Could not process Confluence export {os.path.basename(file_path)} as ZIP or HTML."

    all_text_content = []
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            member_list = zip_ref.namelist()
            html_files_to_process = []

            for member_name in member_list:
                # Heuristic: process HTML files not in common attachment/style folders
                # Confluence exports might have pages in root or a 'pages' subdir.
                if member_name.lower().endswith(('.html', '.htm')):
                    if not any(ignored_folder in member_name.lower() for ignored_folder in ['attachments/', 'styles/', 'images/', 'css/', 'js/']):
                        html_files_to_process.append(member_name)

            if not html_files_to_process:
                logger.warning(f"No primary HTML content files found in Confluence export ZIP: {file_path}")
                return "No processable HTML content found in Confluence ZIP."

            logger.info(f"Found {len(html_files_to_process)} HTML files to process in {file_path}: {html_files_to_process}")

            for html_file_name in sorted(html_files_to_process): # Sort for consistent ordering
                try:
                    with zip_ref.open(html_file_name) as html_file:
                        # Read file content, decode it (assuming UTF-8, common for Confluence)
                        # Add error handling for decoding if necessary
                        html_content_bytes = html_file.read()
                        try:
                            html_content_str = html_content_bytes.decode('utf-8')
                        except UnicodeDecodeError:
                            logger.warning(f"UTF-8 decode failed for {html_file_name} in {file_path}, trying latin-1.")
                            html_content_str = html_content_bytes.decode('latin-1', errors='replace')

                        soup = BeautifulSoup(html_content_str, 'html.parser')

                        # Attempt to find main content area if known (e.g., Confluence specific divs)
                        main_content_area = soup.find(id="main-content") or soup.find(class_="wiki-content") or soup.find("body")
                        if main_content_area:
                            page_text = main_content_area.get_text(separator='\n', strip=True)
                        else:
                            page_text = soup.get_text(separator='\n', strip=True) # Fallback to full text

                        if page_text.strip():
                            all_text_content.append(f"\n\n--- Page: {html_file_name} ---\n{page_text}")
                except Exception as e_page:
                    logger.error(f"Error processing HTML file '{html_file_name}' within ZIP '{file_path}': {e_page}")

        if not all_text_content:
            return f"No text successfully extracted from HTML files in Confluence ZIP: {os.path.basename(file_path)}"

        return "\n".join(all_text_content).strip()

    except zipfile.BadZipFile:
        logger.error(f"File {file_path} is not a valid ZIP file or is corrupted.")
        return f"Error: Corrupted or invalid ZIP file for Confluence export {os.path.basename(file_path)}."
    except Exception as e:
        logger.error(f"General error processing Confluence export ZIP {file_path}: {e}", exc_info=True)
        return f"Error processing Confluence export {os.path.basename(file_path)}."

def process_eml(file_path: str) -> str:
    """Processes EML (email) files using Python's email library."""
    try:
        with open(file_path, 'rb') as fp:
            # Use BytesParser with a policy for modern email handling
            msg = BytesParser(policy=policy.default).parse(fp)

        text_content = []

        # Extract common headers
        subject = msg.get('subject', 'No Subject')
        from_ = msg.get('from', 'Unknown Sender')
        to_ = msg.get('to', 'Unknown Recipient')
        date_ = msg.get('date', 'Unknown Date')

        text_content.append(f"Subject: {subject}")
        text_content.append(f"From: {from_}")
        text_content.append(f"To: {to_}")
        text_content.append(f"Date: {date_}")
        text_content.append("\n--- Body ---")

        body_found = False
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get('Content-Disposition'))

                if "attachment" not in content_disposition:
                    if content_type == "text/plain":
                        try:
                            payload = part.get_payload(decode=True)
                            charset = part.get_content_charset() or 'utf-8' # Default to utf-8
                            text_content.append(payload.decode(charset, errors='replace'))
                            body_found = True
                        except Exception as e:
                            logger.warning(f"Could not decode text/plain part in EML {file_path} with charset {part.get_content_charset()}: {e}")
                    elif content_type == "text/html":
                        try:
                            payload = part.get_payload(decode=True)
                            charset = part.get_content_charset() or 'utf-8'
                            soup = BeautifulSoup(payload.decode(charset, errors='replace'), 'html.parser')
                            html_text = soup.get_text(separator='\n', strip=True)
                            if html_text and not body_found: # Prefer plain text if already found
                                text_content.append(html_text)
                                body_found = True # Consider HTML body found
                        except Exception as e:
                            logger.warning(f"Could not decode text/html part in EML {file_path} with charset {part.get_content_charset()}: {e}")
                else:
                    # Log attachment
                    filename = part.get_filename()
                    if filename:
                        logger.info(f"Attachment found in EML {file_path}: {filename} (Type: {part.get_content_type()}). Processing not yet implemented.")
                        # TODO: Save attachment and queue for recursive processing if supported type
        else:
            # Not multipart, try to get body directly
            try:
                payload = msg.get_payload(decode=True)
                charset = msg.get_content_charset() or 'utf-8'
                text_content.append(payload.decode(charset, errors='replace'))
            except Exception as e:
                 logger.error(f"Could not decode payload for non-multipart EML {file_path}: {e}")

        return "\n".join(text_content)
    except Exception as e:
        logger.error(f"Error processing EML file {file_path}: {e}")
        return ""

def process_msg(file_path: str) -> str:
    """Processes MSG (Outlook email) files using extract_msg library."""
    try:
        msg = extract_msg.Message(file_path)
        text_content = []

        text_content.append(f"Subject: {msg.subject or 'No Subject'}")
        text_content.append(f"From: {msg.sender or 'Unknown Sender'}")
        text_content.append(f"To: {msg.to or 'Unknown Recipient'}")
        text_content.append(f"Date: {msg.date or 'Unknown Date'}")
        if msg.cc:
            text_content.append(f"CC: {msg.cc}")
        if msg.bcc:
             text_content.append(f"BCC: {msg.bcc}")
        text_content.append("\n--- Body ---")
        text_content.append(msg.body or "No Body")

        if msg.attachments:
            text_content.append("\n--- Attachments ---")
            for i, attachment in enumerate(msg.attachments):
                # The attachment object itself might be a Message instance if it's an attached email
                if isinstance(attachment, extract_msg.Message):
                     filename = getattr(attachment, 'filename', f"attached_email_{i}.msg")
                     logger.info(f"Attached email found in MSG {file_path}: {filename}. Processing not yet implemented recursively here.")
                     text_content.append(f"Attached Email: {filename} (Further processing needed)")

                elif hasattr(attachment, 'longFilename') and attachment.longFilename:
                    filename = attachment.longFilename
                    logger.info(f"Attachment found in MSG {file_path}: {filename} (Type: {getattr(attachment, 'type', 'unknown')}). Processing not yet implemented.")
                    text_content.append(f"Attachment: {filename}")
                elif hasattr(attachment, 'shortFilename') and attachment.shortFilename: # Fallback for filename
                    filename = attachment.shortFilename
                    logger.info(f"Attachment found in MSG {file_path}: {filename} (Type: {getattr(attachment, 'type', 'unknown')}). Processing not yet implemented.")
                    text_content.append(f"Attachment: {filename}")

                # TODO: Save attachment (attachment.data) and queue for recursive processing
                # Example:
                # Path(temp_dir_for_attachments / filename).write_bytes(attachment.data)

        return "\n".join(text_content)
    except Exception as e:
        logger.error(f"Error processing MSG file {file_path}: {e}")
        return ""

def process_cad_metadata(file_path: str) -> str:
    """
    Processes DXF CAD files to extract metadata and text entities using ezdxf.
    For other CAD formats, this would need different libraries.
    """
    file_path_obj = Path(file_path)
    if file_path_obj.suffix.lower() != ".dxf":
        logger.warning(f"CAD processing: {file_path_obj.name} is not a .dxf file. This handler primarily supports DXF. Returning basic stats.")
        try:
            stat = os.stat(file_path)
            return (f"CAD File: {os.path.basename(file_path)}\n"
                    f"Format: {file_path_obj.suffix.upper()} (Limited support - basic stats only)\n"
                    f"Size: {stat.st_size} bytes")
        except Exception as e:
            logger.error(f"Error getting basic metadata for non-DXF CAD file {file_path}: {e}")
            return ""

    try:
        import ezdxf
        doc = ezdxf.readfile(file_path)
        msp = doc.modelspace()

        metadata_content = [
            f"CAD Document: {os.path.basename(file_path)}",
            f"DXF Version: {doc.dxfversion}",
            f"File Encoding: {doc.encoding}",
        ]

        # Extract header variables (example: drawing limits, last saved by)
        if "$LIMMIN" in doc.header:
            metadata_content.append(f"Drawing Limits Min: {doc.header['$LIMMIN']}")
        if "$LIMMAX" in doc.header:
            metadata_content.append(f"Drawing Limits Max: {doc.header['$LIMMAX']}")
        if "$LASTSAVEDBY" in doc.header:
             metadata_content.append(f"Last Saved By: {doc.header['$LASTSAVEDBY']}")


        # Extract layer names
        layer_names = [layer.dxf.name for layer in doc.layers if layer.dxf.name.lower() not in ["0", "defpoints"]]
        if layer_names:
            metadata_content.append("\n--- Layers ---")
            metadata_content.append(", ".join(sorted(list(set(layer_names)))))

        # Extract block names
        block_names = [block.name for block in doc.blocks if not block.is_any_anonymous and not block.is_layout_block()]
        if block_names:
            metadata_content.append("\n--- Blocks ---")
            metadata_content.append(", ".join(sorted(list(set(block_names)))))

        # Extract text entities
        text_entities = []
        for text_entity in msp.query('TEXT MTEXT'): # TEXT and MTEXT entities
            text_value = ""
            if text_entity.dxftype() == 'TEXT':
                text_value = text_entity.dxf.text
            elif text_entity.dxftype() == 'MTEXT':
                text_value = text_entity.plain_text() # MTEXT stores text in a more complex way

            if text_value.strip():
                text_entities.append(text_value.strip())

        if text_entities:
            metadata_content.append("\n--- Text Content ---")
            # Join unique text entities to avoid excessive repetition if many are identical
            unique_texts = sorted(list(set(text_entities)))
            metadata_content.extend(unique_texts)

        logger.info(f"Successfully extracted metadata and text from DXF file: {file_path}")
        return "\n".join(metadata_content)

    except ImportError:
        logger.error("ezdxf library is not installed. Cannot process DXF files.")
        return f"Error: ezdxf library not installed. Cannot process DXF: {os.path.basename(file_path)}"
    except ezdxf.DXFStructureError as e:
        logger.error(f"DXF structure error in file {file_path}: {e}")
        return f"Error: DXF structure error in {os.path.basename(file_path)}."
    except Exception as e:
        logger.error(f"Error processing DXF file {file_path}: {e}")
        return f"Error processing DXF file {os.path.basename(file_path)}."

# --- Audio/Video Processing ---
def transcribe_audio_video(file_path: str, whisper_model_name: str = "base", device: Optional[str] = None) -> Dict[str, Any]:
    """
    Transcribes audio/video file using Whisper and performs speaker diarization using pyannote.audio.
    Returns a dictionary containing the full transcript and a list of segments with speaker info.
    Requires HF_TOKEN environment variable for pyannote.audio models.
    """
    import whisper
    from pyannote.audio import Pipeline as DiarizationPipeline
    import torch
    import torchaudio # Required by pyannote for audio loading
    from pydub import AudioSegment # For converting various audio/video to WAV for pyannote
    import tempfile

    logger.info(f"Starting transcription and diarization for: {file_path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN environment variable not set. Speaker diarization might fail if model requires auth.")
        # Some pyannote models might work without it, but many require agreement/token.

    # Create a temporary WAV file for diarization, as pyannote works best with WAV
    # and Whisper can handle more formats directly.
    temp_wav_file = None
    try:
        # 1. Convert input to WAV for pyannote using pydub
        audio_input = AudioSegment.from_file(file_path)
        # Create a named temporary file with .wav suffix
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            temp_wav_file = tmp_wav.name
        audio_input.export(temp_wav_file, format="wav")
        logger.info(f"Converted {file_path} to temporary WAV: {temp_wav_file}")


        # 2. Speaker Diarization with pyannote.audio
        diarization_pipeline = None
        diarization_result = None
        try:
            # Using a common diarization pipeline from pyannote
            # This specific model might require accepting terms on Hugging Face Hub
            diarization_pipeline = DiarizationPipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", # Updated model
                use_auth_token=hf_token # Token can be True to use cached token or the string token itself
            )
            if device == "cuda": diarization_pipeline.to(torch.device("cuda"))

            logger.info(f"Running speaker diarization on {temp_wav_file}...")
            # Diarization pipeline expects a dict with 'uri' and 'audio' (path)
            diarization_input = {"uri": os.path.basename(file_path), "audio": temp_wav_file}
            diarization_result = diarization_pipeline(diarization_input)
            logger.info(f"Diarization complete for {file_path}.")
        except Exception as e_diarize:
            logger.error(f"Speaker diarization failed for {file_path}: {e_diarize}. Proceeding with transcription only.")
            diarization_result = None # Ensure it's None if diarization fails

        # 3. Transcription with Whisper
        logger.info(f"Loading Whisper model: {whisper_model_name}")
        model = whisper.load_model(whisper_model_name, device=device)
        logger.info(f"Transcribing {file_path} with Whisper...")
        # Whisper can often handle the original file path directly
        transcription_result = model.transcribe(file_path, verbose=False) # verbose=False for cleaner logs
        full_transcript = transcription_result["text"]
        segments = transcription_result["segments"] # List of dicts with 'start', 'end', 'text'
        logger.info(f"Transcription complete for {file_path}.")

        # 4. Align transcription with diarization (if diarization was successful)
        # This is a complex step. A common approach is to map each transcribed segment
        # to the speaker who was most active during that segment's timeframe.
        output_segments = []
        if diarization_result:
            logger.info("Aligning transcription with speaker diarization...")
            for seg in segments:
                start_time = seg["start"]
                end_time = seg["end"]
                segment_text = seg["text"]

                # Find speaker for this segment's midpoint or majority overlap
                # This is a simplified alignment; more advanced methods exist.
                mid_time = (start_time + end_time) / 2
                active_speaker = "UNKNOWN_SPEAKER"

                # Iterate through pyannote's speaker turns
                for turn, _, speaker_label in diarization_result.itertracks(yield_label=True):
                    if turn.start <= mid_time < turn.end:
                        active_speaker = speaker_label
                        break
                output_segments.append({
                    "start": start_time,
                    "end": end_time,
                    "speaker": active_speaker,
                    "text": segment_text.strip()
                })
            logger.info("Alignment complete.")
        else: # No diarization, just return Whisper segments
            for seg in segments:
                output_segments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "speaker": "UNKNOWN_SPEAKER", # Default if no diarization
                    "text": seg["text"].strip()
                })

        # Construct a readable combined transcript with speaker labels
        combined_transcript_with_speakers = ""
        current_speaker = None
        for seg_info in output_segments:
            if current_speaker != seg_info["speaker"]:
                if combined_transcript_with_speakers: combined_transcript_with_speakers += "\n"
                combined_transcript_with_speakers += f"[{seg_info['speaker']}] "
                current_speaker = seg_info["speaker"]
            combined_transcript_with_speakers += seg_info["text"] + " "

        final_text_output = combined_transcript_with_speakers.strip()
        if not final_text_output: # Fallback if alignment produced nothing but full transcript exists
            final_text_output = f"[UNKNOWN_SPEAKER] {full_transcript}"


        return {
            "full_transcript": full_transcript,
            "transcript_with_speakers": final_text_output,
            "segments": output_segments,
            "language": transcription_result.get("language", "unknown")
        }

    except ImportError as e:
        logger.error(f"Missing libraries for audio/video processing ({file_path}): {e}. Please install openai-whisper, pyannote.audio, torch, torchaudio, and pydub.")
        return {"error": f"Missing libraries: {e}"}
    except Exception as e:
        logger.error(f"Error processing audio/video file {file_path}: {e}", exc_info=True)
        return {"error": str(e)}
    finally:
        if temp_wav_file and os.path.exists(temp_wav_file):
            try:
                os.remove(temp_wav_file)
                logger.info(f"Cleaned up temporary WAV file: {temp_wav_file}")
            except Exception as e_clean:
                logger.error(f"Error cleaning up temp WAV file {temp_wav_file}: {e_clean}")


def identify_speakers(file_path: str) -> str:
    """
    DEPRECATED: Speaker identification is now integrated into transcribe_audio_video.
    This function is kept for placeholder compatibility but should not be used directly.
    """
    logger.warning("identify_speakers is deprecated. Use transcribe_audio_video which includes diarization.")
    # For demonstration, if you needed just diarization (though it's usually paired with transcription):
    # from pyannote.audio import Pipeline
    # hf_token = os.environ.get("HF_TOKEN")
    # pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=hf_token)
    # diarization = pipeline(file_path)
    # output = ""
    # for turn, _, speaker in diarization.itertracks(yield_label=True):
    #     output += f"Speaker {speaker} from {turn.start:.1f}s to {turn.end:.1f}s.\n"
    # return output if output else "No speaker diarization data."
    return f"Speaker diarization for {os.path.basename(file_path)} (Placeholder - functionality moved to transcribe_audio_video)"

# --- Advanced PDF feature placeholders ---
def extract_tables_from_pdf(file_path: str) -> str:
    """
    Extracts tables from a PDF file using camelot-py.
    Returns a string representation of the extracted tables (e.g., Markdown or CSV).
    """
    try:
        import camelot
        logger.info(f"Attempting table extraction from PDF: {file_path} using Camelot.")
        # Process all pages. 'lattice' is good for tables with clear grid lines.
        # 'stream' can be used for tables without clear lines but might be less accurate.
        tables = camelot.read_pdf(file_path, pages='all', flavor='lattice', suppress_stdout=True)

        if not tables or tables.n == 0:
            logger.info(f"No tables found by Camelot (lattice) in {file_path}. Trying 'stream' strategy.")
            tables = camelot.read_pdf(file_path, pages='all', flavor='stream', suppress_stdout=True)

        if not tables or tables.n == 0:
            logger.info(f"No tables found by Camelot in {file_path} with either strategy.")
            return ""

        all_tables_text = [f"\n\n--- Extracted Table {i+1} (Page {table.page}) ---\n"]
        for i, table in enumerate(tables):
            # Choose a representation for the table, e.g., Markdown or CSV
            # Markdown is often more human-readable in a text dump.
            try:
                # Ensure headers are sensible if possible
                df = table.df
                # A simple heuristic: if first row looks like headers (more strings, less numbers)
                # This is very basic, proper header detection is complex.
                # For now, just use default to_markdown behavior.
                table_markdown = df.to_markdown(index=False)
                all_tables_text.append(table_markdown)
                logger.info(f"Extracted table {i+1} from page {table.page} of {file_path} (Accuracy: {table.accuracy:.2f}%).")
            except Exception as e_df:
                logger.warning(f"Could not convert table {i+1} from {file_path} to markdown: {e_df}")
                all_tables_text.append(f"[Could not parse table {i+1} data from page {table.page}]")

        return "\n".join(all_tables_text)

    except ImportError:
        logger.error("camelot-py library is not installed. Cannot extract tables from PDF.")
        return "Error: camelot-py library not installed for PDF table extraction."
    except Exception as e:
        # Camelot can sometimes fail on complex PDFs or if Ghostscript is not properly installed.
        logger.error(f"Error during table extraction from PDF {file_path} with Camelot: {e}")
        return f"Error extracting tables from {os.path.basename(file_path)}: {e}"

def recognize_charts_from_pdf(file_path: str) -> str:
    """
    Placeholder for chart recognition from PDF.
    This is a highly complex Computer Vision and OCR task. A full implementation
    would require specialized models (e.g., object detection for charts, chart type
    classification, OCR for axes/legends, and logic to extract data points).
    """
    logger.warning(f"Chart recognition from PDF {file_path} is a conceptual placeholder.")

    # Conceptual Steps for a full implementation:
    # 1. Convert PDF pages to high-resolution images.
    # 2. Use an object detection model (e.g., trained on document elements) to find chart bounding boxes.
    # 3. For each detected chart:
    #    a. Crop the chart image.
    #    b. Classify chart type (e.g., bar, line, pie, scatter).
    #    c. Apply type-specific logic:
    #       - OCR axis labels, titles, legends.
    #       - For bar/line charts: Detect bars/lines, estimate their values relative to axes.
    #       - For pie charts: Detect slices, OCR legend to get categories, estimate percentages.
    #    d. Structure the extracted data (e.g., as JSON or CSV representation of the chart's data).
    #
    # A simpler, partial approach (leveraging existing PDF image OCR):
    # If charts are embedded as images within the PDF, the main PDF processing
    # (which extracts images and OCRs them via ImageProcessor) might pick up
    # textual elements like chart titles, axis labels, and legend text.
    # This wouldn't extract the underlying data points or structure but could provide some context.

    # For now, this function returns a placeholder string.
    # The actual text from chart images might be captured during the PDF's image OCR pass
    # if self.config.extract_images and self.config.ocr_enabled are true in DocumentProcessor.
    return f"Chart data extraction from PDF {os.path.basename(file_path)} is not yet implemented. Textual elements within chart images might be captured by general PDF OCR."

def process_webp_svg_ocr(file_path: str) -> str:
    """Placeholder for WEBP/SVG OCR."""
    # For SVG, might need to convert to raster (e.g. PNG) first using cairosvg or similar, then OCR.
    # For WEBP, Pillow should handle it if system libraries are present.
    logger.warning(f"WEBP/SVG OCR for {file_path} is a placeholder. SVG would need rasterization. WEBP depends on Pillow capabilities.")
    # from PIL import Image
    # import pytesseract
    # try:
    #     img = Image.open(file_path)
    #     # If SVG, might need img.load() if it's a vector format that Pillow can rasterize via a delegate
    #     text = pytesseract.image_to_string(img)
    #     return text
    # except Exception as e:
    #     logger.error(f"Error in placeholder WEBP/SVG OCR for {file_path}: {e}")
    return f"OCR Text from {os.path.basename(file_path)} (Placeholder for WEBP/SVG)"

# Example of how an archive might be processed recursively
# This would be part of the main DocumentProcessor class
# def _process_archive_recursively(self, file_path: Path, temp_dir_provider_fn: callable):
#     extracted_content_texts = []
#     extracted_files = process_archive(str(file_path), temp_dir_provider_fn)
#     for extracted_file_path_str in extracted_files:
#         extracted_file_path = Path(extracted_file_path_str)
#         ext = extracted_file_path.suffix.lower()
#         if ext in self.supported_extensions: # Assuming self.supported_extensions is available
#             try:
#                 # If it's another archive, recurse
#                 if ext in [".zip", ".tar", ".gz", ".bz2", ".xz"]: # Add .rar, .7z if supported
#                     extracted_content_texts.extend(self._process_archive_recursively(extracted_file_path, temp_dir_provider_fn))
#                 else:
#                     processor_func = self.supported_extensions[ext]
#                     extracted_content_texts.append(processor_func(extracted_file_path))
#             except Exception as e:
#                 logger.error(f"Error processing extracted file {extracted_file_path}: {e}")
#         else:
#             logger.info(f"Skipping unsupported file type in archive: {extracted_file_path}")
#     # Clean up the specific temp directory for this archive after processing all its files
#     # This needs careful handling of paths if extract_to_temp_dir_fn creates nested dirs.
#     # For now, assume the main processor handles temp dir cleanup.
#     return extracted_content_texts

```
