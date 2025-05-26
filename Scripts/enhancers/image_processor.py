from typing import List, Dict, Any, Optional, Tuple
import pytesseract
from PIL import Image
import pdf2image
import numpy as np
import cv2
import logging
from pathlib import Path
import tempfile
import os
from dataclasses import dataclass
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class OCRConfig:
    use_gpu: bool = True
    enhance_image: bool = True
    use_transformers: bool = True
    language: str = 'eng'
    batch_size: int = 4
    dpi: int = 300
    threshold_method: str = 'adaptive'  # 'simple' or 'adaptive'
    chunk_size: int = 1000  # Default chunk size for text segmentation
    min_confidence: float = 0.6  # Minimum confidence score for OCR results
    image_enhancement: Dict[str, Any] = None  # Image enhancement parameters
    
    def __post_init__(self):
        if self.image_enhancement is None:
            self.image_enhancement = {
                'contrast': 1.5,
                'brightness': 1.2,
                'sharpen': True,
                'denoise': True,
                'deskew': True
            }

class ImageProcessor:
    def __init__(self, config: OCRConfig = None):
        self.config = config or OCRConfig()
        
        # Initialize Tesseract configuration
        if self.config.language != 'eng':
            pytesseract.pytesseract.tesseract_cmd = 'tesseract'
            
        # Initialize Transformer model if enabled
        if self.config.use_transformers:
            try:
                self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
                self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
                if self.config.use_gpu and torch.cuda.is_available():
                    self.model = self.model.to('cuda')
            except Exception as e:
                logger.warning(f"Failed to load TrOCR model: {e}")
                self.config.use_transformers = False

    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Apply various image enhancement techniques"""
        try:
            # Convert to grayscale if not already
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Apply threshold
            if self.config.threshold_method == 'adaptive':
                binary = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
            else:
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Denoise
            denoised = cv2.fastNlMeansDenoising(binary)

            # Deskew if needed
            coords = np.column_stack(np.where(denoised > 0))
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = 90 + angle
            if abs(angle) > 0.5:  # Only rotate if skew is significant
                (h, w) = denoised.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                denoised = cv2.warpAffine(
                    denoised, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )

            return denoised
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return image

    def extract_text_from_image(self, image: Image.Image) -> str:
        """Extract text from a single image using OCR"""
        try:
            # Convert PIL Image to numpy array
            img_array = np.array(image)
            
            # Enhance image if configured
            if self.config.enhance_image:
                img_array = self.enhance_image(img_array)

            # Use Transformer-based OCR if enabled
            if self.config.use_transformers:
                try:
                    pixel_values = self.processor(image, return_tensors="pt").pixel_values
                    if self.config.use_gpu and torch.cuda.is_available():
                        pixel_values = pixel_values.to('cuda')
                    generated_ids = self.model.generate(pixel_values)
                    generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    return generated_text
                except Exception as e:
                    logger.warning(f"Transformer OCR failed, falling back to Tesseract: {e}")

            # Fall back to Tesseract
            return pytesseract.image_to_string(
                img_array,
                lang=self.config.language,
                config='--psm 1'  # Automatic page segmentation with OSD
            )
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return ""

    def process_pdf_images(self, pdf_path: str) -> List[Tuple[int, str]]:
        """Extract images and text from PDF pages"""
        try:
            # Convert PDF to images
            images = pdf2image.convert_from_path(
                pdf_path,
                dpi=self.config.dpi,
                thread_count=os.cpu_count()
            )
            
            # Process images in parallel
            results = []
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                future_to_page = {
                    executor.submit(self.extract_text_from_image, img): i 
                    for i, img in enumerate(images, 1)
                }
                for future in concurrent.futures.as_completed(future_to_page):
                    page_num = future_to_page[future]
                    try:
                        text = future.result()
                        if text.strip():
                            results.append((page_num, text))
                    except Exception as e:
                        logger.error(f"Error processing page {page_num}: {e}")
            
            return sorted(results, key=lambda x: x[0])
        except Exception as e:
            logger.error(f"PDF image extraction failed: {e}")
            return []

    def batch_process_images(self, image_paths: List[str]) -> Dict[str, str]:
        """Process multiple images in parallel"""
        results = {}
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_path = {
                executor.submit(self.process_single_image, path): path 
                for path in image_paths
            }
            for future in concurrent.futures.as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    text = future.result()
                    if text.strip():
                        results[path] = text
                except Exception as e:
                    logger.error(f"Error processing image {path}: {e}")
                    results[path] = ""
        return results

    def process_single_image(self, image_path: str) -> str:
        """Process a single image file"""
        try:
            with Image.open(image_path) as img:
                return self.extract_text_from_image(img)
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return ""

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Advanced image preprocessing pipeline."""
        try:
            # Convert to numpy array for OpenCV operations
            img_array = np.array(image)

            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            # Denoise
            img_array = cv2.fastNlMeansDenoising(img_array)

            # Adaptive thresholding
            if self.config.threshold_method == 'adaptive':
                img_array = cv2.adaptiveThreshold(
                    img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2
                )
            else:
                _, img_array = cv2.threshold(
                    img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )

            # Deskew if needed
            if self.config.image_enhancement.get('deskew', True):
                angle = self._get_skew_angle(img_array)
                if abs(angle) > 0.5:  # Only deskew if angle is significant
                    img_array = self._rotate_image(img_array, angle)

            # Border removal
            img_array = self._remove_borders(img_array)

            # Convert back to PIL Image
            return Image.fromarray(img_array)

        except Exception as e:
            logger.error(f"Error in image preprocessing: {str(e)}")
            return image  # Return original image if preprocessing fails

    def _get_skew_angle(self, image: np.ndarray) -> float:
        """Detect skew angle of the image."""
        # Find all non-zero points
        coords = np.column_stack(np.where(image > 0))
        
        if len(coords) < 20:  # Not enough points for reliable angle detection
            return 0.0
            
        # Find the angle using PCA
        angle = cv2.minAreaRect(coords)[-1]
        
        # Adjust angle
        if angle < -45:
            angle = 90 + angle
        
        return angle

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate the image by the given angle."""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new image dimensions
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])
        new_width = int(height * abs_sin + width * abs_cos)
        new_height = int(height * abs_cos + width * abs_sin)
        
        # Adjust the rotation matrix
        rotation_matrix[0, 2] += new_width / 2 - center[0]
        rotation_matrix[1, 2] += new_height / 2 - center[1]
        
        # Perform the rotation
        rotated = cv2.warpAffine(
            image, rotation_matrix, (new_width, new_height),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated

    def _remove_borders(self, image: np.ndarray) -> np.ndarray:
        """Remove black borders from the image."""
        # Find all non-zero points
        coords = np.nonzero(image)
        
        # Find the bounding box
        y_min, y_max = np.min(coords[0]), np.max(coords[0])
        x_min, x_max = np.min(coords[1]), np.max(coords[1])
        
        # Add small padding
        padding = 5
        y_min = max(0, y_min - padding)
        y_max = min(image.shape[0], y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(image.shape[1], x_max + padding)
        
        # Crop the image
        return image[y_min:y_max, x_min:x_max]
