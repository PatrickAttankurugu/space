import requests
import fitz  # PyMuPDF
import io
from PIL import Image
import logging
from typing import Optional, Tuple, List
import traceback
from pathlib import Path
import tempfile
import os
from app.core.config import settings

class ConversionError(Exception):
    """Base exception for file conversion errors"""
    def __init__(self, message: str, error_code: str, details: Optional[dict] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

class FileConverterService:
    def __init__(self):
        self.logger = logging.getLogger("FileConverterService")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # Supported formats
        self.IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.DOCUMENT_FORMATS = {'.pdf'}
        
    async def get_file_from_url(self, url: str) -> Tuple[bytes, str]:
        """Fetch file from URL and determine its format"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, timeout=10, headers=headers, verify=False)
            response.raise_for_status()
            
            # Try to determine format from response headers
            content_type = response.headers.get('Content-Type', '').lower()
            
            if 'image' in content_type:
                extension = self._get_image_extension(content_type)
            elif 'pdf' in content_type:
                extension = '.pdf'
            else:
                # Try to determine format from URL if header is ambiguous
                extension = Path(url).suffix.lower()
                if not extension:
                    # Try to determine format from file content
                    extension = self._detect_format_from_bytes(response.content)
            
            return response.content, extension
            
        except requests.exceptions.RequestException as e:
            raise ConversionError(
                f"Failed to fetch file from URL: {str(e)}",
                "URL_ACCESS_ERROR",
                {"url": url}
            )

    def _detect_format_from_bytes(self, content: bytes) -> str:
        """Detect file format from its content"""
        # Magic numbers for different file formats
        signatures = {
            b'%PDF': '.pdf',
            b'\xFF\xD8\xFF': '.jpg',
            b'\x89PNG\r\n\x1a\n': '.png',
            b'GIF87a': '.gif',
            b'GIF89a': '.gif',
            b'BM': '.bmp',
            b'II*\x00': '.tiff',
            b'MM\x00*': '.tiff',
        }
        
        for signature, ext in signatures.items():
            if content.startswith(signature):
                return ext
                
        raise ConversionError(
            "Could not determine file format",
            "UNKNOWN_FORMAT"
        )

    def _get_image_extension(self, content_type: str) -> str:
        """Get file extension from content type"""
        content_type_to_ext = {
            'image/jpeg': '.jpg',
            'image/png': '.png',
            'image/gif': '.gif',
            'image/bmp': '.bmp',
            'image/tiff': '.tiff',
            'image/webp': '.webp'
        }
        return content_type_to_ext.get(content_type, '')

    async def convert_to_images(self, file_content: bytes, file_format: str) -> List[Image.Image]:
        """Convert file content to list of PIL Images"""
        try:
            if file_format.lower() in self.IMAGE_FORMATS:
                return [Image.open(io.BytesIO(file_content))]
                
            elif file_format.lower() == '.pdf':
                return await self._pdf_to_images(file_content)
                
            else:
                raise ConversionError(
                    f"Unsupported file format: {file_format}",
                    "UNSUPPORTED_FORMAT"
                )
                
        except Exception as e:
            error_details = {
                "traceback": traceback.format_exc(),
                "error_type": type(e).__name__
            }
            raise ConversionError(
                f"Error converting file: {str(e)}",
                "CONVERSION_ERROR",
                error_details
            )

    async def _pdf_to_images(self, pdf_content: bytes) -> List[Image.Image]:
        """Convert PDF content to list of PIL Images"""
        try:
            # Create temporary file for PyMuPDF
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(pdf_content)
                tmp_path = tmp_file.name

            try:
                pdf_document = fitz.open(tmp_path)
                images = []

                for page_num in range(pdf_document.page_count):
                    page = pdf_document[page_num]
                    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 DPI
                    img_data = pix.tobytes()
                    
                    # Convert to PIL Image
                    img = Image.frombytes('RGB', [pix.width, pix.height], img_data)
                    images.append(img)

                return images

            finally:
                # Clean up
                if 'pdf_document' in locals():
                    pdf_document.close()
                os.unlink(tmp_path)

        except Exception as e:
            raise ConversionError(
                f"Error converting PDF to images: {str(e)}",
                "PDF_CONVERSION_ERROR"
            )

    async def normalize_image(self, image: Image.Image) -> Image.Image:
        """Normalize image for ML processing"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large
            if (image.width > settings.MAX_IMAGE_DIMENSION or 
                image.height > settings.MAX_IMAGE_DIMENSION):
                scale = min(
                    settings.MAX_IMAGE_DIMENSION/image.width,
                    settings.MAX_IMAGE_DIMENSION/image.height
                )
                new_size = (
                    int(image.width * scale),
                    int(image.height * scale)
                )
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Ensure minimum size
            if (image.width < settings.MIN_IMAGE_DIMENSION or 
                image.height < settings.MIN_IMAGE_DIMENSION):
                raise ConversionError(
                    "Image dimensions too small",
                    "INVALID_DIMENSIONS",
                    {
                        "width": image.width,
                        "height": image.height,
                        "min_dimension": settings.MIN_IMAGE_DIMENSION
                    }
                )
            
            return image
            
        except ConversionError:
            raise
        except Exception as e:
            raise ConversionError(
                f"Error normalizing image: {str(e)}",
                "NORMALIZATION_ERROR"
            )

    async def process_file_url(self, url: str) -> List[Image.Image]:
        """Main method to process file URL and convert to images"""
        try:
            self.logger.info(f"Processing file from URL: {url}")
            
            # Get file content and format
            file_content, file_format = await self.get_file_from_url(url)
            
            # Convert to images
            images = await self.convert_to_images(file_content, file_format)
            
            # Normalize each image
            normalized_images = []
            for img in images:
                normalized = await self.normalize_image(img)
                normalized_images.append(normalized)
            
            self.logger.info(
                f"Successfully processed file. "
                f"Format: {file_format}, "
                f"Number of images: {len(normalized_images)}"
            )
            
            return normalized_images
            
        except Exception as e:
            self.logger.error(f"Error processing file: {str(e)}")
            raise