import cv2
import numpy as np
from datetime import datetime
from passporteye import read_mrz
import logging
from typing import Optional, Dict, Tuple
import os
import time
from io import BytesIO
from PIL import Image

class GhanaMRZService:
    """Service for processing MRZ data from Ghana Cards with improved extraction and formatting"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.DEBUG)  # Set to DEBUG for more detailed logs

    def preprocess_image(self, image_bytes: bytes) -> bytes:
        """
        Preprocess the image to improve MRZ detection using enhanced techniques
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            bytes: Preprocessed image bytes
        """
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image")

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply adaptive thresholding with optimized parameters
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 85, 11
            )

            # Optional: Apply additional preprocessing steps
            # Denoise the image
            denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)

            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)

            # Convert back to bytes
            success, buffer = cv2.imencode('.jpg', enhanced)
            if not success:
                raise ValueError("Could not encode preprocessed image")
                
            return buffer.tobytes()

        except Exception as e:
            self.logger.error(f"Error preprocessing image: {str(e)}")
            return image_bytes  # Return original image bytes if preprocessing fails

    def clean_mrz_name(self, name: str) -> str:
        """
        Clean MRZ-extracted name by removing common MRZ artifacts while preserving
        actual name components
        
        Args:
            name: Raw name string from MRZ
            
        Returns:
            str: Cleaned name string
        """
        # Split into words and remove artifacts
        words = name.split()
        cleaned_words = []
        
        for word in words:
            # Only keep words that have at least one character that isn't K or <
            if any(c not in 'K<' for c in word):
                # Clean individual word
                cleaned_word = ''.join(c for c in word if c not in 'K<')
                if cleaned_word:
                    cleaned_words.append(cleaned_word)
        
        cleaned_name = ' '.join(cleaned_words)
        self.logger.debug(f"Cleaned name from '{name}' to '{cleaned_name}'")
        return cleaned_name

    def format_dates(self, date_of_birth: str, expiry_date: str) -> Tuple[str, str]:
        """
        Format dates from MRZ data into a more readable format with improved validation
        
        Args:
            date_of_birth: Raw DOB string from MRZ
            expiry_date: Raw expiry date string from MRZ
            
        Returns:
            Tuple[str, str]: Formatted birth date and expiry date
        """
        def clean_date_string(date_str: str) -> Optional[str]:
            if not date_str:
                return None
            # Remove MRZ filler characters and spaces
            cleaned = date_str.replace('<', '').replace(' ', '')
            # Check if we have a valid 6-digit string
            if len(cleaned) != 6 or not cleaned.isdigit():
                return None
            return cleaned

        def convert_date_of_birth(date_str: str) -> str:
            cleaned_date = clean_date_string(date_str)
            if not cleaned_date:
                return "Invalid date format"
            
            try:
                year = cleaned_date[:2]
                month = cleaned_date[2:4]
                day = cleaned_date[4:6]
                
                # Additional validation for month and day
                if not (1 <= int(month) <= 12 and 1 <= int(day) <= 31):
                    return "Invalid date: month or day out of range"
                
                # For birth dates: if year is > 30, assume 19xx, else 20xx
                year_prefix = '19' if int(year) > 30 else '20'
                full_year = year_prefix + year
                
                date_obj = datetime.strptime(f"{full_year}{month}{day}", "%Y%m%d")
                
                formatted_date = date_obj.strftime("%d{} %B, %Y").format(
                    'th' if 11 <= date_obj.day <= 13 else 
                    {1: 'st', 2: 'nd', 3: 'rd'}.get(date_obj.day % 10, 'th')
                )
                
                self.logger.debug(f"Formatted birth date from {date_str} to {formatted_date}")
                return formatted_date
                
            except ValueError as e:
                self.logger.error(f"Error formatting birth date: {str(e)}")
                return "Invalid date format"

        def convert_expiry_date(date_str: str) -> str:
            cleaned_date = clean_date_string(date_str)
            if not cleaned_date:
                return "Invalid date format"
            
            try:
                year = cleaned_date[:2]
                month = cleaned_date[2:4]
                day = cleaned_date[4:6]
                
                # Additional validation for month and day
                if not (1 <= int(month) <= 12 and 1 <= int(day) <= 31):
                    return "Invalid date: month or day out of range"
                
                # For expiry dates: always assume 20xx
                full_year = '20' + year
                
                date_obj = datetime.strptime(f"{full_year}{month}{day}", "%Y%m%d")
                
                formatted_date = date_obj.strftime("%d{} %B, %Y").format(
                    'th' if 11 <= date_obj.day <= 13 else 
                    {1: 'st', 2: 'nd', 3: 'rd'}.get(date_obj.day % 10, 'th')
                )
                
                self.logger.debug(f"Formatted expiry date from {date_str} to {formatted_date}")
                return formatted_date
                
            except ValueError as e:
                self.logger.error(f"Error formatting expiry date: {str(e)}")
                return "Invalid date format"

        return convert_date_of_birth(date_of_birth), convert_expiry_date(expiry_date)

    async def process_mrz(self, image_bytes: bytes) -> Optional[Dict[str, str]]:
        """
        Process MRZ data from Ghana Card back image with improved extraction and validation
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Optional[Dict[str, str]]: Extracted and formatted MRZ data, or None if extraction fails
        """
        temp_path = None
        preprocessed_path = None
        
        try:
            self.logger.info("Starting MRZ processing...")
            
            # Save original and preprocessed images
            timestamp = int(time.time())
            temp_path = f"temp_ghana_card_{timestamp}.jpg"
            preprocessed_path = f"temp_ghana_card_{timestamp}_preprocessed.jpg"
            
            # Save original image
            with open(temp_path, 'wb') as f:
                f.write(image_bytes)
            
            # Save preprocessed image
            preprocessed_bytes = self.preprocess_image(image_bytes)
            with open(preprocessed_path, 'wb') as f:
                f.write(preprocessed_bytes)
            
            # Try both preprocessed and original images
            mrz_data = None
            
            for path in [preprocessed_path, temp_path]:
                try:
                    mrz = read_mrz(path)
                    if mrz is not None:
                        mrz_data = mrz.to_dict()
                        self.logger.info(f"Successfully read MRZ from {path}")
                        
                        # Log raw MRZ data for debugging
                        self.logger.debug("Raw MRZ data:")
                        for key, value in mrz_data.items():
                            self.logger.debug(f"{key}: {value}")
                            
                        break
                except Exception as e:
                    self.logger.warning(f"Failed to read MRZ from {path}: {str(e)}")
                    continue
            
            if mrz_data is None:
                self.logger.warning("No MRZ data detected in any image")
                return None
            
            # Format the dates
            formatted_dob, formatted_expiry = self.format_dates(
                mrz_data.get('date_of_birth'),
                mrz_data.get('expiration_date')
            )
            
            # Clean up the names field
            given_names = mrz_data.get('names', 'Unknown')
            given_names = self.clean_mrz_name(given_names)
            
            formatted_data = {
                'Document Type': mrz_data.get('type', 'Unknown').upper(),
                'Country Code': mrz_data.get('country', 'Unknown'),
                'Document Number': mrz_data.get('number', 'Unknown'),
                'Surname': self.clean_mrz_name(mrz_data.get('surname', 'Unknown')),
                'Given Names': given_names,
                'Nationality': mrz_data.get('nationality', 'Unknown'),
                'Date of Birth': formatted_dob,
                'Gender': mrz_data.get('sex', 'Unknown').upper(),
                'Expiry Date': formatted_expiry
            }
            
            # Log formatted data
            self.logger.debug("Formatted MRZ data:")
            for key, value in formatted_data.items():
                self.logger.debug(f"{key}: {value}")
                
            return formatted_data
            
        except Exception as e:
            self.logger.error(f"Error processing MRZ: {str(e)}")
            return None
            
        finally:
            # Clean up temporary files
            for path in [temp_path, preprocessed_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        self.logger.warning(f"Failed to remove temp file {path}: {str(e)}")

    def validate_mrz_data(self, mrz_data: Dict) -> bool:
        """
        Validate MRZ data with improved validation rules
        
        Args:
            mrz_data: Dictionary containing MRZ data
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        if not mrz_data:
            return False
        
        # Basic structure check
        required_fields = ['type', 'country', 'number', 'date_of_birth', 'expiration_date']
        if not all(field in mrz_data for field in required_fields):
            self.logger.warning("Missing required fields in MRZ data")
            return False
        
        # Validate document type and country code
        if mrz_data.get('type') != 'ID' or mrz_data.get('country') != 'GHA':
            self.logger.warning("Invalid document type or country code")
            return False
        
        # Validate dates
        try:
            dob = datetime.strptime(mrz_data['date_of_birth'], '%y%m%d')
            exp = datetime.strptime(mrz_data['expiration_date'], '%y%m%d')
            
            # Basic date validations
            if dob > datetime.now() or exp < datetime.now():
                self.logger.warning("Invalid date values in MRZ data")
                return False
                
        except ValueError:
            self.logger.warning("Invalid date format in MRZ data")
            return False
        
        # Document number format validation (basic)
        doc_num = mrz_data.get('number', '')
        if not doc_num or len(doc_num) < 8:
            self.logger.warning("Invalid document number format")
            return False
        
        return True

# Initialize the MRZ service instance
mrz_service = GhanaMRZService()

# Export the instance
__all__ = ['mrz_service']