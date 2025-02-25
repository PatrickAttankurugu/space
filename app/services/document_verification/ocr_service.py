from typing import Dict, Optional, Any, Tuple, Union
import logging
import re
import numpy as np
from PIL import Image, ImageEnhance
from io import BytesIO
import cv2
import os
import tempfile
from app.utils.device_utils import get_optimal_device
from app.core.config import settings
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import requests

class GhanaCardOCR:
    """OCR service for Ghana Card information extraction using DocTR"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = get_optimal_device()
        self.initialized = False
        self.model = None
        
    def initialize(self):
        """Initialize OCR service with DocTR model"""
        if not self.initialized:
            try:
                self.logger.info("Loading OCR model...")
                self.model = ocr_predictor(pretrained=True)
                self.initialized = True
                self.logger.info("OCR service initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize OCR service: {str(e)}")
                self.initialized = False
                raise RuntimeError(f"OCR initialization failed: {str(e)}")
    
    def preprocess_image(self, image_path: str) -> str:
        """
        Preprocess the image to improve OCR quality
        Returns the path to the preprocessed image
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Failed to read image from {image_path}")
                return image_path
                
            # Get image dimensions
            height, width = image.shape[:2]
            self.logger.info(f"Original image dimensions: {width}x{height}")
            
            # Convert to PIL for some processing
            pil_image = Image.open(image_path).convert('RGB')
            
            # Enhance contrast and sharpness
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(1.5)
            enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer.enhance(1.5)
            
            # Ensure minimum dimensions - docTR needs at least 32x32
            min_dimension = max(224, settings.MIN_IMAGE_DIMENSION)
            
            # If image is too small, resize maintaining aspect ratio
            if width < min_dimension or height < min_dimension:
                self.logger.warning(f"Image dimensions too small, resizing to minimum {min_dimension}px")
                
                # Calculate new dimensions
                if width < height:
                    new_width = min_dimension
                    new_height = int(height * (min_dimension / width))
                else:
                    new_height = min_dimension
                    new_width = int(width * (min_dimension / height))
                
                self.logger.info(f"Resizing to {new_width}x{new_height}")
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save the preprocessed image
            output_path = os.path.join(os.path.dirname(image_path), 'preprocessed_' + os.path.basename(image_path))
            pil_image.save(output_path, quality=95)
            self.logger.info(f"Saved preprocessed image to {output_path}")
            
            return output_path
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {str(e)}")
            return image_path  # Return original image path on error
    
    def extract_card_info(self, file_path: str) -> Dict[str, Any]:
        """Extract Ghana Card information using DocTR from a file path"""
        # Initialize model if not already initialized
        if not self.initialized:
            self.initialize()
        
        if not os.path.exists(file_path):
            self.logger.error(f"Error: Image file not found at {file_path}")
            return {
                'success': False,
                'message': f"Error: Image file not found at {file_path}",
                'card_info': None
            }
        
        try:
            # Preprocess the image
            processed_path = self.preprocess_image(file_path)
            
            # Log file sizes
            original_size = os.path.getsize(file_path) / 1024
            processed_size = os.path.getsize(processed_path) / 1024
            self.logger.info(f"Original file: {original_size:.2f}KB, Processed file: {processed_size:.2f}KB")
            
            # Process the image using docTR
            self.logger.info(f"Processing image with DocTR: {processed_path}")
            
            try:
                # Use docTR DocumentFile to handle the image properly
                doc = DocumentFile.from_images(processed_path)
                result = self.model(doc)
                extracted_text = result.export()
                
                # Clean up temporary preprocessed file if different from original
                if processed_path != file_path:
                    try:
                        os.unlink(processed_path)
                    except Exception as e:
                        self.logger.warning(f"Could not remove temporary file: {str(e)}")
            except Exception as e:
                self.logger.error(f"DocTR processing failed: {str(e)}")
                # Fallback to original file if preprocessing caused issues
                self.logger.info("Falling back to original file")
                doc = DocumentFile.from_images(file_path)
                result = self.model(doc)
                extracted_text = result.export()
            
            # Extract all text lines
            all_lines = []
            for page in extracted_text['pages']:
                for block in page['blocks']:
                    for line in block['lines']:
                        line_text = ' '.join([word['value'] for word in line['words']])
                        all_lines.append(line_text)
                        self.logger.debug(f"Detected line: {line_text}")

            # Check for Ghana Card indicators
            ghana_card_indicators = ['ECOWAS', 'IDENTITY', 'GHANA', 'CEDEAO', 'GHA-', 'REPUBLIC']
            found_indicators = []
            for indicator in ghana_card_indicators:
                for line in all_lines:
                    if indicator in line:
                        found_indicators.append(indicator)
                        self.logger.info(f"Found indicator: {indicator}")
                        break

            all_text = ' '.join(all_lines)
            date_count = len(re.findall(r'\d{2}/\d{2}/\d{4}', all_text))

            if len(found_indicators) < 2 and date_count < 1:
                self.logger.warning("Insufficient indicators found in the extracted text.")
                return {
                    'success': False,
                    'message': (
                        "The Ghana Card information could not be properly detected. This might be because:\n"
                        "1. The card is not properly positioned (try landscape orientation)\n"
                        "2. The image is upside down or at an angle\n"
                        "3. The image quality is too low for proper text recognition"
                    ),
                    'card_info': None
                }

            if not all_lines:
                self.logger.warning("No text detected in the image.")
                return {
                    'success': False,
                    'message': "No text detected in the image. Check image quality or try preprocessing.",
                    'card_info': None
                }

            # Initialize dictionary to store Ghana Card information
            ghana_card_info = {
                'surname': None,
                'given_names': None,
                'nationality': None,
                'id_number': None,
                'date_of_birth': None,
                'sex': None,
                'date_of_issue': None,
                'date_of_expiry': None,
                'height': None,
                'document_number': None,
                'place_of_issuance': None
            }

            # Extract surname - using patterns from standalone script
            surname_patterns = [
                r'Sum?ame\/?Nom\s*$',
                r'Surname',
                r'SumameNom',
                r'Sum?ame[\W_]*Nom',
                r'Spmameitiom',
                r'Surname/Nom'
            ]
            
            for i, line in enumerate(all_lines):
                for pattern in surname_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        if i+1 < len(all_lines) and len(all_lines[i+1].strip()) < 30:  # Names are usually short
                            ghana_card_info['surname'] = all_lines[i+1].strip()
                            self.logger.info(f"Found surname: {all_lines[i+1].strip()}")
                            break
                if ghana_card_info['surname']:
                    break
            
            # Extract given names - using patterns from standalone script
            firstname_patterns = [
                r'Fir[s\-]?[mt]nam[e\-s]+Pr[e\-é]noms',
                r'FrsmnamesPrenome',
                r'FirstnamesPrenoms',
                r'Firsmnam.?sPrenoms',
                r'Prsmaneurrenens',
                r'Previous Names?'
            ]
            
            for i, line in enumerate(all_lines):
                for pattern in firstname_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        if i+1 < len(all_lines) and len(all_lines[i+1].strip()) < 50:  # Names are usually short
                            ghana_card_info['given_names'] = all_lines[i+1].strip()
                            self.logger.info(f"Found given names: {all_lines[i+1].strip()}")
                            break
                if ghana_card_info['given_names']:
                    break
            
            # Extract nationality - using improved logic from standalone script
            for i, line in enumerate(all_lines):
                if "GHANAIAN" in line:
                    ghana_card_info['nationality'] = "GHANAIAN"
                    self.logger.info("Found nationality: GHANAIAN")
                    break
                elif re.search(r'Nat[io]?[gn]al[il][tl]y|NationalityNationalite|Masonanyatonane', line, re.IGNORECASE):
                    if i+1 < len(all_lines) and len(all_lines[i+1].strip()) < 20:
                        ghana_card_info['nationality'] = all_lines[i+1].strip()
                        self.logger.info(f"Found nationality: {all_lines[i+1].strip()}")
                        break
            
            # Extract sex - using the improved strategies from standalone script
            for i, line in enumerate(all_lines):
                if re.search(r'Sex\/?Sexe', line, re.IGNORECASE):
                    sex_match = re.search(r'\b[MF]\b', line)
                    if sex_match:
                        ghana_card_info['sex'] = sex_match.group(0)
                        self.logger.info(f"Found sex: {sex_match.group(0)}")
                    elif i+1 < len(all_lines) and re.match(r'^[MF]$', all_lines[i+1].strip()):
                        ghana_card_info['sex'] = all_lines[i+1].strip()
                        self.logger.info(f"Found sex: {all_lines[i+1].strip()}")
                    break
            
            # If sex not found, try additional strategies from standalone script
            if not ghana_card_info['sex']:
                # Look for standalone M or F
                for line in all_lines:
                    if line.strip() == "M" or line.strip() == "F":
                        ghana_card_info['sex'] = line.strip()
                        self.logger.info(f"Found sex: {line.strip()}")
                        break
                    
                # If still not found, try searching in all text
                if not ghana_card_info['sex']:
                    sex_pattern = re.search(r'Sex\/?Sexe\s*([MF])', all_text, re.IGNORECASE)
                    if sex_pattern:
                        ghana_card_info['sex'] = sex_pattern.group(1)
                        self.logger.info(f"Found sex from full text: {sex_pattern.group(1)}")
            
            # Extract date of birth - using patterns from standalone script
            dob_patterns = [
                r'Date\s+of\s+Birth\/?Date\s+de\s+Naissance',
                r'Date of BirthDate de Naissance',
                r'[DP]ate\s+ofBirth',  # Handle OCR mistaking D for P
                r'[DP]ate\s+of[\W_]*Birth'  # More flexible pattern to catch OCR errors
            ]
            
            for i, line in enumerate(all_lines):
                for pattern in dob_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        # Check if date is in the same line
                        dob_match = re.search(r'\d{2}/\d{2}/\d{4}', line)
                        if dob_match:
                            ghana_card_info['date_of_birth'] = dob_match.group(0)
                            self.logger.info(f"Found DOB: {dob_match.group(0)}")
                        # Otherwise check next line
                        elif i+1 < len(all_lines):
                            dob_match = re.search(r'\d{2}/\d{2}/\d{4}', all_lines[i+1])
                            if dob_match:
                                ghana_card_info['date_of_birth'] = dob_match.group(0)
                                self.logger.info(f"Found DOB: {dob_match.group(0)}")
                        break
                if ghana_card_info['date_of_birth']:
                    break
            
            # Extract ID number - using patterns from standalone script
            id_patterns = [
                r'GHA[-\s]*\d{6,9}[-\s]*\d{1,2}',  # Standard format with possible spaces
                r'GHA[-\s]*\d{6,9}',               # Partial format (missing last digit)
                r'GHA.{0,2}\d{6,9}.{0,2}\d?'       # Very flexible pattern for OCR errors
            ]
            
            for i, line in enumerate(all_lines):
                if re.search(r'Personal ID Number', line, re.IGNORECASE):
                    # Try all patterns in current line
                    for pattern in id_patterns:
                        id_match = re.search(pattern, line)
                        if id_match:
                            # Clean up the ID by removing spaces
                            clean_id = re.sub(r'\s+', '', id_match.group(0))
                            # Ensure correct format
                            if len(clean_id) >= 12 and clean_id.startswith('GHA') and '-' in clean_id:
                                ghana_card_info['id_number'] = clean_id
                            else:
                                # Try to fix common format issues
                                if len(clean_id) >= 12 and clean_id.startswith('GHA'):
                                    # Add hyphen if missing
                                    if '-' not in clean_id:
                                        clean_id = f"{clean_id[:13]}-{clean_id[13:]}"
                                    ghana_card_info['id_number'] = clean_id
                            self.logger.info(f"Found ID number: {clean_id}")
                            break
                    
                    # If not found in current line, check next few lines
                    if not ghana_card_info['id_number']:
                        for j in range(i+1, min(i+3, len(all_lines))):
                            for pattern in id_patterns:
                                id_match = re.search(pattern, all_lines[j])
                                if id_match:
                                    clean_id = re.sub(r'\s+', '', id_match.group(0))
                                    if len(clean_id) >= 12 and clean_id.startswith('GHA'):
                                        # Add hyphen if missing
                                        if '-' not in clean_id:
                                            if len(clean_id) > 13:
                                                clean_id = f"{clean_id[:13]}-{clean_id[13:]}"
                                        ghana_card_info['id_number'] = clean_id
                                        self.logger.info(f"Found ID number: {clean_id}")
                                        break
                            if ghana_card_info['id_number']:
                                break
                    break
            
            # If ID not found by context, look in all lines
            if not ghana_card_info['id_number']:
                for line in all_lines:
                    for pattern in id_patterns:
                        id_match = re.search(pattern, line)
                        if id_match:
                            clean_id = re.sub(r'\s+', '', id_match.group(0))
                            if len(clean_id) >= 12 and clean_id.startswith('GHA'):
                                # Add hyphen if missing
                                if '-' not in clean_id and len(clean_id) > 13:
                                    clean_id = f"{clean_id[:13]}-{clean_id[13:]}"
                                ghana_card_info['id_number'] = clean_id
                                self.logger.info(f"Found ID number: {clean_id}")
                                break
                    if ghana_card_info['id_number']:
                        break
            
            # Extract dates using the standalone script logic
            date_pattern = r'\d{2}/\d{2}/\d{4}'
            all_dates = []
            
            for line in all_lines:
                dates = re.findall(date_pattern, line)
                all_dates.extend(dates)
            
            # Extract issue and expiry dates using patterns from standalone script
            issue_patterns = [
                r'Date\s+.?of\s+issuance|Date.or\s+issuance',
                r'd\'émission',
                r'issuance/Date',
                r'Date.*issuance'  # Very flexible pattern
            ]
            
            expiry_patterns = [
                r'Date\s+.?of\s+Expiry|Dateo[fl]\s+Expiry',
                r'd\'expiration',
                r'Expiry/Date',
                r'Date.*Expiry'  # Very flexible pattern
            ]
            
            for i, line in enumerate(all_lines):
                # Check for issue date
                for pattern in issue_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        date_match = re.search(date_pattern, line)
                        if date_match:
                            ghana_card_info['date_of_issue'] = date_match.group(0)
                            self.logger.info(f"Found issue date: {date_match.group(0)}")
                        elif i+1 < len(all_lines):
                            date_match = re.search(date_pattern, all_lines[i+1])
                            if date_match:
                                ghana_card_info['date_of_issue'] = date_match.group(0)
                                self.logger.info(f"Found issue date: {date_match.group(0)}")
                        break
                
                # Check for expiry date
                for pattern in expiry_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        date_match = re.search(date_pattern, line)
                        if date_match:
                            ghana_card_info['date_of_expiry'] = date_match.group(0)
                            self.logger.info(f"Found expiry date: {date_match.group(0)}")
                        elif i+1 < len(all_lines):
                            date_match = re.search(date_pattern, all_lines[i+1])
                            if date_match:
                                ghana_card_info['date_of_expiry'] = date_match.group(0)
                                self.logger.info(f"Found expiry date: {date_match.group(0)}")
                        break
            
            # Extract height using pattern from standalone script
            for i, line in enumerate(all_lines):
                if re.search(r'Height|Taille|HeightTaille', line, re.IGNORECASE):
                    height_match = re.search(r'(\d+\.\d+)\s*(m|cm)?', line)
                    if height_match:
                        ghana_card_info['height'] = height_match.group(1)
                        self.logger.info(f"Found height: {height_match.group(1)}")
                    elif i+1 < len(all_lines):
                        height_match = re.search(r'(\d+\.\d+)\s*(m|cm)?', all_lines[i+1])
                        if height_match:
                            ghana_card_info['height'] = height_match.group(1)
                            self.logger.info(f"Found height: {height_match.group(1)}")
                    break
            
            # If height not found, look for pattern in all lines
            if not ghana_card_info['height']:
                for line in all_lines:
                    if "GHA" not in line and "Date" not in line:
                        height_match = re.search(r'1\.\d{2}', line)  # Most heights are like 1.65, 1.75, etc.
                        if height_match:
                            ghana_card_info['height'] = height_match.group(0)
                            self.logger.info(f"Found height: {height_match.group(0)}")
                            break
            
            # Extract document number using patterns from standalone script
            doc_patterns = [
                r'[A-Z]{2}\d{6,7}',              # More specific format like AR5151853
                r'[A-Z]{2}\d{5,9}',            # Flexible but must start with 2 letters
                r'[A-Z]\d[A-Z0-9]{5,8}'        # Alternative format
            ]
            
            for i, line in enumerate(all_lines):
                if re.search(r'Document\s+Number|Dacument\s+Number|Document Numbed', line, re.IGNORECASE):
                    # Try all patterns in current line
                    for pattern in doc_patterns:
                        doc_match = re.search(pattern, line)
                        if doc_match and "GHA" not in doc_match.group(0) and "IDENTITY" not in doc_match.group(0):
                            ghana_card_info['document_number'] = doc_match.group(0)
                            self.logger.info(f"Found document number: {doc_match.group(0)}")
                            break
                    
                    # If not found in current line, check next line
                    if not ghana_card_info['document_number']:
                        if i+1 < len(all_lines):
                            for pattern in doc_patterns:
                                doc_match = re.search(pattern, all_lines[i+1])
                                if doc_match and "GHA" not in doc_match.group(0) and "IDENTITY" not in doc_match.group(0):
                                    ghana_card_info['document_number'] = doc_match.group(0)
                                    self.logger.info(f"Found document number: {doc_match.group(0)}")
                                    break
                    break
            
            # If document number not found, look through all lines
            if not ghana_card_info['document_number']:
                for line in all_lines:
                    if "GHA" not in line and not re.search(r'Date|Number|Place|Personal', line, re.IGNORECASE):
                        for pattern in doc_patterns:
                            doc_match = re.search(pattern, line)
                            if doc_match and len(doc_match.group(0)) >= 5:
                                ghana_card_info['document_number'] = doc_match.group(0)
                                self.logger.info(f"Found document number: {doc_match.group(0)}")
                                break
                        if ghana_card_info['document_number']:
                            break
            
            # Extract place of issuance using patterns from standalone script
            place_patterns = [
                r'Place\s+of\s+Issuance|Placeof\s+Issuance',
                r'Lieu\s+de\s+delivrance',
                r'Placeof IssuanceLieu',  # Handle OCR merging
                r'Place.*Issuance'        # Very flexible pattern
            ]
            
            for i, line in enumerate(all_lines):
                for pattern in place_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        # Check if ACCRA is in the line
                        if "ACCRA" in line:
                            ghana_card_info['place_of_issuance'] = "ACCRA"
                            self.logger.info("Found place of issuance: ACCRA")
                        # Otherwise check next line
                        elif i+1 < len(all_lines) and len(all_lines[i+1].strip()) < 20:
                            if all_lines[i+1].strip() == "ACCRA":
                                ghana_card_info['place_of_issuance'] = "ACCRA"
                                self.logger.info("Found place of issuance: ACCRA")
                            else:
                                ghana_card_info['place_of_issuance'] = all_lines[i+1].strip()
                                self.logger.info(f"Found place of issuance: {all_lines[i+1].strip()}")
                        break
                if ghana_card_info['place_of_issuance']:
                    break
            
            # If place of issuance not found, look for ACCRA
            if not ghana_card_info['place_of_issuance']:
                for line in all_lines:
                    if line.strip() == "ACCRA":
                        ghana_card_info['place_of_issuance'] = "ACCRA"
                        self.logger.info("Found place of issuance: ACCRA")
                        break
            
            # If dates not found through context, use position in all_dates as in standalone script
            if len(all_dates) >= 1 and not ghana_card_info['date_of_birth']:
                ghana_card_info['date_of_birth'] = all_dates[0]
                self.logger.info(f"Found DOB (from dates list): {all_dates[0]}")
            
            if len(all_dates) >= 2 and not ghana_card_info['date_of_issue']:
                ghana_card_info['date_of_issue'] = all_dates[1]
                self.logger.info(f"Found issue date (from dates list): {all_dates[1]}")
            
            if len(all_dates) >= 3 and not ghana_card_info['date_of_expiry']:
                ghana_card_info['date_of_expiry'] = all_dates[2]
                self.logger.info(f"Found expiry date (from dates list): {all_dates[2]}")
            
            # Check extraction success using criteria from standalone script
            found_count = sum(1 for value in ghana_card_info.values() if value is not None)
            extraction_success = found_count >= 3  # Consider it successful if we found at least 3 fields
            
            if not extraction_success:
                return {
                    'success': False,
                    'message': f'Only {found_count} pieces of information could be extracted. This usually happens with poor image quality or orientation.',
                    'card_info': ghana_card_info
                }

            self.logger.info(f"Extraction completed successfully. Found {found_count} fields.")
            return {
                'success': True,
                'message': 'Information extracted successfully',
                'card_info': ghana_card_info
            }

        except Exception as e:
            self.logger.error(f"Error in OCR extraction for {file_path}: {str(e)}")
            return {
                'success': False,
                'message': f'Error processing image: {str(e)}',
                'card_info': None
            }

# Initialize the OCR service
ocr_service = GhanaCardOCR()

# Export the instance
__all__ = ['ocr_service']