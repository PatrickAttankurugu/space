import easyocr
import os
import numpy

def download_models():
    print("Pre-downloading OCR models...")
    model_path = '/app/models/ocr'
    os.makedirs(model_path, exist_ok=True)
    
    # Initialize with downloads enabled to get all required models
    reader = easyocr.Reader(
        ['en'],
        model_storage_directory=model_path,
        download_enabled=True,  # Enable downloads during build
        gpu=False,
        detector=True,  # Ensure detector is enabled
        recognizer=True  # Ensure recognizer is enabled
    )
    
    # Force model initialization to trigger downloads
    _ = reader.readtext(
        numpy.zeros((224, 224, 3), dtype=numpy.uint8)
    )
    
    print("OCR models downloaded successfully")

if __name__ == "__main__":
    download_models() 