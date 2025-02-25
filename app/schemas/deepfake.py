from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List, Dict

class DeepfakeDetectionRequest(BaseModel):
    video_url: HttpUrl = Field(..., description="URL of the video to analyze")
    generate_report: Optional[bool] = Field(False, description="Whether to generate a detailed report") 

class FrameLevelResults(BaseModel):
    """Frame-by-frame analysis results"""
    real_frames: int = Field(..., description="Number of frames classified as real")
    fake_frames: int = Field(..., description="Number of frames classified as fake")
    average_confidence: float = Field(..., description="Average confidence score across frames")

class DeepfakeAnalysisResponse(BaseModel):
    """Response model for deepfake detection analysis"""
    is_deepfake: bool = Field(..., description="Whether the video is classified as deepfake")
    confidence: float = Field(..., description="Overall confidence score (0-100)", ge=0, le=100)
    processing_time: float = Field(..., description="Total processing time in seconds")
    frames_analyzed: int = Field(..., description="Total number of frames analyzed")
    video_duration: float = Field(..., description="Video duration in seconds")
    detection_method: str = Field(..., description="Method used for detection (single/ensemble)")
    frame_level_results: FrameLevelResults = Field(..., description="Detailed frame analysis results")
    error_message: Optional[str] = Field(None, description="Error message if analysis failed")
    error_code: Optional[str] = Field(None, description="Error code if analysis failed")

    class Config:
        json_schema_extra = {
            "example": {
                "is_deepfake": False,
                "confidence": 98.5,
                "processing_time": 15.2,
                "frames_analyzed": 450,
                "video_duration": 30.5,
                "detection_method": "ensemble",
                "frame_level_results": {
                    "real_frames": 445,
                    "fake_frames": 5,
                    "average_confidence": 98.5
                }
            }
        } 

class DetectionStats(BaseModel):
    """Statistics for deepfake detection service"""
    total_processed: int = Field(..., description="Total number of videos processed")
    average_processing_time: float = Field(..., description="Average processing time in seconds")
    models_loaded: int = Field(..., description="Number of models currently loaded")
    device: str = Field(..., description="Current processing device")
    batch_size: int = Field(..., description="Current batch size")
    memory_usage: str = Field(..., description="Current memory usage")
    uptime: float = Field(..., description="Service uptime in hours")
    success_rate: float = Field(..., description="Percentage of successful detections")

    class Config:
        json_schema_extra = {
            "example": {
                "total_processed": 1500,
                "average_processing_time": 12.3,
                "models_loaded": 2,
                "device": "cuda:0",
                "batch_size": 32,
                "memory_usage": "2.1GB",
                "uptime": 72.5,
                "success_rate": 99.8
            }
        } 