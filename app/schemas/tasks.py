from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class TaskResponse(BaseModel):
    """Response model for asynchronous task submission"""
    task_id: str = Field(..., description="Unique identifier for the submitted task")
    status: str = Field(..., description="Current status of the task")
    message: str = Field(..., description="Status message or description")

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "7a1b3b11-3568-424f-8ca9-ce1da94d3344",
                "status": "processing",
                "message": "Task submitted successfully"
            }
        }

class TaskResult(BaseModel):
    """Model for task execution results"""
    status: str = Field(..., description="Task execution status")
    result: Dict[str, Any] = Field(..., description="Task execution result")
    processing_time: float = Field(..., description="Task processing time in seconds")
    error: Optional[str] = None
    traceback: Optional[str] = None