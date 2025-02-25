# Import all tasks here for easy access
from app.celery_app.tasks.face_tasks import face_comparison
from app.celery_app.tasks.document_tasks import document_verification
from app.celery_app.tasks.spoof_tasks import spoof_detection

__all__ = [
    'face_comparison',
    'document_verification',
    'spoof_detection'
]