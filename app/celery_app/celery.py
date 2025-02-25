from celery import Celery
import os
from kombu import Exchange, Queue
import logging
import multiprocessing

# Force spawn method for CUDA compatibility
multiprocessing.set_start_method('spawn', force=True)

# Configure logging
logger = logging.getLogger(__name__)

# Redis configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)

# Broker URL with password if set
if REDIS_PASSWORD:
    broker_url = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
else:
    broker_url = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

# Define exchanges
high_priority_exchange = Exchange('high_priority', type='direct')
default_exchange = Exchange('default', type='direct')
low_priority_exchange = Exchange('low_priority', type='direct')

# Define queues with priorities
task_queues = [
    Queue('high_priority', high_priority_exchange, routing_key='high_priority', queue_arguments={'x-max-priority': 10}),
    Queue('default', default_exchange, routing_key='default', queue_arguments={'x-max-priority': 5}),
    Queue('low_priority', low_priority_exchange, routing_key='low_priority', queue_arguments={'x-max-priority': 1})
]

# Initialize Celery
celery_app = Celery(
    'ml_tasks',
    broker=broker_url,
    backend=broker_url,
    include=[
        'app.celery_app.tasks.face_tasks',
        'app.celery_app.tasks.document_tasks',
        'app.celery_app.tasks.spoof_tasks'
    ]
)

# Worker settings optimized for g4dn.2xlarge
worker_settings = {
    'worker_concurrency': 1,  # Single worker for GPU tasks
    'worker_prefetch_multiplier': 1,  # Disable prefetching
    'worker_max_tasks_per_child': 50,  # Restart worker periodically to prevent GPU memory leaks
    'worker_max_memory_per_child': 8000000,  # 8GB memory limit per child
    'worker_pool': 'solo',  # Use solo pool for GPU tasks
    'worker_proc_alive_timeout': 120,
}

# GPU specific settings for Tesla T4
gpu_settings = {
    'task_time_limit': 3600,  # 1 hour max
    'task_soft_time_limit': 3300,  # 55 minutes soft limit
    'task_acks_late': True,  # Prevent multiple workers from picking up same task
    'task_reject_on_worker_lost': True,  # Requeue tasks if worker dies
    'max_memory_per_container': int(15360 * 0.8),  # 80% of T4 GPU memory (in MB)
}

# Redis/Broker settings optimized for g4dn.2xlarge
broker_settings = {
    'broker_transport_options': {
        'visibility_timeout': 3600,
        'max_connections': 100,
        'socket_timeout': 30,
        'socket_connect_timeout': 30,
        'socket_keepalive': True,
        'health_check_interval': 10,
        'max_retries': 3,
        'interval_start': 0,
        'interval_step': 0.2,
        'interval_max': 0.5,
    }
}

# Task routing configuration
task_routes = {
    # High priority tasks
    'app.celery_app.tasks.face_tasks.face_comparison': {'queue': 'high_priority'},
    'app.celery_app.tasks.face_tasks.check_task_status': {'queue': 'high_priority'},
    
    # Default priority tasks
    'app.celery_app.tasks.document_tasks.document_verification': {'queue': 'default'},
    'app.celery_app.tasks.spoof_tasks.spoof_detection': {'queue': 'default'},
    
    # Low priority tasks
    '*': {'queue': 'low_priority'}  # All other tasks
}

# Configure Celery app
celery_app.conf.update(
    result_expires=3600,  # Results expire after 1 hour
    task_track_started=True,
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    task_routes={
        'app.celery_app.tasks.face_tasks.*': {'queue': 'high_priority'},
        'app.celery_app.tasks.spoof_tasks.*': {'queue': 'default'},
        'app.celery_app.tasks.document_tasks.*': {'queue': 'default'},
    },
    task_default_queue='default',
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50
)

# Main Celery configuration
celery_app.conf.update(
    # Worker settings
    **worker_settings,
    
    # Broker settings
    broker_url=broker_url,
    broker_connection_retry=True,
    broker_connection_max_retries=None,
    broker_connection_timeout=30,
    broker_heartbeat=10,
    broker_transport_options=broker_settings['broker_transport_options'],
    
    # Task settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    task_queues=task_queues,
    task_routes=task_routes,
    task_default_queue='default',
    task_default_exchange='default',
    task_default_routing_key='default',
    
    # Task execution settings
    task_acks_late=gpu_settings['task_acks_late'],
    task_reject_on_worker_lost=gpu_settings['task_reject_on_worker_lost'],
    task_time_limit=gpu_settings['task_time_limit'],
    task_soft_time_limit=gpu_settings['task_soft_time_limit'],
    task_track_started=True,
    
    # Performance optimizations
    task_compression='gzip',
    result_compression='gzip',
    task_ignore_result=False,
    
    # Result backend settings
    result_backend=broker_url,
    result_extended=True,
    
    # Retry settings
    task_publish_retry=True,
    task_publish_retry_policy={
        'max_retries': 3,
        'interval_start': 0,
        'interval_step': 0.2,
        'interval_max': 0.5,
    },
    
    # General settings
    enable_utc=True,
    timezone='UTC',
    broker_connection_retry_on_startup=True
)

# Startup logging
logger.info(f"Celery configured with broker: {broker_url}")
logger.info(f"Task queues: {[q.name for q in task_queues]}")
logger.info(f"Worker concurrency: {celery_app.conf.worker_concurrency}")
logger.info(f"Worker pool: {celery_app.conf.worker_pool}")
logger.info(f"GPU settings enabled: {torch.cuda.is_available() if 'torch' in globals() else 'torch not imported'}")

if __name__ == '__main__':
    celery_app.start()