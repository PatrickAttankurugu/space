from redis import Redis
from rq import Queue
import os
from typing import Optional
from functools import wraps
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

# Initialize Redis connection
redis_conn = Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    password=os.getenv('REDIS_PASSWORD', None),
    db=0
)

# Create queues with different priorities
high_priority_queue = Queue('high', connection=redis_conn)
default_queue = Queue('default', connection=redis_conn)
low_priority_queue = Queue('low', connection=redis_conn)

def get_queue(priority: str = 'default') -> Queue:
    """Get queue based on priority level"""
    queues = {
        'high': high_priority_queue,
        'default': default_queue,
        'low': low_priority_queue
    }
    return queues.get(priority, default_queue)

def enqueue_task(func, *args, priority='default', **kwargs):
    """Enqueue a task with specified priority"""
    queue = get_queue(priority)
    try:
        job = queue.enqueue(
            func,
            *args,
            **kwargs,
            job_timeout='10m',
            result_ttl=3600,
            failure_ttl=3600
        )
        return job.id
    except Exception as e:
        logger.error(f"Failed to enqueue task: {str(e)}")
        raise HTTPException(status_code=503, detail="Queue service unavailable")
