import os


import os
from celery import Celery

BROKER_URI = "redis://localhost:6379/0"
BACKEND_URI = "redis://localhost:6379/0"

celery = Celery(
    'celery_worker',
    broker=BROKER_URI,
    backend=BACKEND_URI,
    include=['celeryapp.tasks']
)