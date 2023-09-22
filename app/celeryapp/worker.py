import os
from celery import Celery
from app.config.config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND


celery = Celery(
    'celery_worker',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=['app.celeryapp.tasks']
)