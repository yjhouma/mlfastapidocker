from starlette.config import Config


config = Config(".env")

# CELERY_BROKER_URL=redis://celerybackend:6379/0
# CELERY_RESULT_BACKEND=redis://celerybackend:6379/0

CELERY_BROKER_URL: str = config("CELERY_BROKER_URL", default="")
CELERY_RESULT_BACKEND: str = config("CELERY_RESULT_BACKEND", default="")