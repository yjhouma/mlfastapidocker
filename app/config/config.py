from starlette.config import Config


config = Config(".env")

# CELERY_BROKER_URL=redis://celerybackend:6379/0
# CELERY_RESULT_BACKEND=redis://celerybackend:6379/0

CELERY_BROKER_URL: str = config("CELERY_BROKER_URL", default="")
CELERY_RESULT_BACKEND: str = config("CELERY_RESULT_BACKEND", default="")
GOOGLE_CLOUD_STORAGE_BUCKET_NAME: str = config("GOOGLE_CLOUD_STORAGE_BUCKET_NAME", default="")
MODEL_ARTIFACT_ROOT_DIRECTORY: str = config("MODEL_ARTIFACT_ROOT_DIRECTORY", default="")
GOOGLE_CLOUD_PROJECT: str = config("GOOGLE_CLOUD_PROJECT", default="")
FIRESTORE_COLLECTION: str = config("FIRESTORE_COLLECTION", default="")