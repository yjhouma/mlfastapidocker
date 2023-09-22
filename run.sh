celery -A app.celeryapp.worker worker  -l info --concurrency=2 &
uvicorn app.main:app --host 0.0.0.0 --reload --workers 2