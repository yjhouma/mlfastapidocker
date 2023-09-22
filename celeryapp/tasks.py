from celery import Celery
from celery import current_task

import mlmodule.train as train
from .worker import celery



# celery = Celery("tasks", broker="redis://localhost:6379/0", backend='redis://localhost:6379/0')

@celery.task
def train_model_task(alpha,l1_ratio):
    model_id = current_task.request.id
    try:
        (rmse, mae, r2) = train.train_elastic_model(alpha=alpha, l1_ratio=l1_ratio, model_id=model_id)
        return {"status": "completed", "model_id": model_id, "rmse":rmse,"mae":mae,"r2":r2}
    except Exception as e:
        return {"status": "failed", "error_message": str(e)}
