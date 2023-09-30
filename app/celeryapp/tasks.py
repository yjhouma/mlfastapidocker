from celery import Celery
from celery import current_task

# import app.mlmodule.train as train
from .worker import celery

from app.mlmodule.connectors import CSVGoogleCloudStorageDataLoader, FirestoreTrainLogger
from app.mlmodule.models import ElasticNetModel, GeneralRegressionEvaluator
from app.mlmodule.connectors import GoogleCloudStorageArtifactSaver, GoogleCloudStorageModelLoader



# celery = Celery("tasks", broker="redis://localhost:6379/0", backend='redis://localhost:6379/0')

@celery.task
def train_model_task(alpha,l1_ratio):
    model_id = current_task.request.id
    try:
        loader = CSVGoogleCloudStorageDataLoader(blob_name='data/winequality-white.csv',target='quality',delimiter=';')
        logger = FirestoreTrainLogger()
        evaluator = GeneralRegressionEvaluator()
        artifact_saver = GoogleCloudStorageArtifactSaver()
        model = ElasticNetModel(model_id, alpha=alpha,l1_ratio=l1_ratio)
        result = model.train(train_data=loader, evaluator=evaluator,logger=logger)
        artifact_saver.save_model(model=model)
        return {"status": "completed", "model_id": model_id, "evaluation":result}
    except Exception as e:
        return {"status": "failed", "error_message": str(e)}


@celery.task
def prediction_task(model_id, data_input):
    try:
        model = GoogleCloudStorageModelLoader().load_model(model_id)
        return {"status": "completed", "model_id":model_id, "predictions": model.predict(data_input)}
    except Exception as e:
        {"status": "failed", "error_message": str(e)}
        


    # print(model.predict(inpt))

