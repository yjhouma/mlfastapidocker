from .base import DataLoader, TrainLogger, ArtifactSaver, Model, ModelLoader
from google.cloud import storage, firestore
from app.config.config import GOOGLE_CLOUD_STORAGE_BUCKET_NAME, MODEL_ARTIFACT_ROOT_DIRECTORY,GOOGLE_CLOUD_PROJECT,FIRESTORE_COLLECTION
from io import BytesIO
from datetime import datetime
import pytz
import pandas as pd
import pickle



class CSVGoogleCloudStorageDataLoader(DataLoader):
    def __init__(self, blob_name, target, delimiter=',' ):
        self.delimiter=delimiter
        self.target = target
        self.__storage_client = storage.Client()
        self.__bucket = self.__storage_client.get_bucket(GOOGLE_CLOUD_STORAGE_BUCKET_NAME)
        self.blob = self.__bucket.blob(blob_name)

    def load_data(self):
        byte_stream = BytesIO()
        self.blob.download_to_file(byte_stream)
        byte_stream.seek(0)
        df = pd.read_csv(byte_stream, delimiter=self.delimiter)
        x = df.drop([self.target], axis=1)
        y = df[[self.target]]
        return x, y
    

class FirestoreTrainLogger(TrainLogger):
    def log_train(self, model_id, result: dict):
        db=firestore.Client(GOOGLE_CLOUD_PROJECT)
        doc_ref = db.collection(FIRESTORE_COLLECTION).document(model_id)
        train_time = datetime.now(pytz.timezone('Asia/Jakarta')).isoformat()
        doc_ref.set({"completed_time":train_time, "model_id": model_id, "eval_metric":result})

    
class GoogleCloudStorageArtifactSaver(ArtifactSaver):
    def save_model(self, model: Model) -> None:
        blob_name = MODEL_ARTIFACT_ROOT_DIRECTORY+model.model_id+".pkl"
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(GOOGLE_CLOUD_STORAGE_BUCKET_NAME)
        blob = bucket.blob(blob_name)
        m = pickle.dumps(model)
        blob.upload_from_string(m)


class GoogleCloudStorageModelLoader(ModelLoader):
    def load_model(self, model_id):
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(GOOGLE_CLOUD_STORAGE_BUCKET_NAME)
        blob_name = MODEL_ARTIFACT_ROOT_DIRECTORY+model_id+".pkl"
        blob = bucket.blob(blob_name=blob_name)
        pickle_in = blob.download_as_string()
        model = pickle.loads(pickle_in)
        return model
