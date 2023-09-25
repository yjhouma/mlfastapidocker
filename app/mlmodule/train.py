import warnings
import sys
import random
import string
from google.cloud import firestore
import pickle
import pandas as pd
import numpy as np
from google.cloud import storage
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from mlmodule.base import Model
from config.config import GOOGLE_CLOUD_STORAGE_BUCKET_NAME, MODEL_ARTIFACT_ROOT_DIRECTORY,GOOGLE_CLOUD_PROJECT,FIRESTORE_COLLECTION
from io import BytesIO
import warnings
from mlmodule.base import *






# def save_artifact(model, model_id):
#     blob_name = MODEL_ARTIFACT_ROOT_DIRECTORY+model_id+".pkl"
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(GOOGLE_CLOUD_STORAGE_BUCKET_NAME)
#     blob = bucket.blob(blob_name)
#     m = pickle.dumps(model)
#         # blob.upload_from_filename(file_path)
#     blob.upload_from_string(m)
    



def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual,pred))
    mae = mean_absolute_error(actual,pred)
    r2 = r2_score(actual, pred)
    return rmse,mae,r2

def load_data(data_path, target="quality", delimiter=','):
    data = pd.read_csv(data_path,delimiter=delimiter)
    train, test = train_test_split(data,random_state=100)
    train_x = train.drop([target], axis=1)
    test_x = test.drop([target], axis=1)
    train_y = train[[target]]
    test_y = test[[target]]

    return train_x, train_y, test_x, test_y

def generate_model_id():
    l = string.ascii_letters
    ids = ''.join(random.choice(l) for i in range(10))
    return ids

def write_log(model_id, alpha, l1_ratio, rmse,mae,r2):
    db=firestore.Client(GOOGLE_CLOUD_PROJECT)
    doc_ref = db.collection(FIRESTORE_COLLECTION).document(model_id)
    doc_ref.set({"model_id": model_id, "alpha": alpha, "l1_ratio":l1_ratio, "rmse":rmse, "mae":mae, "r2":r2})

def save_artifact(model, model_id):
    blob_name = MODEL_ARTIFACT_ROOT_DIRECTORY+model_id+".pkl"
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(GOOGLE_CLOUD_STORAGE_BUCKET_NAME)
    blob = bucket.blob(blob_name)
    m = pickle.dumps(model)
        # blob.upload_from_filename(file_path)
    blob.upload_from_string(m)
    
def train_elastic_model(alpha, l1_ratio, model_id=None):
    if model_id is None:
        model_id = generate_model_id()

    train_x, train_y, test_x, test_y = load_data("data/winequality-white.csv",delimiter=';')
    lr = ElasticNet(alpha=alpha,l1_ratio=l1_ratio,random_state=101)
    lr.fit(train_x, train_y)
    y_predict = lr.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, y_predict)

    write_log(
         model_id=model_id,
         alpha=alpha,
         l1_ratio=l1_ratio,
         rmse=rmse,
         mae=mae,
         r2=r2
    )

    save_artifact(lr, model_id)
    
    return (rmse, mae, r2)
     


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(100)
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    train_elastic_model(alpha=alpha, l1_ratio=l1_ratio)

    # load_model = pickle.load(open(model_path+model_id+'.pkl','rb'))

    # print(load_model.predict(test_x))


    # "fixed acidity";"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"