import os
import warnings
import sys
import random
import string

import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet


model_path = "model/"
log_file = "logs/logs.csv"


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
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.writelines("model_id,alpha,l1_ratio,rmse,mae,r2\n")
    with open(log_file, "a") as f:
            f.writelines("{},{},{},{},{},{}\n".format(model_id, alpha, l1_ratio, rmse,mae,r2))


def save_artifact(model, model_id):
    pass

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

    with open(model_path+model_id+'.pkl','wb') as f:
        pickle.dump(lr, f)
    
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