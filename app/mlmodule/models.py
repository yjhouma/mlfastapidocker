import warnings
from .base import Model, Evaluator, DataLoader, TrainLogger
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd



class GeneralRegressionEvaluator(Evaluator):
    def evaluate(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual,pred))
        mae = mean_absolute_error(actual,pred)
        r2 = r2_score(actual, pred)
        return {"rmse":rmse, "mae":mae, "r2":r2}
             

class ElasticNetModel(Model):
    def __init__(self, model_id: str, alpha=0.5, l1_ratio=0.5):
        self.model_id = model_id
        self.alpha=alpha
        self.l1_ratio=l1_ratio
        self.model = ElasticNet(alpha=alpha,l1_ratio=l1_ratio,random_state=101)

    def train(self, train_data: DataLoader, evaluator: Evaluator, logger: TrainLogger,split_test_from_train=True, test_data: DataLoader = None):
        if not split_test_from_train and test_data is None:
            raise "test_data cannot be None if split_test_from_train is False"
        elif split_test_from_train and test_data is not None:
            warnings.warn("Warning: split_test_from_train is True although test_data is give, test_data will be ignored")
            X,y = train_data.load_data()
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25,random_state=101)
        elif not split_test_from_train and test_data is not None:
            X_train, y_train = train_data.load_data()
            X_test, y_test = test_data.load_data()
        else:
            X,y = train_data.load_data()
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25,random_state=101)
        
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        test_metrics = evaluator.evaluate(y_test, y_pred)
        logger.log_train(model_id=self.model_id, result=test_metrics)
        return test_metrics
    
    def predict(self, input_data: dict):
        # print(input_data)
        inpt = pd.DataFrame([input_data])
        result = self.model.predict(inpt)
        return result