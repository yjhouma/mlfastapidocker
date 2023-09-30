from abc import ABC, abstractmethod

class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, actual, pred):
        pass

class DataLoader(ABC):
    @abstractmethod
    def load_data(self, target):
        # and target
        # Return a 2 DataFrame One is Input the other is target
        pass
    
class TrainLogger(ABC):
    @abstractmethod
    def log_train(self, model_id, result: dict):
        pass


class Model(ABC):
    @abstractmethod
    def train(self, train_data: DataLoader, evaluator: Evaluator, logger: TrainLogger,split_test_from_train=True, test_data: DataLoader = None):
        pass

    @abstractmethod
    def predict(self, input_data: dict):
        pass



class ModelLoader(ABC):
    @abstractmethod
    def load_model(self, model_id: str) -> Model:
        pass

class ArtifactSaver(ABC):
    @abstractmethod
    def save_model(self, model: Model) -> None:
        pass

