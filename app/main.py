from fastapi import FastAPI
import uvicorn
import pickle
from app.celeryapp.tasks import train_model_task, prediction_task
from .schemas import TrainInput, PredictInput

app = FastAPI(debug=True)

@app.get('/')
def home():
    return {'text': 'home page'}

@app.post('/train_model')
def run_train_model(trainInput: TrainInput):
    result = train_model_task.delay(**trainInput.parameters)
    return {
        "message": "Training task started with alpha={} and l1_ratio={}".format(trainInput.parameters['alpha'],trainInput.parameters['l1_ratio']),
        "task_id":result.id
        }

# @app.post('/predict')
# def predict(model_id: str
#             ,fixed_acidity: float
#             ,volatile_acidity: float
#             ,citric_acid: float
#             ,residual_sugar: float
#             ,chlorides: float
#             ,free_sulfur_dioxide: float
#             ,total_sulfur_dioxide: float
#             ,density: float
#             ,pH: float
#             ,sulphates: float
#             ,alcohol: float):
#     model = pickle.load(open('model/'+model_id+'.pkl','rb'))
#     prediction = model.predict([[fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol]])
#     print(prediction)

#     return {round(prediction[0],2)}

@app.post('/predict')
def predict(predict_input: PredictInput):
    result = prediction_task.delay(predict_input.model_id, dict(predict_input.data_input))
    return {
        "message": "Predicting Task started using model_id = {}".format(predict_input.model_id),
        "predict_id": result.id
    }

    # model_id
    # model = pickle.load(open('model/'+model_id+'.pkl','rb'))
    # prediction = model.predict([[fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol]])
    # print(prediction)


@app.get('/train-status/{task_id}')
def get_task_status(task_id: str):
    celery_result = train_model_task.AsyncResult(task_id)

    if celery_result.state == "PENDING":
        return {"status":"pending"}
    elif celery_result.state == "SUCCESS":
        return {"status": "completed", "result":celery_result.result}
    elif celery_result.state == "FAILURE":
        return {"status": "failed", "error_message": str(celery_result.result)}


@app.get('/predict-result/{predict_id}')
def get_predict_result(predict_id):
    celery_result = prediction_task.AsyncResult(predict_id)
    if celery_result.state == "PENDING":
        return {"status":"pending"}
    elif celery_result.state == "SUCCESS":
        return {"status": "completed", "result":celery_result.result}
    elif celery_result.state == "FAILURE":
        return {"status": "failed", "error_message": str(celery_result.result)}


if __name__ == '__main__':
    # print(predict(7,0.27,0.36,20.7,0.045,45,170,1.001,3,0.45,8.8))
    uvicorn.run(app)
    