import os
from fastapi import FastAPI, Body
from predict import predict
from improve import ImproveModel
from workflow import run_suggest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()

improver = ImproveModel(
    model_path=os.path.join(BASE_DIR, "model.pkl"),
    le_path=os.path.join(BASE_DIR, "label_encoder.pkl"),
    data_path=os.path.join(BASE_DIR, "dice_data.pkl")
)

@app.get("/")
def home():
    return {"message": "ML API is running"}

@app.post("/predict")
def get_prediction(data: dict):
    try:
        result = predict(data)
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}

@app.post("/improve")
def improve_api(input_json: dict):
    try:
        result = improver.improve(input_json)

        return {
            "status": "success",
            **result
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/suggest")
def suggest_api(data: dict):
    try:
        result = run_suggest(data)

        return {
            "status": "success",
            **result
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }