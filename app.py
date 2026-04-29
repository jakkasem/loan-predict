from fastapi import FastAPI, Body
from pydantic import BaseModel
from predict import predict
from improve import ImproveModel
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VERSION = "1.1.0"

class LoanInput(BaseModel):
    final_d: int
    emp_length_int: float
    home_ownership_cat: int
    income_cat: int
    annual_inc: int
    loan_amount: int
    term_cat: int
    application_type_cat: int
    purpose_cat: int
    interest_payment_cat: int
    interest_rate: float
    grade: str
    dti: float
    total_pymnt: float
    total_rec_prncp: float
    recoveries: float
    installment: float

    class Config:
        json_schema_extra = {
            "example": {
                "final_d": 104203,
                "emp_length_int": 0.5,
                "home_ownership_cat": 1,
                "income_cat": 1,
                "annual_inc": 30000,
                "loan_amount": 2500,
                "term_cat": 2,
                "application_type_cat": 1,
                "purpose_cat": 2,
                "interest_payment_cat": 2,
                "interest_rate": 15.27,
                "grade": "C",
                "dti": 1,
                "total_pymnt": 1008.71,
                "total_rec_prncp": 456.46,
                "recoveries": 117,
                "installment": 59.83
            }
        }

app = FastAPI(
    title="Loan Predict API",
    description="API สำหรับทำนายความเสี่ยงสินเชื่อ",
    version=VERSION
)

# improver = ImproveModel(
#     model_path=os.path.join(BASE_DIR, "model.pkl"),
#     le_path=os.path.join(BASE_DIR, "label_encoder.pkl"),
#     data_path=os.path.join(BASE_DIR, "dice_data.pkl")
# )

@app.get("/")
def home():
    return {"message": "ML API is running", "version": VERSION}

@app.post("/predict")
def get_prediction(data: LoanInput):
    try:
        result = predict(data.model_dump())
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}

# @app.post("/improve")
# def improve_api(input_json: dict):
#     try:
#         result = improver.improve(input_json)

#         return {
#             "status": "success",
#             **result
#         }
#     except Exception as e:
#         return {
#             "status": "error",
#             "message": str(e)
#         }
