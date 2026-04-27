import seaborn as sns
import numpy as np
import pandas as pd
import joblib
import pyodbc
import urllib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.metrics import classification_report 
from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy import create_engine
from fastapi import FastAPI
from pydantic import BaseModel

le = LabelEncoder()
model = RandomForestClassifier(random_state=42)

param_grid = {
 'model__n_estimators': [100,200,300,400,500],
 'model__max_features': ['sqrt'],
 'model__max_depth': [5,6,7,8,9,10,11,12], 
 'model__criterion': ['gini', 'entropy'] 
}

# host = 'localhost'
# db = 'mydb' 
# user = 'postgres'
# pw = 'postgres'
# port = '5432'

host = '34.170.136.87'
db = 'mydb' 
user = 'loanuser'
pw = 'loanpass'
port = '5432'

connection_url = f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}"
engine = create_engine(connection_url)

query = "SELECT final_d, emp_length_int, home_ownership_cat, income_cat, annual_inc, loan_amount, term_cat " \
    ", application_type_cat, purpose_cat, interest_payment_cat, interest_rate, grade, dti, total_pymnt " \
    ", total_rec_prncp, recoveries, installment, loan_condition_cat" \
    " FROM loan.customer" # ใส่ชื่อ Table ของคุณ
data = pd.read_sql(query, engine)

data1=data.dropna(subset=['loan_condition_cat'])
joblib.dump(data, r"C:\Python_Project\Loan_Data\dice_data.pkl")
X = data1.drop('loan_condition_cat', axis=1)
y = data1['loan_condition_cat']
y = le.fit_transform(y)

cat_cols = X.select_dtypes(include=['object', 'string']).columns
num_cols = X.select_dtypes(exclude='object').columns

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", model)
])

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=2,
    scoring="accuracy"
)

feature_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)
grid.fit(X_train, y_train)
print(grid.best_params_)
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

model1 = best_model.named_steps["model"]

# ดึง preprocessor
preprocess = best_model.named_steps["preprocess"]

# แยก encoder
ohe = preprocess.named_transformers_["cat"]

# ดึงชื่อ feature หลัง encode
encoded_cols = ohe.get_feature_names_out(cat_cols)

# รวม feature ทั้งหมด
all_features = list(num_cols) + list(encoded_cols)

# สร้าง importance
importance = pd.DataFrame({
    "feature": all_features,
    "importance": model1.feature_importances_
})

importance = importance.sort_values("importance", ascending=False)

# print(importance.head(20))
# feature_names = X.columns.tolist()
# joblib.dump(feature_names, r"C:\Python_Project\Loan_Data\features.pkl")
# joblib.dump(best_model, r"C:\Python_Project\Loan_Data\model.pkl")
# joblib.dump(le, r"C:\Python_Project\Loan_Data\label_encoder.pkl")

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()
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

@app.post("/predict")
def predict_loan(data: LoanInput):
    # ใช้ตัวแปร best_model และ le ที่อยู่ใน Memory ได้เลย
    input_df = pd.DataFrame([data.dict()])
    input_df = input_df[feature_names]
    
    # 1. ใช้ predict_proba เพื่อเอาค่าความน่าจะเป็น
    # ผลลัพธ์จะเป็น list ของ list เช่น [[0.06342, 0.93658]]
    probabilities = best_model.predict_proba(input_df)[0]
    
    # 2. ดึงชื่อ Class จาก Label Encoder (เช่น ['Good', 'Bad'] หรือ ['0', '1'])
    class_names = le.classes_
    
    # 3. สร้าง Dictionary ผลลัพธ์ และปัดเศษทศนิยม
    # เราใช้ .item() เพื่อแปลง numpy float เป็น python float ป้องกัน Error เดิม
    result_detail = {}
    for i, class_name in enumerate(class_names):
        percentage = round(probabilities[i] * 100, 2)
        result_detail[str(class_name)] = percentage
        
    return {"prediction": result_detail}

# ส่วนสำคัญ: ป้องกันไม่ให้รัน API อัตโนมัติเวลาเราต้องการแค่เทรน Model
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
