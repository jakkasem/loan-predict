import seaborn as sns
import numpy as np
import pandas as pd
import joblib
import os  

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

le = LabelEncoder()
model = RandomForestClassifier(random_state=42)

param_grid = {
 'model__n_estimators': [100],
 'model__max_features': ['sqrt'],
 'model__max_depth': [12], 
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

# joblib.dump(data, r"C:\Python_Project\Loan_Data_20260422\dice_data.pkl")
#data = data.drop('grade_cat', axis=1)
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
    cv=3,
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

# joblib.dump(fc1, r"C:\Python_Project\Loan_Data\model.pkl")
# joblib.dump(scale, r"C:\Python_Project\Loan_Data\scaler.pkl")
# joblib.dump(feature_names, r"C:\Python_Project\Loan_Data\features.pkl")
# joblib.dump(le, r"C:\Python_Project\Loan_Data\label_encoder.pkl")

feature_names = X.columns.tolist()
joblib.dump(feature_names, os.path.join(BASE_DIR, "features.pkl"))
joblib.dump(best_model, os.path.join(BASE_DIR, "model.pkl"))
joblib.dump(le, os.path.join(BASE_DIR, "label_encoder.pkl"))

