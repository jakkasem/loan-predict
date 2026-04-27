import io
import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# โหลดผ่าน joblib จาก Memory Stream แทน Path
pipeline = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
feature_names = joblib.load(os.path.join(BASE_DIR, "features.pkl"))

def predict(data_dict):
    df = pd.DataFrame([data_dict])

    # เติม feature ที่ขาด
    for col in feature_names:
        if col not in df:
            df[col] = None

    df = df[feature_names]

    # predict probability
    proba = pipeline.predict_proba(df)[0]

    # ใช้ classes_ จาก pipeline
    classes = list(pipeline.classes_)

    result = {
        str(classes[i]): round(float(proba[i]) * 100, 2)
        for i in range(len(classes))
    }

    return result