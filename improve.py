# improve.py

import pandas as pd
import joblib
import dice_ml

class ImproveModel:

    def __init__(self, model_path, le_path, data_path):

        # load model
        self.model = joblib.load(model_path)
        self.le = joblib.load(le_path)

        self.TARGET = "loan_condition_cat"

        # self.DROP_COLS = [
        #     'id','year','loan_condition_cat',
        #     'application_type','home_ownership_cat',
        #     'application_type_cat','issue_d'
        # ]

        # ===== prepare dice schema =====
        data = joblib.load(data_path)
        # data = pd.read_csv(data_path, low_memory=False)

        # data1 = data.drop(self.DROP_COLS, axis=1)
        # data1 = data1.dropna(subset=[self.TARGET])
        data1 = data.dropna(subset=[self.TARGET])

        X = data1.drop(self.TARGET, axis=1)
        y = data1[self.TARGET]
        # y = self.le.transform(y)

        df_dice = X.copy()
        df_dice[self.TARGET] = y

        continuous_features = X.select_dtypes(exclude='object').columns.tolist()

        d = dice_ml.Data(
            dataframe=df_dice,
            continuous_features=continuous_features,
            outcome_name=self.TARGET
        )

        m = dice_ml.Model(model=self.model, backend="sklearn")

        self.exp = dice_ml.Dice(d, m, method="genetic")

    # =============================
    # prepare input
    # =============================
    def _prepare_input(self, input_json):
        df = pd.DataFrame([input_json])

        # drop column ที่ไม่ใช้
        # df = df.drop(columns=self.DROP_COLS, errors="ignore")

        # 🔥 เพิ่มบรรทัดนี้
        df = df.drop(columns=[self.TARGET], errors="ignore")

        # fix type
        if "total_pymnt" in df.columns:
            df["total_pymnt"] = pd.to_numeric(df["total_pymnt"], errors="coerce")

        return df
        
    # =============================
    # main function
    # =============================
    def improve(self, input_json):

        query_df = self._prepare_input(input_json)

        # predict
        pred = self.model.predict(query_df)[0]
        pred_label = self.le.inverse_transform([pred])[0]

        permitted_range = {
            "annual_inc": [20000, 150000],
            "loan_amount": [1000, 20000],
            "dti": [0, 40],
            "interest_rate": [5, 25]
        }

        # generate cf
        dice_exp = self.exp.generate_counterfactuals(
            query_df,
            total_CFs=3,
            desired_class="opposite",
            features_to_vary="all",
            maxiterations=1000,   # 🔥 เพิ่ม
            stopping_threshold=0.5,
            permitted_range=permitted_range
        )

        cf_df = dice_exp.cf_examples_list[0].final_cfs_df

        def convert_numpy(obj):
            import numpy as np

            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        return convert_numpy({
            "current_prediction": str(pred_label),
            "suggestions": cf_df.to_dict(orient="records")
        })
    
_model_instance = ImproveModel(
    model_path=r"C:\Python_Project\Loan_Data\model.pkl",
    le_path=r"C:\Python_Project\Loan_Data\label_encoder.pkl",
    data_path=r"C:\Python_Project\Loan_Data\dice_data.pkl"
)

def improve(input_json):
    return _model_instance.improve(input_json)