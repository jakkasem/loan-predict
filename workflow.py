# workflow.py

from langgraph.graph import StateGraph
from typing import TypedDict

from predict import predict
from improve import ImproveModel

# =============================
# init improve model (ครั้งเดียว)
# =============================
improver = ImproveModel(
    model_path=r"C:\Python_Project\Loan_Data\model.pkl",
    le_path=r"C:\Python_Project\Loan_Data\label_encoder.pkl",
    data_path=r"C:\Python_Project\Loan_Data\dice_data.pkl"
)

# =============================
# state
# =============================
class SuggestState(TypedDict):
    input: dict
    prediction: dict
    improve_result: dict | None


# =============================
# nodes
# =============================
def predict_node(state: SuggestState):
    result = predict(state["input"])
    return {"prediction": result}


def check_node(state: SuggestState):
    # 🔥 ใช้ key "0" = Good Loan
    threshold = 80
    good_score = float(state["prediction"].get("0", 0))

    if good_score <= threshold:
        return "improve"
    return "end"


def improve_node(state: SuggestState):
    result = improver.improve(state["input"])
    return {"improve_result": result}


# =============================
# build graph
# =============================
builder = StateGraph(SuggestState)

builder.add_node("predict", predict_node)
builder.add_node("improve", improve_node)

builder.set_entry_point("predict")

builder.add_conditional_edges(
    "predict",
    check_node,
    {
        "improve": "improve",
        "end": "__end__"
    }
)

builder.add_edge("improve", "__end__")

graph = builder.compile()


# =============================
# function สำหรับเรียกใช้
# =============================
def run_suggest(input_json: dict):
    result = graph.invoke({
        "input": input_json,
        "prediction": {},
        "improve_result": None
    })

    return {
        "prediction": result["prediction"],
        "suggestion": result.get("improve_result")
    }