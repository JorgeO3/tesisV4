import numpy as np

from typing import Union
from fastapi import FastAPI

from ts_model import TSModel
from wvp_model import WVPModel


app = FastAPI()
ts_model = TSModel()
wvp_model = WVPModel()


@app.get("/")
def read_root():
    return "the API is working!"


@app.post("/model/{model_name}")
async def predict(model_name: str, data: list[list[float]]):
    if model_name == "ts":
        result = ts_model.inference(data)
        result_list = result.tolist()
        return {"data": result_list}
    if model_name == "wvp":
        result = ts_model.inference(data)
        result_list = result.tolist()
        return {"data": result_list}
    else:
        return {"message": "Model not found"}
