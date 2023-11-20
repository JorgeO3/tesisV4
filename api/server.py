from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from .e_model import EModel
from .ts_model import TSModel
from .wvp_model import WVPModel

app = FastAPI()

e_model = EModel()
ts_model = TSModel()
wvp_model = WVPModel()


class Request(BaseModel):
    model: str
    payload: list[list[float]]


@app.get("/")
def read_root():
    return "the API is working!"


@app.post("/model")
async def predict(req: Request):
    model_name, payload = req.model, req.payload

    if model_name == "ts":
        return ts_model.inference(payload)
    if model_name == "wvp":
        return wvp_model.inference(payload)
    if model_name == "e":
        return e_model.inference(payload)
    else:
        return {"message": "Model not found"}
