import numpy as np
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
from pickle import load
from pandas import read_csv
from fastapi.responses import FileResponse

app = FastAPI()
with open('elastic_net.pickle', 'rb') as file:
    model = load(file)
with open('scaler.pickle', 'rb') as file:
    scaler = load(file)
class Item(BaseModel):
    year: int
    km_driven: int
    mileage: float
    engine: float
    max_power: float
    seats: int


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> dict: #чтобы не возиться с нелогичными категориальными сиденьями как в блокноте было принято решение использовать ElasticNet -- это не повлияет на
    lst = np.array([item.year, item.km_driven, item.mileage,item.engine, item.max_power, item.seats]).reshape(1, -1)
    return {"result": model.predict(scaler.transform(lst))[0]}


@app.post("/predict_items")
def predict_items(items: UploadFile=File(...)):
    data = read_csv(items.file, index_col=0)
    preds = model.predict(scaler.transform(data))
    data['predictions']= preds
    data.to_csv('response.csv')
    return FileResponse('response.csv')