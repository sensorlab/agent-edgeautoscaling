import joblib

from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from typing import List

import numpy as np

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", KNeighborsRegressor(weights="distance", p=1, metric="euclidean", n_jobs=-1)),
    ], verbose=True
)

features, targets = joblib.load("./spring.pkl")

model = pipeline.fit(features, targets)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    # features: List[float]
    feature: int

@app.post("/predict")
async def predict(item: Item):
    try:
        # prediction = model.predict(np.array(item.features).reshape(1, -1))
        prediction = model.predict(np.array(features.iloc[item.feature].tolist()).reshape(1, -1))
        return {"x": prediction[0][0], "y": prediction[0][1]}
    except:
        raise HTTPException(status_code=400, detail="Error making prediction")

