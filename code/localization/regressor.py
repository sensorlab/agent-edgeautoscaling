import joblib

# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.neural_network import MLPRegressor
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from typing import List

import numpy as np

import warnings
import logging

warnings.filterwarnings(action='ignore', category=UserWarning)
logging.basicConfig(level=logging.CRITICAL)

# pipeline = Pipeline([
#     ("scaler", StandardScaler()),
#     ("regressor", KNeighborsRegressor(weights="distance", p=1, metric="euclidean", n_jobs=-1)),
#     # ("regressor", MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=250, verbose=False))
#     ], verbose=True
# )

features, targets = joblib.load("./spring.pkl")

# model = pipeline.fit(features, targets)

model = joblib.load("trained_model.pkl")

unique_features = features[~targets.duplicated()]

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
        # prediction = model.predict(np.array(features.iloc[item.feature].tolist()).reshape(1, -1))
        prediction = model.predict(np.array(unique_features.iloc[item.feature].tolist()).reshape(1, -1))
        return {"x": prediction[0][0], "y": prediction[0][1]}
    except:
        raise HTTPException(status_code=400, detail="Error making prediction")

