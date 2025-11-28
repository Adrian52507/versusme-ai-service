from fastapi import FastAPI
import json
import os

app = FastAPI()

from recommender import load_models_safely, recommend_full

# Modelos se cargan la primera vez que alguien llama a /recommend
models_loaded = False

@app.post("/recommend")
async def recommend(data: dict):

    global models_loaded
    if not models_loaded:
        load_models_safely()
        models_loaded = True

    result = recommend_full(
        data["gender"],
        data["age"],
        data["height_cm"],
        data["weight_kg"],
        data["activity_0_4"],
        data["goal_str"]
    )
    return result

@app.get("/")
async def root():
    return {"status": "ok", "message": "IA running"}
