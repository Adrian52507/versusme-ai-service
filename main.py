from fastapi import FastAPI
from pydantic import BaseModel
from recommender import recommend_full

app = FastAPI()

class RequestModel(BaseModel):
    gender: str
    age: int
    height_cm: int
    weight_kg: int
    activity_0_4: int
    goal_str: str

@app.post("/recommend")
def recommend(data: RequestModel):
    result = recommend_full(
        data.gender,
        data.age,
        data.height_cm,
        data.weight_kg,
        data.activity_0_4,
        data.goal_str
    )
    return result

@app.get("/")
def root():
    return {"status": "IA microservice running"}
