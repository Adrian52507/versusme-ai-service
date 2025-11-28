from fastapi import FastAPI
from pydantic import BaseModel
from recommender import recommend_full
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Permitir requests desde tu backend Node
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # cámbialo si quieres limitar
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserInput(BaseModel):
    gender: str
    age: int
    height_cm: int
    weight_kg: int
    activity_0_4: int
    goal_str: str

@app.post("/predict_full")
def predict(data: UserInput):
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
    return {"msg": "IA service running!"}
