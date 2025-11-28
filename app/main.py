from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.recommender import recommend_full

app = FastAPI()

# Permitir CORS (Vercel → Railway)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Luego puedes poner tu dominio específico
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/recommend")
async def recommend(payload: dict):
    """ Recibe el JSON del frontend y devuelve la recomendación completa """
    return recommend_full(payload)

@app.get("/")
async def home():
    return {"status": "ok", "msg": "VersusMe AI is running"} 
