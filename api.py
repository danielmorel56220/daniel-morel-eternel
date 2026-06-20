"""
Daniel Morel Éternel — API de recherche
Lancement : uvicorn api:app --host 0.0.0.0 --port $PORT
"""

import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:CG0xmL8kmikYT3tD@db.igdodyugqyeprtufohea.supabase.co:5432/postgres"
)
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HF_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL}"

app = FastAPI(title="Daniel Morel Éternel", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_embedding(text: str) -> list[float]:
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    response = requests.post(
        HF_URL,
        headers=headers,
        json={"inputs": text, "options": {"wait_for_model": True}},
        timeout=30
    )
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Erreur embeddings : {response.text}")
    data = response.json()
    # L'API retourne [[valeurs]] pour une phrase
    if isinstance(data[0], list):
        return data[0]
    return data


class Question(BaseModel):
    question: str
    nb_resultats: int = 5


class Extrait(BaseModel):
    source: str
    contenu: str
    similarite: float


class Reponse(BaseModel):
    question: str
    extraits: list[Extrait]


@app.get("/")
def racine():
    return {"message": "Daniel Morel Éternel — API en ligne ✅"}


@app.post("/recherche", response_model=Reponse)
def recherche(body: Question):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="La question est vide.")

    embedding = get_embedding(body.question)

    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        cur.execute(
            "SELECT contenu, source, 1 - (embedding <=> %s::vector) AS similarite "
            "FROM documents ORDER BY embedding <=> %s::vector LIMIT %s",
            (str(embedding), str(embedding), body.nb_resultats)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    extraits = [
        Extrait(contenu=row[0], source=row[1], similarite=round(row[2], 4))
        for row in rows
    ]

    return Reponse(question=body.question, extraits=extraits)
