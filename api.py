"""
Daniel Morel Éternel — API de recherche
Lancement : ~/miniconda3/bin/uvicorn api:app --reload --port 8000
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
from sentence_transformers import SentenceTransformer

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:CG0xmL8kmikYT3tD@db.igdodyugqyeprtufohea.supabase.co:5432/postgres"
)
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

app = FastAPI(title="Daniel Morel Éternel", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement du modèle au démarrage (une seule fois)
model = SentenceTransformer(MODEL_NAME)


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

    embedding = model.encode(body.question).tolist()

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
