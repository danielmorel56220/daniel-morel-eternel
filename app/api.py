"""
Daniel Morel Éternel — API de recherche
Lancement : uvicorn app.api:app --host 0.0.0.0 --port $PORT
"""

import os
from pathlib import Path

import anthropic
import requests as http
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastembed import TextEmbedding
from pydantic import BaseModel

from app.prompts import (
    PROMPT_DANIEL_ADMIN,
    PROMPT_DANIEL_PUBLIC,
    PROMPT_REFORMULATION,
)

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"

load_dotenv(ROOT / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://igdodyugqyeprtufohea.supabase.co")
SUPABASE_KEY = (
    os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_PUBLISHABLE_KEY") or ""
).strip()
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

app = FastAPI(title="Daniel Morel Éternel", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèle chargé une fois au démarrage
embedder = TextEmbedding("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


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


class ReponseChat(BaseModel):
    question: str
    reponse: str


def _headers_supabase() -> dict:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


def _rechercher_extraits(question: str, nb_resultats: int) -> list:
    embedding = [float(x) for x in next(iter(embedder.embed([question])))]
    resp = http.post(
        f"{SUPABASE_URL}/rest/v1/rpc/recherche_documents",
        headers=_headers_supabase(),
        json={"query_embedding": embedding, "nb_resultats": nb_resultats},
        timeout=15,
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Supabase: {resp.text}")
    return resp.json()


def reformuler_question(question: str, client: anthropic.Anthropic) -> str:
    """Reformule la question en termes PNL/Ennéagramme pour améliorer la recherche."""
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system=PROMPT_REFORMULATION,
        messages=[{"role": "user", "content": question}],
    )
    return msg.content[0].text.strip()


def _generer_reponse(
    question: str, system_prompt: str, nb_resultats: int = 8
) -> ReponseChat:
    if not ANTHROPIC_KEY:
        raise HTTPException(status_code=503, detail="Clé Anthropic manquante.")
    if not SUPABASE_KEY:
        raise HTTPException(status_code=503, detail="Clé Supabase manquante.")

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    question_enrichie = reformuler_question(question, client)
    extraits = _rechercher_extraits(question_enrichie, nb_resultats)
    contexte = "\n\n---\n\n".join(
        f"[Source: {e['source']}]\n{e['contenu']}" for e in extraits
    )

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Voici des extraits de mes travaux pertinents pour ta question :\n\n"
                    f"{contexte}\n\n---\n\nQuestion : {question}"
                ),
            }
        ],
    )
    return ReponseChat(question=question, reponse=message.content[0].text)


@app.get("/")
def racine():
    return {"message": "Daniel Morel Éternel — API en ligne ✅"}


@app.get("/sante")
def sante():
    """Vérifie rapidement si les clés nécessaires sont bien présentes (sans les afficher)."""
    return {
        "api": "ok",
        "anthropic_cle_presente": bool(ANTHROPIC_KEY),
        "supabase_url": SUPABASE_URL,
        "supabase_cle_presente": bool(SUPABASE_KEY),
        "supabase_cle_prefixe": (SUPABASE_KEY[:12] + "…") if SUPABASE_KEY else "",
    }


@app.get("/chat-public")
def chat_public_page():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/admin")
def admin_page():
    return FileResponse(FRONTEND_DIR / "admin.html")


@app.post("/recherche", response_model=Reponse)
def recherche(body: Question):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="La question est vide.")
    if not SUPABASE_KEY:
        raise HTTPException(status_code=503, detail="Clé Supabase manquante.")

    try:
        rows = _rechercher_extraits(body.question, body.nb_resultats)
        extraits = [
            Extrait(
                contenu=r["contenu"],
                source=r["source"],
                similarite=round(r["similarite"], 4),
            )
            for r in rows
        ]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur embedding/recherche: {type(e).__name__}: {e}"
        )

    return Reponse(question=body.question, extraits=extraits)


@app.post("/chat", response_model=ReponseChat)
def chat(body: Question):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="La question est vide.")
    return _generer_reponse(body.question, PROMPT_DANIEL_PUBLIC)


@app.post("/chat-admin", response_model=ReponseChat)
def chat_admin(body: Question, x_admin_password: str = Header(None)):
    if not ADMIN_PASSWORD or x_admin_password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Mot de passe admin incorrect.")
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="La question est vide.")
    return _generer_reponse(body.question, PROMPT_DANIEL_ADMIN)
