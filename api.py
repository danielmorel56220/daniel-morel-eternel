"""
Daniel Morel Éternel — API de recherche
Lancement : uvicorn api:app --host 0.0.0.0 --port $PORT
"""

import os
import requests as http
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastembed import TextEmbedding
import anthropic

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://igdodyugqyeprtufohea.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "sb_publishable_325QLr_MStROaoHZ50DypQ_8rHCU0cH")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")

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


@app.get("/")
def racine():
    return {"message": "Daniel Morel Éternel — API en ligne ✅"}


@app.post("/recherche", response_model=Reponse)
def recherche(body: Question):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="La question est vide.")

    embedding = [float(x) for x in next(iter(embedder.embed([body.question])))]

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }

    resp = http.post(
        f"{SUPABASE_URL}/rest/v1/rpc/recherche_documents",
        headers=headers,
        json={"query_embedding": embedding, "nb_resultats": body.nb_resultats},
        timeout=15,
    )

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Supabase: {resp.text}")

    rows = resp.json()
    extraits = [
        Extrait(contenu=r["contenu"], source=r["source"], similarite=round(r["similarite"], 4))
        for r in rows
    ]

    return Reponse(question=body.question, extraits=extraits)


class ReponseChat(BaseModel):
    question: str
    reponse: str


def reformuler_question(question: str, client: anthropic.Anthropic) -> str:
    """Reformule la question en termes PNL/Ennéagramme pour améliorer la recherche."""
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system="""Tu es un expert en PNL et Ennéagramme. Ton rôle est de reformuler une question en langage technique PNL/Ennéagramme pour optimiser une recherche vectorielle dans une base de connaissances.

Reformule la question en ajoutant les termes techniques pertinents : noms de techniques (ancrage, recadrage, dissociation, Core Process, métaprogrammes, niveaux logiques, multi-dissociation, STRATEX...), types Ennéagramme (type 1 à 9, cellule de crise, EGO, centres instinctif/émotionnel/mental...), niveaux PNL (Bases, Technicien, Praticien, Maître Praticien).

Réponds UNIQUEMENT avec la question reformulée, rien d'autre.""",
        messages=[{"role": "user", "content": question}]
    )
    return msg.content[0].text.strip()


@app.post("/chat", response_model=ReponseChat)
def chat(body: Question):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="La question est vide.")

    if not ANTHROPIC_KEY:
        raise HTTPException(status_code=503, detail="Clé Anthropic manquante.")

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    # Reformuler la question avant la recherche
    question_enrichie = reformuler_question(body.question, client)

    # Recherche avec la question enrichie
    embedding = [float(x) for x in next(iter(embedder.embed([question_enrichie])))]
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    resp = http.post(
        f"{SUPABASE_URL}/rest/v1/rpc/recherche_documents",
        headers=headers,
        json={"query_embedding": embedding, "nb_resultats": 8},
        timeout=15,
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Supabase: {resp.text}")

    extraits = resp.json()
    contexte = "\n\n---\n\n".join(
        f"[Source: {e['source']}]\n{e['contenu']}" for e in extraits
    )

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system="""Tu es Daniel Morel, formateur expert en PNL, Ennéagramme, management et développement personnel depuis plus de 30 ans (Daniel Morel Institut / Résonance Formation).

## TA MISSION
Répondre aux questions en t'appuyant exclusivement sur les extraits fournis, qui proviennent de tes propres formations, écrits et supports pédagogiques. Tu es une mémoire vivante de 30 ans de travail.

## RÈGLES DE RÉPONSE

**Structure claire :**
- Commence par répondre directement à la question (pas d'introduction inutile)
- Organise ta réponse avec des titres ou des points si la réponse est longue
- Termine par une synthèse ou une question de relance si pertinent

**Contenu :**
- Appuie-toi sur les extraits fournis — c'est ta propre connaissance
- Si les extraits contiennent un protocole ou des étapes, donne-les complètement
- Si les extraits ne permettent pas de répondre à la question, dis-le clairement : "Je n'ai pas d'information précise sur ce point dans mes supports."
- Ne cite jamais les extraits mot pour mot — synthétise et reformule
- Ne cite jamais les noms de fichiers source

**Précision pédagogique :**
- Si la question porte sur un exercice PNL, indique le niveau (Bases / Technicien / Praticien / Maître Praticien)
- Si la question porte sur un type Ennéagramme, précise la dynamique EGO concernée
- Donne toujours un exemple concret ou une application pratique

**Ton :**
- Français, clair, structuré
- Pédagogique sans être condescendant
- Pas de formules creuses ("bien sûr", "excellente question", "en conclusion")""",
        messages=[
            {
                "role": "user",
                "content": f"Voici des extraits de mes travaux pertinents pour ta question :\n\n{contexte}\n\n---\n\nQuestion : {body.question}"
            }
        ]
    )

    return ReponseChat(question=body.question, reponse=message.content[0].text)
