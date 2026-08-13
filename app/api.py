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
# Côté serveur : préférer la clé secret (lit la base malgré RLS).
# La clé publishable seule ne voit rien si RLS est actif sans policy.
SUPABASE_KEY = (
    os.getenv("SUPABASE_KEY")
    or os.getenv("SUPABASE_SECRET_KEY")
    or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    or os.getenv("SUPABASE_PUBLISHABLE_KEY")
    or ""
).strip()
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
# IA locale gratuite (Ollama). Ex. http://127.0.0.1:11434 — inutilisable depuis Railway
# sauf si tu exposes Ollama via un tunnel.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2").strip()
# auto | ollama | anthropic
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").strip().lower()

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
    try:
        resp = http.post(
            f"{SUPABASE_URL}/rest/v1/rpc/recherche_documents",
            headers=_headers_supabase(),
            json={"query_embedding": embedding, "nb_resultats": nb_resultats},
            timeout=15,
        )
    except http.exceptions.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=(
                "Impossible de joindre Supabase "
                f"({SUPABASE_URL}). Vérifiez que le projet existe encore. "
                f"Détail: {type(e).__name__}"
            ),
        ) from e
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Supabase: {resp.text}")
    return resp.json()


def _anthropic_indisponible(exc: Exception) -> bool:
    """True si Claude est injoignable (crédits, auth, quota) — on bascule en secours."""
    low = str(exc).lower()
    marqueurs = (
        "credit balance is too low",
        "purchase credits",
        "plans & billing",
        "authentication",
        "invalid api key",
        "rate_limit",
        "overloaded",
        "529",
        "401",
        "402",
        "403",
    )
    return any(m in low for m in marqueurs)


def _reponse_secours(question: str, extraits: list) -> str:
    """Réponse sans Claude : s'appuie uniquement sur les extraits de la base."""
    if not extraits:
        return (
            "Je suis là, mais je ne trouve pas d'extrait assez proche dans mes enseignements "
            "pour cette question. Reformulez un peu (PNL, émotions, relation, décision…) "
            "et je réessaierai."
        )

    parties = []
    for e in extraits[:4]:
        texte = (e.get("contenu") or "").strip()
        if len(texte) > 700:
            texte = texte[:700].rsplit(" ", 1)[0] + "…"
        source = (e.get("source") or "enseignement").strip()
        parties.append(f"À partir de « {source} » :\n{texte}")

    corps = "\n\n".join(parties)
    return (
        f"Voici ce que je peux vous dire à partir de mes enseignements, "
        f"en lien avec votre question (« {question.strip()} ») :\n\n"
        f"{corps}\n\n"
        "Si vous voulez aller plus loin, précisez une situation concrète "
        "(relation, travail, décision, émotion) et nous continuerons."
    )


def _ollama_disponible() -> bool:
    try:
        r = http.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _appeler_ollama(system_prompt: str, user_content: str, max_tokens: int = 2000) -> str:
    """Appelle Ollama (API locale compatible /api/chat)."""
    resp = http.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "stream": False,
            "options": {"num_predict": max_tokens},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        },
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    texte = (data.get("message") or {}).get("content") or data.get("response") or ""
    if not texte.strip():
        raise RuntimeError("Ollama a renvoyé une réponse vide.")
    return texte.strip()


def reformuler_question(question: str, client: anthropic.Anthropic):
    """Reformule la question. Retourne None si Claude est indisponible."""
    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=PROMPT_REFORMULATION,
            messages=[{"role": "user", "content": question}],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        if _anthropic_indisponible(e):
            return None
        raise


def _generer_reponse(
    question: str, system_prompt: str, nb_resultats: int = 8
) -> ReponseChat:
    if not SUPABASE_KEY:
        raise HTTPException(status_code=503, detail="Clé Supabase manquante.")

    question_recherche = question
    client = None
    utiliser_anthropic = LLM_PROVIDER in ("auto", "anthropic") and bool(ANTHROPIC_KEY)
    utiliser_ollama = LLM_PROVIDER in ("auto", "ollama")

    if utiliser_anthropic:
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        reformulee = reformuler_question(question, client)
        if reformulee:
            question_recherche = reformulee

    extraits = _rechercher_extraits(question_recherche, nb_resultats)
    # Si la reformulation a donné 0 résultat, retenter avec la question brute
    if not extraits and question_recherche != question:
        extraits = _rechercher_extraits(question, nb_resultats)

    contexte = "\n\n---\n\n".join(
        f"[Source: {e['source']}]\n{e['contenu']}" for e in extraits
    )
    user_content = (
        f"Voici des extraits de mes travaux pertinents pour ta question :\n\n"
        f"{contexte}\n\n---\n\nQuestion : {question}"
    )

    # 1) Claude (si demandé et possible)
    if client is not None and LLM_PROVIDER != "ollama":
        try:
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
            )
            return ReponseChat(question=question, reponse=message.content[0].text)
        except Exception as e:
            if not _anthropic_indisponible(e):
                raise HTTPException(
                    status_code=502,
                    detail=f"Erreur Anthropic: {type(e).__name__}: {e}",
                ) from e
            # Crédits / auth → on tente Ollama puis secours RAG

    # 2) Ollama local (gratuit)
    if utiliser_ollama and _ollama_disponible():
        try:
            texte = _appeler_ollama(system_prompt, user_content)
            return ReponseChat(question=question, reponse=texte)
        except Exception:
            pass  # tombe sur le secours RAG

    # 3) Secours : extraits bruts
    return ReponseChat(question=question, reponse=_reponse_secours(question, extraits))


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
        "llm_provider": LLM_PROVIDER,
        "ollama_url": OLLAMA_BASE_URL,
        "ollama_model": OLLAMA_MODEL,
        "ollama_disponible": _ollama_disponible(),
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
    try:
        return _generer_reponse(body.question, PROMPT_DANIEL_PUBLIC)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur chat: {type(e).__name__}: {e}",
        ) from e


@app.post("/chat-admin", response_model=ReponseChat)
def chat_admin(body: Question, x_admin_password: str = Header(None)):
    if not ADMIN_PASSWORD or x_admin_password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Mot de passe admin incorrect.")
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="La question est vide.")
    return _generer_reponse(body.question, PROMPT_DANIEL_ADMIN)
