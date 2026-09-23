"""
Daniel Morel Éternel — API de recherche
Lancement : uvicorn app.api:app --host 0.0.0.0 --port $PORT
"""

import os
import re
from pathlib import Path

import anthropic
import requests as http
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastembed import TextEmbedding
from pydantic import BaseModel

from app.llm_free import generer_gratuit
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
# auto | free | ollama | anthropic | rag
# free = Groq/Gemini (si clés) → OVH UE anonyme → Pollinations → Ollama → RAG
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "free").strip().lower()

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


_RE_TIMESTAMP_LINE = re.compile(
    r"^\s*\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?\s*$", re.MULTILINE
)
_RE_TIMESTAMP_PREFIX = re.compile(
    r"^\s*\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?\s+", re.MULTILINE
)
_RE_BRACKET_TIME = re.compile(
    r"\[\s*\d+(?:[.,]\d+)?s?\s*(?:→|->|–|-)\s*\d+(?:[.,]\d+)?s?\s*\]"
)
_RE_SRT_INDEX = re.compile(r"^\s*\d+\s*$", re.MULTILINE)
_RE_SPACES = re.compile(r"[ \t]+")
_RE_NEWLINES = re.compile(r"\n{3,}")
_RE_DIALOGUE_FAIBLE = re.compile(
    r"^(il|elle|on|je|tu|nous|vous)\s+dit\b|^ben\b|^euh\b",
    re.IGNORECASE,
)
_MOTS_VIDES = {
    "les", "des", "une", "un", "le", "la", "de", "du", "et", "en", "au", "aux",
    "que", "qui", "quoi", "dont", "est", "sont", "pour", "pas", "plus", "avec",
    "dans", "sur", "par", "ce", "ces", "cet", "cette", "il", "elle", "on", "nous",
    "vous", "ils", "elles", "a", "à", "d", "l", "y", "ne", "se", "sa", "son",
    "ses", "mon", "ton", "ma", "ta", "mes", "tes", "notre", "votre", "leurs",
    "ou", "où", "si", "comme", "mais", "donc", "alors", "très", "aussi",
    "quels", "quelles", "quel", "quelle", "comment", "pourquoi", "quand",
}


def _nettoyer_extrait(texte: str) -> str:
    """Enlève timestamps vidéo, indices SRT, espaces sales."""
    if not texte:
        return ""
    t = texte.replace("\r\n", "\n").replace("\r", "\n")
    t = _RE_TIMESTAMP_LINE.sub(" ", t)
    t = _RE_TIMESTAMP_PREFIX.sub(" ", t)
    t = _RE_BRACKET_TIME.sub(" ", t)
    t = _RE_SRT_INDEX.sub(" ", t)
    t = t.replace(">>", " ").replace("…", "...")
    t = _RE_SPACES.sub(" ", t)
    t = _RE_NEWLINES.sub("\n\n", t)
    t = t.strip(" \n-–—:;")
    # Couper un début tronqué du type "éussissent" / "uveau"
    if t and t[0].islower() and " " in t[:40]:
        t = t.split(" ", 1)[1]
    return t.strip()


def _mots_cles(question: str) -> set[str]:
    mots = re.findall(r"[a-zàâäéèêëïîôùûüçœ-]{3,}", question.lower())
    return {m for m in mots if m not in _MOTS_VIDES}


def _decouper_phrases(texte: str) -> list[str]:
    brut = re.split(r"(?<=[.!?…])\s+|\n+", texte)
    phrases = []
    for p in brut:
        p = _RE_BRACKET_TIME.sub(" ", p)
        p = p.strip(" \n\"'«»[]")
        if len(p) < 50 or len(p) > 280:
            continue
        if re.search(r"\d{1,2}:\d{2}|\d+\.\d+s", p):
            continue
        if p[0].islower():
            continue
        if _RE_DIALOGUE_FAIBLE.search(p):
            continue
        if p.count(" ") < 6:
            continue
        phrases.append(p)
    return phrases


def _score_phrase(phrase: str, cles: set[str]) -> float:
    pl = phrase.lower()
    score = 0.0
    hits = 0
    for m in cles:
        if re.search(rf"\b{re.escape(m)}\b", pl) or m in pl:
            score += 1.2
            hits += 1
    if hits == 0:
        return 0.0
    if any(x in pl for x in ("parce que", "c'est", "cela permet", "alors", "donc", "quand on")):
        score += 0.3
    if "écoute" in pl or "ecoute" in pl:
        score += 0.8
    return score


def _dedupliquer(phrases: list[str]) -> list[str]:
    vues = []
    out = []
    for p in phrases:
        cle = re.sub(r"\W+", "", p.lower())[:80]
        if any(cle[:50] in v or v[:50] in cle for v in vues):
            continue
        vues.append(cle)
        out.append(p)
    return out


def _reponse_secours(question: str, extraits: list) -> str:
    """Réponse lisible sans Claude : synthétise les meilleurs passages nettoyés."""
    if not extraits:
        return (
            "Je suis là, mais je ne trouve pas d'extrait assez proche dans mes enseignements "
            "pour cette question. Reformulez un peu (écoute, relation, émotion, décision…) "
            "et je réessaierai."
        )

    cles = _mots_cles(question)
    candidates: list[tuple[float, str]] = []
    for e in extraits[:8]:
        texte = _nettoyer_extrait(e.get("contenu") or "")
        if not texte:
            continue
        for phrase in _decouper_phrases(texte):
            sc = _score_phrase(phrase, cles)
            if sc >= 1.2:
                candidates.append((sc, phrase))

    candidates.sort(key=lambda x: x[0], reverse=True)
    meilleures = _dedupliquer([p for _, p in candidates])[:4]

    if not meilleures:
        # Dernier recours : 2 extraits courts nettoyés
        bouts = []
        for e in extraits[:2]:
            t = _nettoyer_extrait(e.get("contenu") or "")
            if t:
                bouts.append(t[:280].rsplit(" ", 1)[0] + "…")
        if not bouts:
            return (
                "Je trouve des éléments dans mes enseignements, mais trop bruités "
                "(transcriptions brutes). Reposez la question avec un mot-clé plus précis."
            )
        meilleures = bouts

    puces = "\n".join(f"• {p}" for p in meilleures)
    return (
        f"Au sujet de « {question.strip()} », voici l’essentiel que j’en tire "
        f"de mes enseignements :\n\n"
        f"{puces}\n\n"
        "Si vous voulez, donnez-moi une situation concrète (couple, travail, famille) "
        "et on l’applique ensemble."
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
    utiliser_free = LLM_PROVIDER in ("auto", "free")
    utiliser_ollama = LLM_PROVIDER in ("auto", "free", "ollama")
    if LLM_PROVIDER == "rag":
        utiliser_anthropic = False
        utiliser_free = False
        utiliser_ollama = False

    if utiliser_anthropic:
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        reformulee = reformuler_question(question, client)
        if reformulee:
            question_recherche = reformulee

    extraits = _rechercher_extraits(question_recherche, nb_resultats)
    if not extraits and question_recherche != question:
        extraits = _rechercher_extraits(question, nb_resultats)

    # Contexte propre pour les LLM (pas les timestamps bruts)
    extraits_propres = []
    for e in extraits[:6]:
        t = _nettoyer_extrait(e.get("contenu") or "")
        if t:
            if len(t) > 500:
                t = t[:500].rsplit(" ", 1)[0] + "…"
            extraits_propres.append(f"[Source: {e.get('source', '')}]\n{t}")
    contexte = "\n\n---\n\n".join(extraits_propres)
    q_lower = question.lower()
    consigne_metamodele = ""
    if "métamodèle" in q_lower or "metamodele" in q_lower or "meta modele" in q_lower:
        consigne_metamodele = (
            "Cette question porte sur le MÉTAMODÈLE : intro brève, puis 3 questions "
            "courtes au client (guillemets « »), une par violation linguistique repérable "
            "dans l'énoncé du client. Pas de théorie longue.\n"
        )
    user_content = (
        "RÈGLES STRICTES (prioritaires):\n"
        "- Réponds UNIQUEMENT à partir des EXTRAITS ci-dessous. N'invente aucun fait, "
        "aucune citation, aucune référence à des notes ou fichiers.\n"
        "- Guillemets « » : uniquement pour des questions à poser au client, ou pour une "
        "phrase présente mot pour mot dans les EXTRAITS.\n"
        "- Longueur : environ 120 à 280 mots (sauf demande explicite de développer).\n"
        "- Français correct, phrases claires, pas de style littéraire ni de remplissage.\n"
        f"{consigne_metamodele}\n"
        f"EXTRAITS:\n{contexte}\n\n---\n\nQUESTION: {question}"
    )

    # 1) Claude (optionnel / payant)
    if client is not None and LLM_PROVIDER in ("auto", "anthropic"):
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

    # 2) LLM gratuits (Groq/Gemini/OVH/Pollinations)
    if utiliser_free:
        texte, _fournisseur = generer_gratuit(system_prompt, user_content)
        if texte:
            return ReponseChat(question=question, reponse=texte)

    # 3) Ollama local
    if utiliser_ollama and _ollama_disponible():
        try:
            texte = _appeler_ollama(system_prompt, user_content)
            return ReponseChat(question=question, reponse=texte)
        except Exception:
            pass

    # 4) Secours extractif nettoyé
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
        "groq_cle_presente": bool(os.getenv("GROQ_API_KEY", "").strip()),
        "gemini_cle_presente": bool(
            (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
        ),
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
