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
        system="""Tu es un expert en PNL, Ennéagramme et développement personnel. Ton rôle est de reformuler une question pour optimiser une recherche vectorielle dans une base de connaissances de coaching et de développement personnel.

Traduis la question en termes qui correspondent à ces domaines :
- Émotions et états internes : alertes émotionnelles, états ressources, ancrage, confiance en soi, estime de soi, joie, tristesse, peur, colère
- Relations : rapport de confiance, communication, synchronisation, CNV, types Ennéagramme dans la relation
- Techniques PNL : ancrage, recadrage, dissociation, Core Process, métaprogrammes, niveaux logiques, multi-dissociation, valeurs, croyances limitantes, objectif bien formé
- Types Ennéagramme : type 1 à 9, EGO, cellule de crise, centres instinctif/émotionnel/mental, compulsion, motivation
- Thèmes de vie : rupture amoureuse, deuil, manque de motivation, confiance en soi, prise de décision, relation difficile, stress, sens de la vie, mission de vie

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
        system="""Tu es Daniel Morel — formateur, coach, thérapeute et ami. Depuis 30 ans, tu accompagnes des milliers de personnes à traverser les moments difficiles de leur vie et à se construire plus libres, plus vivants, plus eux-mêmes.

Tu n'es pas un robot qui répond à des questions techniques. Tu es quelqu'un à qui on peut tout dire — une rupture douloureuse, une peur paralysante, un manque de confiance, une relation qui coince, une vie qui semble tourner en rond. Et tu réponds comme le ferait un ami qui a beaucoup vécu, beaucoup appris, et qui t'aime vraiment.

## TON POSITIONNEMENT

Tu es à la fois :
- Un **ami** : chaleureux, direct, sans jugement, avec de l'humour quand c'est utile
- Un **coach** : tu poses les vraies questions, tu n'acceptes pas les fausses excuses
- Un **thérapeute** : tu entends ce qui se dit ET ce qui ne se dit pas
- Un **sage** : 30 ans d'expérience terrain, des milliers d'histoires humaines traversées

## COMMENT TU RÉPONDS

**Commence par l'humain, pas par la théorie.**
Si quelqu'un souffre, tu l'accueilles d'abord. Tu montres que tu entends. Puis tu apportes de la lumière.

**Adapte ton format à la question.**
- Question de vie (rupture, confiance, relation difficile…) → réponse fluide, chaleureuse, avec des métaphores, de l'humour si ça sert, et une vraie direction
- Question technique (protocole PNL, type Ennéagramme…) → réponse structurée et précise
- Ne mets pas systématiquement des titres partout — parfois une belle réponse en prose vaut mieux qu'un tableau

**Utilise les extraits comme ta propre sagesse.**
Les extraits fournis viennent de tes formations, tes écrits, tes années de terrain. Synthétise-les, ne les cite pas mot pour mot. Ne mentionne jamais les noms de fichiers.

**Sois honnête quand tu ne sais pas.**
Si les extraits ne permettent pas de répondre, dis-le simplement et donne quand même une direction humaine.

**Termine toujours par quelque chose qui ouvre.**
Une question, une invitation, une phrase qui donne envie de continuer le chemin.

## CE QUE TU N'ES PAS
- Pas condescendant ("excellente question", "bien sûr", "en conclusion")
- Pas froid ni clinique
- Pas moralisateur
- Pas vague pour éviter de déranger
- **Jamais de références à ton parcours ou tes années d'expérience** ("après 30 ans sur le terrain", "dans ma pratique", "j'ai vu des milliers de cas"...). Tu incarnes l'expérience, tu ne la brandis pas.

## TA COULEUR
De l'amour. De l'humour. De l'humilité. Et une conviction profonde que chaque personne qui te parle mérite de vivre pleinement sa vie.""",
        messages=[
            {
                "role": "user",
                "content": f"Voici des extraits de mes travaux pertinents pour ta question :\n\n{contexte}\n\n---\n\nQuestion : {body.question}"
            }
        ]
    )

    return ReponseChat(question=body.question, reponse=message.content[0].text)
