"""
Recherche dans Daniel Morel Éternel — via Supabase cloud
Usage : python3 pipeline/search/recherche_supabase.py "votre question" [nombre_resultats]
"""

import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.paths import ROOT

load_dotenv(ROOT / ".env")

DB_URL = os.getenv("SUPABASE_DB_URL", "")
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

def rechercher(question, n=5):
    print(f"Recherche : « {question} »\n")

    model = SentenceTransformer(MODEL_NAME)
    embedding = model.encode(question).tolist()

    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    cur.execute(
        "SELECT contenu, source, 1 - (embedding <=> %s::vector) AS similarite "
        "FROM documents ORDER BY embedding <=> %s::vector LIMIT %s",
        (str(embedding), str(embedding), n)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    for i, (contenu, source, sim) in enumerate(rows, 1):
        print(f"\n{'='*60}")
        print(f"[{i}] Source : {source}  (similarité : {sim:.2f})")
        print(f"{'='*60}")
        print(contenu[:600])

    return rows

if __name__ == "__main__":
    args = sys.argv[1:]
    n = 5
    if args and args[-1].isdigit():
        n = int(args[-1])
        args = args[:-1]
    question = " ".join(args) if args else "ego"
    rechercher(question, n)
