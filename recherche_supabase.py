"""
Recherche dans Daniel Morel Éternel — via Supabase cloud
Usage : python3 recherche_supabase.py "votre question" [nombre_resultats]
"""

import sys
import psycopg2
from sentence_transformers import SentenceTransformer

DB_URL = "postgresql://postgres:CG0xmL8kmikYT3tD@db.igdodyugqyeprtufohea.supabase.co:5432/postgres"
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
