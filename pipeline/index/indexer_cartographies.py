"""
Indexation des deux cartographies PNL et Ennéagramme dans Supabase.
"""

import hashlib
import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from fastembed import TextEmbedding

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.paths import DATA_DIR, ROOT

load_dotenv(ROOT / ".env")

DB_URL = os.getenv("SUPABASE_DB_URL", "")
CHUNK_SIZE = 800
OVERLAP = 150

FICHIERS = [
    DATA_DIR / "cartographie_pnl_exhaustive_2026-06-22.txt",
    DATA_DIR / "cartographie_enneagramme_exhaustive_2026-06-22.txt",
]

def chunker(texte):
    chunks, i = [], 0
    while i < len(texte):
        chunk = texte[i:i+CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        i += CHUNK_SIZE - OVERLAP
    return chunks

def hash_texte(t):
    return hashlib.md5(t.encode()).hexdigest()

print("Chargement du modèle fastembed...")
embedder = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

print("Connexion Supabase...")
conn = psycopg2.connect(DB_URL)
cur = conn.cursor()

cur.execute("SELECT md5(contenu) FROM documents")
deja_indexes = set(r[0] for r in cur.fetchall())
print(f"{len(deja_indexes)} chunks déjà dans la base\n")

total_chunks = 0

for filepath in FICHIERS:
    path = Path(filepath)
    texte = path.read_text(encoding="utf-8").strip()
    chunks = chunker(texte)
    nouveaux = [c for c in chunks if hash_texte(c) not in deja_indexes]

    print(f"{path.name} — {len(chunks)} chunks total, {len(nouveaux)} nouveaux à indexer")
    if not nouveaux:
        print("  Déjà indexé, on passe.\n")
        continue

    source = f"Cartographies/{path.name}"
    embeddings = list(embedder.embed(nouveaux))

    for chunk, emb in zip(nouveaux, embeddings):
        emb_list = [float(x) for x in emb]
        cur.execute(
            "INSERT INTO documents (contenu, source, embedding) VALUES (%s, %s, %s::vector)",
            (chunk, source, str(emb_list))
        )
        deja_indexes.add(hash_texte(chunk))

    conn.commit()
    total_chunks += len(nouveaux)
    print(f"  ✅ {len(nouveaux)} chunks indexés\n")

cur.close()
conn.close()
print(f"Terminé — {total_chunks} chunks ajoutés au total.")
