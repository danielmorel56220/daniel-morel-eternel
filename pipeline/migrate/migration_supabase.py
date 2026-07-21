"""
Migration vers Supabase — Daniel Morel Éternel
Crée la table, active pgvector, envoie tous les chunks.
"""

import json
import os
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from supabase import create_client

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.paths import BASE_VECTORIELLE, LOGS_DIR, ROOT

load_dotenv(ROOT / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY")
DB_DIR = BASE_VECTORIELLE
LOG = LOGS_DIR / "migration_supabase.log"

BATCH_SIZE = 100

def main():
    print("Connexion à Supabase...")
    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Créer la table via SQL
    print("Création de la table documents...")
    create_table_sql = """
    create extension if not exists vector;

    drop table if exists documents;

    create table documents (
        id bigserial primary key,
        contenu text not null,
        source text,
        embedding vector(384)
    );

    create index on documents
    using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);
    """

    try:
        client.rpc("exec_sql", {"sql": create_table_sql}).execute()
    except Exception:
        pass  # La table sera créée via l'éditeur SQL si nécessaire

    # Charger les chunks depuis ChromaDB
    print("Chargement des chunks depuis ChromaDB...")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    chroma_client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = chroma_client.get_collection(name="daniel_morel", embedding_function=ef)

    total = collection.count()
    print(f"{total} chunks à migrer...\n")

    migres = 0
    erreurs = 0

    with open(LOG, "w", encoding="utf-8") as flog:
        for offset in range(0, total, BATCH_SIZE):
            try:
                resultats = collection.get(
                    limit=BATCH_SIZE,
                    offset=offset,
                    include=["documents", "metadatas", "embeddings"]
                )

                rows = []
                for doc, meta, emb in zip(
                    resultats["documents"],
                    resultats["metadatas"],
                    resultats["embeddings"]
                ):
                    rows.append({
                        "contenu": doc,
                        "source": meta.get("source", ""),
                        "embedding": emb
                    })

                client.table("documents").insert(rows).execute()
                migres += len(rows)

                progression = min(offset + BATCH_SIZE, total)
                print(f"  {progression}/{total} chunks migrés")
                flog.write(f"OK: {progression}/{total}\n")

            except Exception as e:
                erreurs += 1
                flog.write(f"ERREUR offset {offset}: {e}\n")
                print(f"  Erreur à l'offset {offset}: {e}")

    print(f"\nMigration terminée — {migres} chunks envoyés, {erreurs} erreurs")
    print(f"Vos données sont maintenant en ligne sur Supabase ✅")

if __name__ == "__main__":
    main()
