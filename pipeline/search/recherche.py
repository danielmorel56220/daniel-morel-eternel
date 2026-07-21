"""
Recherche dans la base Daniel Morel Éternel
Usage : python3 recherche.py "votre question" [nombre_resultats]
"""

import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.paths import BASE_VECTORIELLE

DB_DIR = BASE_VECTORIELLE

def rechercher(question, n=5):
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(name="daniel_morel", embedding_function=ef)

    resultats = collection.query(
        query_texts=[question],
        n_results=n
    )

    docs = resultats["documents"][0]
    metas = resultats["metadatas"][0]

    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        print(f"\n{'='*60}")
        print(f"[Extrait {i}] Source : {meta['source']}")
        print(f"{'='*60}")
        print(doc)

    return docs, metas

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "ego"
    rechercher(question)
