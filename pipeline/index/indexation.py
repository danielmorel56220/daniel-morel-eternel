"""
Indexation vectorielle — Daniel Morel Éternel
Découpe les textes en chunks et les stocke dans ChromaDB.
"""

import re
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.paths import BASE_VECTORIELLE, TEXTES_DIR

TEXTES = TEXTES_DIR
DB_DIR = BASE_VECTORIELLE
DB_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 800      # caractères par chunk
CHUNK_OVERLAP = 150   # chevauchement pour ne pas couper les idées


def decouper(texte, source):
    """Découpe un texte en chunks avec chevauchement."""
    chunks = []
    debut = 0
    while debut < len(texte):
        fin = debut + CHUNK_SIZE
        chunk = texte[debut:fin]
        if chunk.strip():
            chunks.append({
                "texte": chunk.strip(),
                "source": source
            })
        debut += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def main():
    print("Chargement du modèle d'embeddings (première fois = téléchargement ~500Mo)...")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Supprimer la collection si elle existe déjà (reprise propre)
    try:
        client.delete_collection("daniel_morel")
    except Exception:
        pass

    collection = client.create_collection(
        name="daniel_morel",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    fichiers = list(TEXTES.rglob("*.txt"))
    total = len(fichiers)
    print(f"{total} fichiers à indexer...\n")

    tous_chunks = []
    for fichier in fichiers:
        try:
            texte = fichier.read_text(encoding="utf-8", errors="ignore")
            if texte.strip():
                source = str(fichier.relative_to(TEXTES))
                tous_chunks.extend(decouper(texte, source))
        except Exception as e:
            print(f"  Erreur: {fichier.name} — {e}")

    print(f"{len(tous_chunks)} chunks créés. Indexation en cours...\n")

    # Insérer par lots de 500
    BATCH = 500
    for i in range(0, len(tous_chunks), BATCH):
        lot = tous_chunks[i:i+BATCH]
        ids = [f"chunk_{i+j}" for j in range(len(lot))]
        documents = [c["texte"] for c in lot]
        metadatas = [{"source": c["source"]} for c in lot]

        collection.add(ids=ids, documents=documents, metadatas=metadatas)

        progression = min(i + BATCH, len(tous_chunks))
        print(f"  {progression}/{len(tous_chunks)} chunks indexés")

    print(f"\nBase vectorielle prête dans : {DB_DIR}")
    print(f"Total chunks : {collection.count()}")


if __name__ == "__main__":
    main()
