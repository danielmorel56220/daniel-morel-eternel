"""
Indexation des transcriptions audio/vidéo dans Supabase.
Lit les .docx sur les 3 disques, découpe en chunks, insère avec reprise.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import psycopg2
from docx import Document
from dotenv import load_dotenv
from fastembed import TextEmbedding

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.paths import LOGS_DIR, ROOT

load_dotenv(ROOT / ".env")

DB_URL = os.getenv("SUPABASE_DB_URL", "")
PROGRESS_FILE = LOGS_DIR / "indexation_transcriptions_done.json"
CHUNK_SIZE = 800
OVERLAP = 150

DISQUES = [
    Path("/Volumes/clean Disk"),
    Path("/Volumes/Securité 500 g"),
    Path("/Volumes/hysta"),
]

def charger_progres():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return set(json.load(f))
    return set()

def sauvegarder_progres(deja_faits):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(list(deja_faits), f)

def extraire_texte(docx_path):
    try:
        doc = Document(str(docx_path))
        paragraphes = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        # Ignorer les 4 premières lignes (métadonnées : nom, fichier, langue, date)
        if len(paragraphes) > 4:
            paragraphes = paragraphes[4:]
        return " ".join(paragraphes)
    except Exception as e:
        print(f"  ERREUR lecture {docx_path.name}: {e}")
        return ""

def decouper(texte, taille=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    debut = 0
    while debut < len(texte):
        fin = min(debut + taille, len(texte))
        chunk = texte[debut:fin].strip()
        if chunk:
            chunks.append(chunk)
        debut += taille - overlap
    return chunks

def main():
    print("Chargement du modèle d'embedding...")
    embedder = TextEmbedding("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    print("Connexion Supabase...")
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    deja_faits = charger_progres()
    print(f"Fichiers déjà indexés : {len(deja_faits)}")

    # Trouver tous les .docx sur les 3 disques
    tous_docx = []
    for disque in DISQUES:
        if disque.exists():
            found = list(disque.rglob("*.docx"))
            found = [f for f in found if not f.name.startswith("._")]
            print(f"  {disque.name}: {len(found)} .docx")
            tous_docx.extend(found)
        else:
            print(f"  {disque.name}: NON MONTÉ")

    print(f"\nTotal: {len(tous_docx)} fichiers .docx")
    a_faire = [f for f in tous_docx if str(f) not in deja_faits]
    print(f"À indexer: {len(a_faire)} fichiers\n")

    total_chunks = 0
    for i, docx_path in enumerate(a_faire, 1):
        print(f"[{i}/{len(a_faire)}] {docx_path.name}...", end=" ", flush=True)

        texte = extraire_texte(docx_path)
        if len(texte) < 50:
            print("(trop court, ignoré)")
            deja_faits.add(str(docx_path))
            continue

        chunks = decouper(texte)
        source = f"Transcription: {docx_path.stem}"

        # Calculer les embeddings pour tous les chunks du fichier
        embeddings = list(embedder.embed(chunks))

        # Insérer en base
        for chunk, emb in zip(chunks, embeddings):
            vecteur = [float(x) for x in emb]
            cur.execute(
                "INSERT INTO documents (contenu, source, embedding) VALUES (%s, %s, %s)",
                (chunk, source, vecteur)
            )
        conn.commit()

        total_chunks += len(chunks)
        deja_faits.add(str(docx_path))

        print(f"{len(chunks)} chunks ✓  (total: {total_chunks:,})")

        # Sauvegarder la progression tous les 10 fichiers
        if i % 10 == 0:
            sauvegarder_progres(deja_faits)

    sauvegarder_progres(deja_faits)
    cur.close()
    conn.close()

    print(f"\nTerminé. {total_chunks:,} chunks indexés depuis {len(a_faire)} fichiers.")

if __name__ == "__main__":
    main()
