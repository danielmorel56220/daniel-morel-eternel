"""
Deduplication Supabase — Daniel Morel Eternel
Lit la table par pages, identifie les doublons en Python, supprime par lots.
"""

import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.paths import ROOT

load_dotenv(ROOT / ".env")

DB_URL = os.getenv("SUPABASE_DB_URL", "")
PAGE = 2000
BATCH = 200

conn = psycopg2.connect(DB_URL)
cur = conn.cursor()

cur.execute("SELECT COUNT(*) FROM documents")
avant = cur.fetchone()[0]
print(f"Chunks avant : {avant:,}")

print("Lecture par pages...")
vus = {}
a_supprimer = []
offset = 0

while True:
    cur.execute(
        "SELECT id, md5(contenu) FROM documents ORDER BY id LIMIT %s OFFSET %s",
        (PAGE, offset)
    )
    rows = cur.fetchall()
    if not rows:
        break
    for row_id, h in rows:
        if h in vus:
            a_supprimer.append(row_id)
        else:
            vus[h] = row_id
    offset += PAGE
    print(f"  {offset:,} lignes lues — {len(a_supprimer):,} doublons trouves")

print(f"\n{len(a_supprimer):,} doublons a supprimer par lots de {BATCH}...")

supprimes = 0
for i in range(0, len(a_supprimer), BATCH):
    lot = a_supprimer[i:i+BATCH]
    cur.execute("DELETE FROM documents WHERE id = ANY(%s)", (lot,))
    supprimes += cur.rowcount
    conn.commit()
    if (i // BATCH) % 10 == 0:
        print(f"  {supprimes:,}/{len(a_supprimer):,} supprimes")

cur.execute("SELECT COUNT(*) FROM documents")
apres = cur.fetchone()[0]
cur.close()
conn.close()

print(f"\nChunks supprimes : {supprimes:,}")
print(f"Chunks apres     : {apres:,}")
print(f"Base allegee de  : {round(supprimes/avant*100)}%")
