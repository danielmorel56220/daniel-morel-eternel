"""
Tagging sémantique — séquentiel, arrêt immédiat sur erreur crédit.
"""

import os
import json
import time
import psycopg2
import anthropic
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DB_URL = "postgresql://postgres:CG0xmL8kmikYT3tD@db.igdodyugqyeprtufohea.supabase.co:5432/postgres"
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
PROGRESS_FILE = Path("/Users/danielmorel/tagging_progress.json")
BATCH_SIZE = 20
LOG_FILE = Path("/Users/danielmorel/tagging.log")

SYSTEM_PROMPT = """Tu es un expert en PNL et Ennéagramme. Tu analyses des extraits de formations et tu assignes des tags précis.

Pour chaque extrait, réponds UNIQUEMENT avec un JSON valide sur une seule ligne, format exact :
{"theme": "...", "type_enneagramme": "...", "technique_pnl": "...", "niveau_logique": "..."}

Valeurs possibles :

theme : confiance_en_soi | estime_de_soi | ego_cellule_de_crise | motivation | rupture_amoureuse | deuil | relation_difficile | communication | emotions | peur | colere | tristesse | joie | stress | sens_de_vie | mission_de_vie | prise_de_decision | management | leadership | parentalite | identite | croyances | valeurs | hypnose | meditation | general | administratif

type_enneagramme : type_1 | type_2 | type_3 | type_4 | type_5 | type_6 | type_7 | type_8 | type_9 | tous | aucun

technique_pnl : ancrage | recadrage | dissociation | core_process | niveaux_logiques | metaprogrammes | objectif_bien_forme | synchronisation | cnv | multi_dissociation | timeline | sous_modalites | swish | carre_magique | marelle_croyance | aucune

niveau_logique : environnement | comportement | competence | croyance | identite | mission | aucun

Si administratif : {"theme": "administratif", "type_enneagramme": "aucun", "technique_pnl": "aucune", "niveau_logique": "aucun"}"""


def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


def charger_progres():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"dernier_id": 0, "traites": 0}


def sauvegarder_progres(state):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(state, f)


def tagger_batch(chunks, client):
    messages_content = ""
    for i, (chunk_id, contenu) in enumerate(chunks):
        messages_content += f"EXTRAIT {i+1} (id={chunk_id}):\n{contenu[:300]}\n\n"

    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Tague ces {len(chunks)} extraits. Réponds avec EXACTEMENT {len(chunks)} lignes JSON :\n\n{messages_content}"
        }]
    )

    lines = [l.strip() for l in msg.content[0].text.strip().split("\n") if l.strip().startswith("{")]
    results = []
    for line in lines:
        try:
            results.append(json.loads(line))
        except Exception:
            results.append({"theme": "general", "type_enneagramme": "aucun", "technique_pnl": "aucune", "niveau_logique": "aucun"})

    while len(results) < len(chunks):
        results.append({"theme": "general", "type_enneagramme": "aucun", "technique_pnl": "aucune", "niveau_logique": "aucun"})

    return list(zip([r[0] for r in chunks], results))


def main(limite=None):
    if not ANTHROPIC_KEY:
        log("ERREUR: ANTHROPIC_API_KEY manquante")
        return

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    state = charger_progres()
    log(f"Reprise depuis id={state['dernier_id']} — déjà traités: {state['traites']:,}")

    cur.execute("SELECT COUNT(*) FROM documents WHERE theme IS NULL")
    total = cur.fetchone()[0]
    log(f"Chunks à taguer: {total:,}\n")

    traites_session = 0

    while True:
        if limite and traites_session >= limite:
            log(f"\nLimite de {limite} chunks atteinte. Arrêt.")
            break

        cur.execute(
            "SELECT id, contenu FROM documents WHERE theme IS NULL AND id > %s ORDER BY id LIMIT %s",
            (state["dernier_id"], BATCH_SIZE)
        )
        rows = cur.fetchall()
        if not rows:
            log("\nTous les chunks sont tagués.")
            break

        try:
            rows_tags = tagger_batch(rows, client)

            for chunk_id, tag in rows_tags:
                cur.execute(
                    "UPDATE documents SET theme=%s, type_enneagramme=%s, technique_pnl=%s, niveau_logique=%s WHERE id=%s",
                    (tag.get("theme"), tag.get("type_enneagramme"), tag.get("technique_pnl"), tag.get("niveau_logique"), chunk_id)
                )
            conn.commit()

            state["dernier_id"] = rows[-1][0]
            state["traites"] += len(rows)
            traites_session += len(rows)
            sauvegarder_progres(state)

            log(f"[{state['traites']:,}/{total:,}] id={state['dernier_id']} — {rows_tags[0][1].get('theme')}")
            time.sleep(0.5)

        except anthropic.BadRequestError as e:
            if "credit balance" in str(e):
                log(f"\nARRÊT — solde insuffisant. Aucune retry. Rechargez le compte.")
                break
            else:
                log(f"\nARRÊT — erreur inattendue: {e}")
                break
        except Exception as e:
            log(f"\nARRÊT — erreur: {e}")
            break

    cur.close()
    conn.close()
    log(f"\nSession terminée. {traites_session} chunks tagués cette session.")


if __name__ == "__main__":
    main()
