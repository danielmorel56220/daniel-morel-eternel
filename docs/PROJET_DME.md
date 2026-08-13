# Daniel Morel Éternel — Documentation Projet

> Dernière mise à jour : juillet 2026  
> Projet RAG (Retrieval-Augmented Generation) — chatbot IA incarnant Daniel Morel

---

## 1. Vue d'ensemble

Un chatbot qui répond "dans la voix de Daniel Morel" en s'appuyant sur 30 ans d'enseignements (PNL, Ennéagramme, développement personnel). L'utilisateur pose une question, le système trouve les extraits les plus pertinents dans la base de connaissances, et Claude génère une réponse dans le style et la philosophie de Daniel Morel.

---

## 2. État actuel de la base de données

| Indicateur | Valeur |
|---|---|
| Chunks totaux | 52 318 |
| Chunks tagués | 52 318 (100%) |
| Sources normalisées | Oui (noms lisibles) |
| Dernière mise à jour | Juin 2026 |

### Répartition des thèmes (top)
| Thème | Chunks |
|---|---|
| general | ~11 900 |
| identite | ~5 200 |
| croyances | ~3 500 |
| emotions | ~3 500 |
| communication | ~3 300 |
| administratif | ~3 100 |
| prise_de_decision | ~2 200 |
| motivation | ~1 800 |
| relation_difficile | ~1 700 |
| ego_cellule_de_crise | ~1 400 |

---

## 3. Architecture technique

```
Fichiers originaux (Mac/disques)
        ↓
extraction.py — découpe en chunks 800 mots / overlap 150
        ↓
fastembed — transforme en vecteur 384 dimensions (local, gratuit)
        ↓
tagger_chunks.py — tags sémantiques via Claude Haiku (payant)
        ↓
Supabase PostgreSQL + pgvector (cloud, gratuit)
        ↓
api.py FastAPI — Railway (cloud, gratuit)
        ↓
Claude Haiku (reformule question) + Claude Sonnet (génère réponse)
        ↓
Interface utilisateur (frontend/index.html + frontend/admin.html)
```

---

## 4. Stack technique

| Composant | Technologie | Hébergement | Coût |
|---|---|---|---|
| Base de données | PostgreSQL + pgvector | Supabase (cloud) | Gratuit |
| Embeddings | fastembed + paraphrase-multilingual-MiniLM-L12-v2 | Local (Mac) | Gratuit |
| Backend API | FastAPI + uvicorn | Railway | Gratuit |
| IA réponse | Claude Sonnet 4.6 | Anthropic API | ~$0.01/question |
| IA reformulation | Claude Haiku 4.5 | Anthropic API | ~$0.001/question |
| IA taguage | Claude Haiku 4.5 | Anthropic API | ~$1.50/1000 chunks |
| Interface | HTML/CSS vanilla | Servi par FastAPI | Gratuit |

---

## 5. Connexions et accès

Les secrets (Supabase, Anthropic, admin) sont **uniquement** dans `.env` à la racine.
Ne jamais les coller dans ce fichier ni les committer.

### Variables attendues dans `.env`
```env
SUPABASE_URL=...
SUPABASE_DB_URL=...
SUPABASE_DB_PASSWORD=...
SUPABASE_PUBLISHABLE_KEY=...
SUPABASE_SECRET_KEY=...
ANTHROPIC_API_KEY=...
ADMIN_PASSWORD=...
```

### Liens utiles (accès chat)

> Fiche dédiée, plus claire : **[LIENS_ACCES.md](LIENS_ACCES.md)**

| Usage | Adresse |
|---|---|
| **Public** (clients) | https://daniel-morel-eternel-production.up.railway.app/chat-public |
| **Propriétaires** (mot de passe) | https://daniel-morel-eternel-production.up.railway.app/admin |

Mot de passe admin = variable **`ADMIN_PASSWORD`** (Railway Variables / `.env` local).  
Ne jamais le coller dans ce fichier ni le committer.

```
Supabase dashboard : https://supabase.com → projet igdodyugqyeprtufohea
Railway production : https://daniel-morel-eternel-production.up.railway.app
Anthropic console  : https://console.anthropic.com
```

### Lancement local de l'API
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

### Scripts pipeline (exemples)
```bash
python3 pipeline/extract/extraction.py
python3 -c "import sys; sys.path.insert(0,'.'); from pipeline.tag.tagger_chunks import main; main(limite=100)"
python3 pipeline/search/recherche_supabase.py "manque de confiance"
bash scripts/orchestrateur.sh
```

---

## 6. Structure des fichiers

```
Daniel Morel Eternel/
├── app/                            # Production (API)
│   ├── api.py                      # FastAPI — point d'entrée
│   └── prompts/                    # System prompts Claude
├── pipeline/                       # Scripts one-shot (jamais déployés)
│   ├── paths.py                    # Chemins relatifs du projet
│   ├── extract/                    # Extraction disques → textes
│   ├── index/                      # Indexation / embeddings
│   ├── tag/                        # Taguage sémantique
│   ├── migrate/                    # Migration + dédup Supabase
│   └── search/                     # Tests de recherche
├── frontend/                       # Interfaces HTML
│   ├── index.html                  # Chat public
│   └── admin.html                  # Espace admin Daniel
├── data/                           # Cartographies & références métier
├── scripts/                        # Orchestration shell
├── docs/                           # Documentation projet
│   └── PROJET_DME.md
├── logs/                           # Logs runtime (gitignored)
├── textes_extraits/                # Sortie extraction (gitignored)
├── base_vectorielle/               # Ancien Chroma local (gitignored)
├── requirements.txt
├── Procfile                        # Railway → uvicorn app.api:app
└── .env                            # Secrets (ne pas committer)
```

---

## 7. Table Supabase — structure

```sql
CREATE TABLE documents (
    id               BIGINT PRIMARY KEY,
    contenu          TEXT,           -- texte du chunk
    source           TEXT,           -- nom lisible de la source
    embedding        VECTOR(384),    -- vecteur fastembed
    theme            TEXT,           -- tag thématique
    type_enneagramme TEXT,           -- type ennéagramme concerné
    technique_pnl    TEXT,           -- technique PNL associée
    niveau_logique   TEXT            -- niveau logique (Dilts)
);
```

### Fonction de recherche vectorielle
```sql
-- Appelée via Supabase RPC
SELECT * FROM recherche_documents(query_embedding, nb_resultats);
```

---

## 8. Valeurs des tags

### theme
```
confiance_en_soi | estime_de_soi | ego_cellule_de_crise | motivation
rupture_amoureuse | deuil | relation_difficile | communication
emotions | peur | colere | tristesse | joie | stress
sens_de_vie | mission_de_vie | prise_de_decision | management
leadership | parentalite | identite | croyances | valeurs
hypnose | meditation | general | administratif
```

### type_enneagramme
```
type_1 | type_2 | type_3 | type_4 | type_5 | type_6 | type_7 | type_8 | type_9 | tous | aucun
```

### technique_pnl
```
ancrage | recadrage | dissociation | core_process | niveaux_logiques
metaprogrammes | objectif_bien_forme | synchronisation | cnv
multi_dissociation | timeline | sous_modalites | swish
carre_magique | marelle_croyance | aucune
```

### niveau_logique
```
environnement | comportement | competence | croyance | identite | mission | aucun
```

---

## 9. Flux d'une question (endpoint /chat)

```
1. Utilisateur envoie une question
2. Claude Haiku reformule la question en termes PNL/Ennéagramme
3. fastembed transforme la question reformulée en vecteur 384D
4. Supabase RPC recherche_documents() → 8 chunks les plus proches
5. Claude Sonnet reçoit : chunks + question originale + system prompt "Daniel Morel"
6. Claude Sonnet génère la réponse dans la voix et la philosophie de Daniel Morel
7. Réponse retournée à l'utilisateur
```

---

## 10. Ce qui a été fait (historique des sessions)

### Nettoyage de la base (juin 2026)
- Suppression des fichiers bruts ayant une version Transcription : **-49 264 chunks**
- Suppression contenus externes (KDP Nathan Lemire, médical, juridique) : **-2 000 chunks**
- Suppression doublons (_dup, _copy, versions multiples) : **-5 000 chunks**
- Suppression hors-sujet (admin clients, devis, golf, MEP) : **-6 000 chunks**
- Suppression chunks < 50 chars et exactement identiques : **-11 800 chunks**
- Suppression transcriptions bruitées (répétitions, euh euh) : **-236 chunks**
- Suppression ressources hors-sujet (neurofeedback, gène égoïste, KBase) : **-529 chunks**
- **Résultat : de 126 135 → 52 318 chunks (-59%)**

### Normalisation des sources (juin 2026)
- Tous les chemins disque (`Securité_500_g/BACKUP_PAPA_LULU/...`) remplacés par des noms lisibles
- Format : `Support PNL: ...` / `Transcription: ...` / `Formation: ...` / `Ressource: ...`
- 0 source avec chemin technique restant

### Taguage sémantique (juin 2026)
- 9 591 chunks non tagués → tagués via Claude Haiku
- Coût total sessions taguage : ~$3.50
- Résultat : 52 318/52 318 chunks tagués (100%)

---

## 11. Règles absolues (CLAUDE.md)

1. Vérifier si un script Python tourne avant toute action (`pgrep -f python`)
2. Annoncer le coût estimé avant tout script API — attendre "oui" explicite
3. Tout script de masse doit avoir une limite de chunks
4. Arrêt immédiat sur erreur, jamais de relance automatique
5. Ne jamais supprimer de fichiers sur les disques physiques — uniquement Supabase
6. Signaler ce qu'on ne sait pas avec certitude

---

## 12. Prochaines étapes possibles

- [ ] Filtrage par tags avant recherche vectorielle (plus précis, plus rapide)
- [ ] Ajouter de nouveaux thèmes si besoin (ex: `spiritualite`, `corps`)
- [ ] Interface utilisateur améliorée
- [ ] Ajouter de nouveaux documents au fil du temps
- [x] Structure du repo clarifiée (app / pipeline / frontend / data) — juillet 2026
- [ ] Redéployer Railway avec `Procfile` → `uvicorn app.api:app`
- [ ] Vérifier que les variables d'environnement Railway sont à jour (plus de secrets dans le code)

