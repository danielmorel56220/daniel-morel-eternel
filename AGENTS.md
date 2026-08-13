# 01_DME — Daniel Morel Éternel

**But :** chatbot clients « voix Daniel » (RAG production).
**Racine :** `/Users/danielmorel/Cursor_projets/01_DME`

## Accès (liens chat)
- Fiche : `docs/LIENS_ACCES.md`
- Public : https://daniel-morel-eternel-production.up.railway.app/chat-public
- Propriétaires (mdp) : https://daniel-morel-eternel-production.up.railway.app/admin  
  → mot de passe = `ADMIN_PASSWORD` (Railway / `.env`, jamais dans Git)
- IA locale gratuite : `docs/OLLAMA.md` + `scripts/demarrer_local_ollama.sh`

## Source of truth
- Runtime API : `app/`
- Corpus brut re-indexable : `textes_extraits/`
- Ontologie / cartographies : `data/`
- Prompts produit : `app/prompts/`
- Labs (Baer, Lopvet, règles POC) : `labs/`

## Interdits
- Ne pas utiliser `base_vectorielle/` (Chroma OBSOLETE — voir `base_vectorielle/OBSOLETE.md`)
- Ne pas confondre avec `02_KDP` (ventes livre)
- Pas d’appel API Anthropic de masse sans validation coût + « oui » explicite

## Entités données (plateforme)
`KnowledgeChunk` (+ tags theme / PNL / Dilts / ennéagramme)
