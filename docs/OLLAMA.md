# Ollama — IA locale gratuite pour DME

**Ollama** = une IA qui tourne **sur ton Mac**, sans payer Anthropic.

## Ce qui est déjà en place

- Ollama installé (Homebrew)
- Service démarré
- Modèle téléchargé : `llama3.2` (~2 Go)
- DME peut l’utiliser automatiquement si Ollama est joignable

## Important (à comprendre)

| Où tourne le chat | Peut utiliser Ollama ? |
|---|---|
| **Sur ton Mac** (API locale) | Oui — gratuit, réponses “IA” |
| **Sur Railway** (lien public) | Non directement — Railway ne voit pas ton Mac |

Donc :
- **Site public Railway** = mode actuel (extraits de ta base, 0 € API)
- **Chat “voix IA” gratuit** = lancer DME en local sur le Mac avec Ollama

## Démarrer le chat local (gratuit + Ollama)

1. Vérifier Ollama :
   ```bash
   brew services start ollama
   ollama list
   ```
2. Dans le projet DME :
   ```bash
   cd "/Users/danielmorel/Cursor_projets/01_DME"
   export LLM_PROVIDER=ollama
   export OLLAMA_MODEL=llama3.2
   uvicorn app.api:app --host 127.0.0.1 --port 8000
   ```
3. Ouvrir dans le navigateur :
   - Public local : http://127.0.0.1:8000/chat-public  
   - Admin local : http://127.0.0.1:8000/admin  

(Ou utiliser le script `scripts/demarrer_local_ollama.sh`.)

## Variables utiles (`.env` ou export)

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2
```

`LLM_PROVIDER=auto` = essaie Claude, puis Ollama, puis extraits.

## Améliorer la qualité plus tard (toujours gratuit)

```bash
ollama pull mistral
# puis OLLAMA_MODEL=mistral
```

`mistral` est souvent meilleur en français, un peu plus lourd.
