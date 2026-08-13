#!/usr/bin/env bash
# Démarre DME en local avec Ollama (IA gratuite sur le Mac).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "→ Démarrage Ollama…"
brew services start ollama >/dev/null 2>&1 || true
sleep 1

if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null; then
  echo "Ollama ne répond pas. Lance : brew services start ollama"
  exit 1
fi

export LLM_PROVIDER="${LLM_PROVIDER:-ollama}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"

echo "→ Provider: $LLM_PROVIDER | modèle: $OLLAMA_MODEL"
echo "→ Chat public: http://127.0.0.1:8000/chat-public"
echo "→ Admin:       http://127.0.0.1:8000/admin"
exec uvicorn app.api:app --host 127.0.0.1 --port 8000
