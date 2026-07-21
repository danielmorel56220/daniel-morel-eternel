#!/bin/bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$DIR/logs/orchestrateur.log"
mkdir -p "$DIR/logs"

echo "[$(date '+%H:%M:%S')] Orchestrateur démarré" >> "$LOG"

# Étape 1 — Attendre la fin de extraction.py (Cle 2T)
echo "[$(date '+%H:%M:%S')] Attente de la fin de l'extraction Cle 2T..." >> "$LOG"

while pgrep -f "extraction.py" > /dev/null; do
    sleep 30
done

echo "[$(date '+%H:%M:%S')] Extraction Cle 2T terminée." >> "$LOG"

# Bilan Cle 2T
OK=$(wc -l < "$DIR/logs/extraction_ok.log" 2>/dev/null || echo 0)
ERR=$(wc -l < "$DIR/logs/extraction_erreurs.log" 2>/dev/null || echo 0)
echo "[$(date '+%H:%M:%S')] Cle 2T — $OK extraits, $ERR erreurs/vides" >> "$LOG"

# Étape 2 — Lancer l'extraction multi-disques
echo "[$(date '+%H:%M:%S')] Démarrage extraction Sécurité 500g + Clean Disk + HYSTA..." >> "$LOG"
python3 "$DIR/pipeline/extract/extraction_multi_disques.py" >> "$LOG" 2>&1

echo "[$(date '+%H:%M:%S')] Extraction multi-disques terminée." >> "$LOG"

# Bilan final
TOTAL=$(find "$DIR/textes_extraits" -type f -name "*.txt" | wc -l)
echo "[$(date '+%H:%M:%S')] BILAN FINAL — $TOTAL documents uniques prêts dans textes_extraits/" >> "$LOG"
echo "[$(date '+%H:%M:%S')] Daniel Morel Éternel — base de textes constituée." >> "$LOG"
