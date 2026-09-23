"""
Fournisseurs LLM gratuits / bas coût pour DME.
Ordre typique : Groq (si clé) → Gemini (si clé) → OVHcloud (anonyme EU) → Pollinations → Ollama local.
"""

from __future__ import annotations

import os
from typing import Optional

import requests as http

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = (
    os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
).strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
OVH_BASE_URL = os.getenv(
    "OVH_BASE_URL",
    "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
).rstrip("/")
OVH_MODEL = os.getenv("OVH_MODEL", "Mistral-Nemo-Instruct-2407").strip()
POLLINATIONS_URL = os.getenv(
    "POLLINATIONS_URL",
    "https://text.pollinations.ai/openai",
).strip()


def _openai_chat(
    base_url: str,
    model: str,
    system_prompt: str,
    user_content: str,
    api_key: Optional[str] = None,
    max_tokens: int = 900,
    timeout: int = 90,
    temperature: float = 0.4,
) -> str:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = http.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers=headers,
        json={
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        },
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"{base_url} HTTP {resp.status_code}: {resp.text[:240]}")
    data = resp.json()
    choix = (data.get("choices") or [{}])[0]
    texte = (choix.get("message") or {}).get("content") or choix.get("text") or ""
    if not str(texte).strip():
        raise RuntimeError(f"{base_url}: réponse vide")
    return str(texte).strip()


def essayer_groq(system_prompt: str, user_content: str) -> Optional[str]:
    if not GROQ_API_KEY:
        return None
    try:
        return _openai_chat(
            "https://api.groq.com/openai/v1",
            GROQ_MODEL,
            system_prompt,
            user_content,
            api_key=GROQ_API_KEY,
            max_tokens=650,
            timeout=75,
            temperature=0.25,
        )
    except Exception:
        return None


def essayer_gemini(system_prompt: str, user_content: str) -> Optional[str]:
    if not GEMINI_API_KEY:
        return None
    try:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        )
        resp = http.post(
            url,
            json={
                "system_instruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_content}]}],
                "generationConfig": {"maxOutputTokens": 900, "temperature": 0.4},
            },
            timeout=90,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        parts = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [])
        )
        texte = "".join(p.get("text", "") for p in parts).strip()
        return texte or None
    except Exception:
        return None


def essayer_ovh(system_prompt: str, user_content: str) -> Optional[str]:
    """Tier anonyme OVHcloud AI Endpoints (UE) — pas de clé."""
    try:
        return _openai_chat(
            OVH_BASE_URL,
            OVH_MODEL,
            system_prompt,
            user_content,
            api_key=None,
            timeout=75,
        )
    except Exception:
        # Essai modèle plus petit si le premier est saturé
        try:
            return _openai_chat(
                OVH_BASE_URL,
                "Mistral-7B-Instruct-v0.3",
                system_prompt,
                user_content,
                api_key=None,
                timeout=75,
            )
        except Exception:
            return None


def essayer_pollinations(system_prompt: str, user_content: str) -> Optional[str]:
    """Endpoint public Pollinations (parfois instable / rate-limité)."""
    try:
        resp = http.post(
            POLLINATIONS_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": "openai",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "max_tokens": 800,
            },
            timeout=60,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        if isinstance(data, str) and data.strip():
            return data.strip()
        choix = (data.get("choices") or [{}])[0]
        texte = (choix.get("message") or {}).get("content") or ""
        return texte.strip() or None
    except Exception:
        return None


def generer_gratuit(system_prompt: str, user_content: str) -> tuple[Optional[str], str]:
    """
    Tente les fournisseurs gratuits dans l'ordre.
    Retourne (texte|None, nom_du_fournisseur_utilise).
    """
    for nom, fn in (
        ("groq", essayer_groq),
        ("gemini", essayer_gemini),
        ("ovh", essayer_ovh),
        ("pollinations", essayer_pollinations),
    ):
        texte = fn(system_prompt, user_content)
        if texte:
            return texte, nom
    return None, "aucun"
