"""Minimal Gemini REST client (no SDK). The key is read from GEMINI_API_KEY and sent in a header,
so it never appears in URLs or error messages."""
import json
import os
import time

import requests

BASE = "https://generativelanguage.googleapis.com/v1beta"


def _settings():
    return {
        "key": os.environ.get("GEMINI_API_KEY", ""),
        # `or` (not a default argument) so an empty value from an unset GitHub variable falls back too
        "model": os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash",
        "embed_model": os.environ.get("GEMINI_EMBED_MODEL") or "gemini-embedding-001",
        "min_interval": 60.0 / float(os.environ.get("GEMINI_RPM") or "12"),
    }


_last_call = 0.0


class GeminiError(RuntimeError):
    pass


def _post(path: str, payload: dict, retries: int = 6) -> dict:
    global _last_call
    s = _settings()
    if not s["key"]:
        raise GeminiError("GEMINI_API_KEY is not set")
    last_err = ""
    for attempt in range(retries):
        wait = s["min_interval"] - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        try:
            r = requests.post(
                f"{BASE}/{path}",
                json=payload,
                headers={"x-goog-api-key": s["key"], "Content-Type": "application/json"},
                timeout=120,
            )
        except requests.RequestException as e:
            last_err = type(e).__name__
            time.sleep(min(60, 5 * 2 ** attempt))
            continue
        if r.status_code == 200:
            return r.json()
        last_err = f"HTTP {r.status_code}: {r.text[:200]}"
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(90, 5 * 2 ** attempt))
            continue
        raise GeminiError(last_err)
    raise GeminiError(f"retries exhausted ({last_err})")


def generate_json(prompt: str):
    s = _settings()
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    d = _post(f"models/{s['model']}:generateContent", payload)
    try:
        text = d["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise GeminiError("empty response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Occasionally the model wraps JSON in a code fence.
        t = text.strip().strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
        return json.loads(t)


def embed(texts, task: str = "CLUSTERING"):
    s = _settings()
    out = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i + 100]
        payload = {
            "requests": [
                {
                    "model": f"models/{s['embed_model']}",
                    "content": {"parts": [{"text": t[:2000]}]},
                    "taskType": task,
                }
                for t in batch
            ]
        }
        d = _post(f"models/{s['embed_model']}:batchEmbedContents", payload)
        out.extend(e["values"] for e in d["embeddings"])
    return out
