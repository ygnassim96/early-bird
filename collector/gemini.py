"""Minimal Gemini REST client (no SDK). The key is read from GEMINI_API_KEY and sent in a header,
so it never appears in URLs or error messages.

Text generation uses a chain of models. If the main model is busy (503), rate-limited or out of
quota (429), the same request goes straight to the next model instead of waiting. A model that
no longer exists (404) or whose daily quota is used up is skipped for the rest of the run."""
import json
import os
import time

import requests

BASE = "https://generativelanguage.googleapis.com/v1beta"

# Tried in this order after GEMINI_MODEL. Duplicates of the main model are skipped.
DEFAULT_FALLBACKS = "gemini-3.1-flash-lite,gemini-3.5-flash-lite"
BUSY = {429, 500, 502, 503, 504}


def _settings():
    return {
        "key": os.environ.get("GEMINI_API_KEY", ""),
        # `or` (not a default argument) so an empty value from an unset GitHub variable falls back too
        "model": os.environ.get("GEMINI_MODEL") or "gemini-3.1-flash-lite",
        "fallbacks": os.environ.get("GEMINI_FALLBACK_MODELS") or DEFAULT_FALLBACKS,
        "embed_model": os.environ.get("GEMINI_EMBED_MODEL") or "gemini-embedding-001",
        "min_interval": 60.0 / float(os.environ.get("GEMINI_RPM") or "12"),
    }


_last_call = 0.0
_skip = set()      # models found unusable during this run (retired, or daily quota used up)
last_model = ""    # the model that answered the most recent generate_json call


class GeminiError(RuntimeError):
    pass


class GeminiBusy(GeminiError):
    """Every model was busy or out of quota. Worth retrying later; not the article's fault."""


def _models():
    s = _settings()
    out = []
    for m in [s["model"]] + s["fallbacks"].split(","):
        m = m.strip()
        if m and m not in out and m not in _skip:
            out.append(m)
    return out


def _request(path: str, payload: dict):
    """One HTTP call. Returns (json_or_None, status_code, error_text). Status 0 = network error."""
    global _last_call
    s = _settings()
    if not s["key"]:
        raise GeminiError("GEMINI_API_KEY is not set")
    wait = s["min_interval"] - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()
    try:
        r = requests.post(
            f"{BASE}/{path}",
            json=payload,
            headers={"x-goog-api-key": s["key"], "Content-Type": "application/json"},
            timeout=90,
        )
    except requests.RequestException as e:
        return None, 0, type(e).__name__
    if r.status_code == 200:
        return r.json(), 200, ""
    return None, r.status_code, f"HTTP {r.status_code}: {r.text[:200]}"


def _post(path: str, payload: dict, retries: int = 6) -> dict:
    """Single model with retries (used for embeddings)."""
    last_err = ""
    for attempt in range(retries):
        data, status, err = _request(path, payload)
        if data is not None:
            return data
        last_err = err
        if status == 0 or status in BUSY:
            time.sleep(min(60, 5 * 2 ** attempt))
            continue
        raise GeminiError(last_err)
    raise GeminiError(f"retries exhausted ({last_err})")


def _generate(payload: dict, rounds: int = 2) -> dict:
    """Try each model in turn; go round the list `rounds` times with a short pause in between."""
    global last_model
    errors = []
    for rnd in range(rounds):
        models = _models()
        if not models:
            break
        for m in models:
            data, status, err = _request(f"models/{m}:generateContent", payload)
            if data is not None:
                last_model = m
                return data
            errors.append(f"{m}: {err[:120]}")
            if status in (401, 403):
                raise GeminiError(err)                     # key problem: no model will work
            if status == 404 or (status == 429 and "quota" in err.lower()):
                _skip.add(m)                               # retired, or out of quota for today
                print(f"  skipping {m} for the rest of this run ({err[:80]})")
            # 400 (model-specific rejection), busy or network: move on to the next model
        if rnd < rounds - 1:
            time.sleep(15)
    raise GeminiBusy("all models failed: " + " | ".join(errors[-3:]))


def generate_json(prompt: str):
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    d = _generate(payload)
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
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            raise GeminiError("response was not valid JSON")


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
