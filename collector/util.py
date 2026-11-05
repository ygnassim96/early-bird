import datetime as dt
import hashlib
import html
import json
import os
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

TRACKING = re.compile(r"^(utm_|fbclid|gclid|mc_|ref$|ocid|cmpid|smid|at_|cid$)", re.I)


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s):
    if not s:
        return None
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)


def canonical_url(u: str) -> str:
    s = urlsplit(u.strip())
    q = [(k, v) for k, v in parse_qsl(s.query, keep_blank_values=True) if not TRACKING.match(k)]
    path = s.path.rstrip("/") or "/"
    return urlunsplit(((s.scheme or "https").lower(), s.netloc.lower(), path, urlencode(q), ""))


def article_id(url: str) -> str:
    return hashlib.sha1(canonical_url(url).encode()).hexdigest()[:16]


def domain_of(url: str) -> str:
    host = urlsplit(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def strip_html(text: str, limit: int = 600) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(path, obj):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def load_jsonl(path):
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except FileNotFoundError:
        pass
    return out


def save_jsonl(path, rows):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
