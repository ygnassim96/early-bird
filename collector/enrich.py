"""Stage 2: fetch article text where allowed, then tag each new article with Gemini
(topic, region, urgency, entities and a one-line summary), as Meridian does."""
import csv
import threading
import time
import urllib.robotparser
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

import requests

from . import config, gemini
from .util import domain_of, load_jsonl, save_json, save_jsonl, utcnow

TOPICS = "politics | international | business | technology | health | science | environment | security | society | sports | culture | other"

PROMPT = """You are tagging news articles for a clustering pipeline. Return one JSON object per article, as a JSON array and nothing else.
Write every field in English even if the article is in another language. Do not repeat information across fields.

Fields:
- i: the article number shown in brackets
- topic: one of {topics}
- subtopic: specific area, lowercase with hyphens (e.g. ai-research, elections, trade-policy, ceasefire-talks)
- geography: global | us | china | europe | middle-east | africa | asia | latin-america, or a specific country, city or region in lowercase
- scope: policy | technical | market | social-impact | breaking-news | analysis | opinion
- urgency: breaking | developing | routine | historical
- source_type: mainstream | trade | academic | government | blog | state-media | think-tank
- entities: up to 5 key people, organisations or places
- tags: up to 5 further specifics not covered above
- summary: one neutral factual sentence of at most 30 words
- hard_news: true if it reports a real-world event or decision, false for lifestyle, reviews, puzzles, adverts or evergreen explainers

Articles:
{articles}
"""


def _load_state_media_outlets():
    """Outlet names feeds.csv flags state_media=true, with the "(via Google News)"
    suffix stripped: that's how Google News items end up labelled once collect.py
    swaps in the publisher's own name from the RSS entry's <source> tag."""
    names = set()
    try:
        with open(config.FEEDS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if str(row.get("state_media", "")).strip().lower() == "true":
                    outlet = row["outlet"].replace(" (via Google News)", "").strip()
                    names.add(outlet)
    except (FileNotFoundError, KeyError):
        pass
    return names


STATE_MEDIA_OUTLETS = _load_state_media_outlets()


class DomainGate:
    """At most one request per domain per `delay` seconds."""

    def __init__(self, delay=1.0):
        self.delay = delay
        self._locks = {}
        self._last = {}
        self._g = threading.Lock()

    def wait(self, domain):
        with self._g:
            lock = self._locks.setdefault(domain, threading.Lock())
        with lock:
            d = self.delay - (time.time() - self._last.get(domain, 0))
            if d > 0:
                time.sleep(d)
            self._last[domain] = time.time()


_robots = {}
gate = DomainGate()


def robots_allows(url: str) -> bool:
    p = urlsplit(url)
    base = f"{p.scheme}://{p.netloc}"
    rp = _robots.get(base)
    if rp is None:
        rp = urllib.robotparser.RobotFileParser()
        try:
            r = requests.get(base + "/robots.txt", headers={"User-Agent": config.UA}, timeout=8)
            if r.status_code == 200:
                rp.parse(r.text.splitlines())
            elif r.status_code in (401, 403):
                rp.disallow_all = True
            else:
                rp.parse([])
        except requests.RequestException:
            rp.parse([])
        _robots[base] = rp
    return rp.can_fetch(config.UA_TOKEN, url)


decode_stats = {"tried": 0, "decoded": 0, "failed": 0, "errors": []}
_stats_lock = threading.Lock()


def _note_decode(ok: bool, message: str = "") -> None:
    with _stats_lock:
        decode_stats["tried"] += 1
        decode_stats["decoded" if ok else "failed"] += 1
        if not ok and len(decode_stats["errors"]) < 5:
            decode_stats["errors"].append(message[:200])


def resolve_google(url: str):
    """Turn a news.google.com link into the publisher's own URL.
    googlenewsdecoder 0.2+ returns {"success": True, "decoded_url": ...}; 0.1.x used "status"."""
    try:
        from googlenewsdecoder import gnewsdecoder
        r = gnewsdecoder(url, interval=1)
        if (r.get("success") or r.get("status")) and r.get("decoded_url"):
            _note_decode(True)
            return r["decoded_url"]
        _note_decode(False, str(r.get("message") or r.get("error") or r))
    except Exception as e:
        _note_decode(False, f"{type(e).__name__}: {e}")
    return None


def fetch_text(url: str) -> str:
    """Return the first part of the article text, or '' if we may not or cannot fetch it."""
    import trafilatura

    dom = domain_of(url)
    if any(dom == d or dom.endswith("." + d) for d in config.PAYWALLED_DOMAINS):
        return ""
    if not robots_allows(url):
        return ""
    gate.wait(dom)
    try:
        r = requests.get(url, headers={"User-Agent": config.UA}, timeout=15)
        if r.status_code != 200:
            return ""
        text = trafilatura.extract(r.text, include_comments=False, include_tables=False) or ""
        return text[: config.TEXT_CHARS]
    except Exception:
        return ""


def prepare(a, allow_decode: bool):
    """Add article text (and the real URL for Google News items)."""
    if "text" in a:
        return a
    url = a["url"]
    if "news.google.com" in url:
        real = resolve_google(url) if allow_decode else None
        if real:
            a["real_url"] = real
            url = real
    a["text"] = fetch_text(url) if not url.startswith("https://news.google.com") else ""
    return a


def tag_batch(batch):
    lines = []
    for n, a in enumerate(batch, 1):
        body = (a.get("text") or a.get("summary") or "")[: config.TEXT_CHARS]
        lines.append(f'[{n}] Title: {a["title"]} | Outlet: {a["outlet"]} | Text: {body}')
    result = gemini.generate_json(PROMPT.format(topics=TOPICS, articles="\n".join(lines)))
    if isinstance(result, dict):
        result = result.get("articles") or result.get("results") or []
    tagged = 0
    for obj in result:
        try:
            a = batch[int(obj["i"]) - 1]
        except (KeyError, ValueError, IndexError, TypeError):
            continue
        a["topic"] = str(obj.get("topic", "other")).lower()
        a["subtopic"] = str(obj.get("subtopic", ""))[:60]
        a["geography"] = str(obj.get("geography", ""))[:40]
        a["scope"] = str(obj.get("scope", ""))[:30]
        a["urgency"] = str(obj.get("urgency", "routine")).lower()
        # Trust the feed list over Gemini's guess for outlets we already know
        # are state-affiliated (TASS, Tehran Times, Xinhua, Global Times,
        # Ahram Online): Gemini called these "mainstream" often enough that
        # the brief's source-preference rule couldn't rely on the tag.
        if a["outlet"] in STATE_MEDIA_OUTLETS:
            a["source_type"] = "state-media"
        else:
            a["source_type"] = str(obj.get("source_type", ""))[:30]
        a["entities"] = [str(x)[:60] for x in (obj.get("entities") or [])][:5]
        a["tags"] = [str(x)[:40] for x in (obj.get("tags") or [])][:5]
        a["ai_summary"] = str(obj.get("summary", ""))[:300]
        a["hard_news"] = bool(obj.get("hard_news", True))
        a["tagged"] = True
        tagged += 1
    return tagged


def run():
    path = config.DATA / "articles.jsonl"
    articles = load_jsonl(path)
    todo = [a for a in articles if not a.get("tagged") and a.get("tag_attempts", 0) < 3]
    todo = todo[: config.MAX_ENRICH_PER_RUN]
    print(f"{len(todo)} articles to enrich")

    decode_budget = [config.MAX_GNEWS_DECODE_PER_RUN]
    lock = threading.Lock()

    def job(a):
        allow = False
        with lock:
            if decode_budget[0] > 0 and "news.google.com" in a["url"]:
                decode_budget[0] -= 1
                allow = True
        return prepare(a, allow)

    if todo:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(job, todo))

    # Backfill: articles that were already tagged before the decoder fix
    # (2026-11-07) and are still stuck with a dead news.google.com link. They
    # don't need re-tagging, only a real URL and, where the publisher allows
    # it, text. Spends whatever of this run's decode budget the pass above
    # (which only covers newly-tagged articles) didn't use.
    if decode_budget[0] > 0:
        backlog = [
            a for a in articles
            if a.get("tagged") and "news.google.com" in a["url"] and not a.get("real_url")
        ][: decode_budget[0]]
        if backlog:
            def backfill_job(a):
                real = resolve_google(a["url"])
                if real:
                    a["real_url"] = real
                    a["text"] = fetch_text(real)
                return a

            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(backfill_job, backlog))
            print(f"backfilled {len(backlog)} already-tagged google news links")

    if decode_stats["tried"]:
        print(f"google news decode: {decode_stats['decoded']}/{decode_stats['tried']} decoded")
        for m in decode_stats["errors"]:
            print(f"  decode error: {m}")
    save_json(config.DATA / "decode_stats.json", {"at": utcnow().isoformat(timespec="seconds"), **decode_stats})
    save_jsonl(path, articles)  # persist any backfill even if there's nothing left to tag

    if not todo:
        return

    total = 0
    busy_streak = 0
    for i in range(0, len(todo), config.TAG_BATCH):
        batch = todo[i:i + config.TAG_BATCH]
        k = i // config.TAG_BATCH + 1
        t0 = time.time()
        try:
            n = tag_batch(batch)
            total += n
            busy_streak = 0
            print(f"batch {k}: tagged {n}/{len(batch)} with {gemini.last_model} in {time.time() - t0:.0f}s")
        except gemini.GeminiBusy as e:
            # Gemini overloaded: not the articles' fault, so don't count it against them.
            busy_streak += 1
            print(f"batch {k}: every model busy, left for the next run ({e})")
            if busy_streak >= 3:
                print("stopping early: Gemini is overloaded right now")
                break
            continue
        except gemini.GeminiError as e:
            print(f"batch {k} failed: {e}")
        for a in batch:
            if not a.get("tagged"):
                a["tag_attempts"] = a.get("tag_attempts", 0) + 1
        save_jsonl(path, articles)  # save as we go so a timeout loses little
    print(f"tagged {total}/{len(todo)}")


if __name__ == "__main__":
    run()
