"""Stage 1: fetch the RSS feeds that are due and add new articles to data/articles.jsonl."""
import argparse
import csv
import datetime as dt
import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor

import feedparser
import requests

from . import config
from .util import (article_id, canonical_url, iso, load_json, load_jsonl, parse_iso,
                   save_json, save_jsonl, strip_html, utcnow)


def load_feeds(path=None):
    feeds = []
    with open(path or config.FEEDS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("url"):
                continue
            row["id"] = hashlib.sha1(row["url"].encode()).hexdigest()[:10]
            row["tier"] = int(row.get("poll_tier") or 2)
            feeds.append(row)
    return feeds


def is_due(feed, st, now) -> bool:
    if not st or not st.get("last_checked"):
        return True
    base = config.TIER_MINUTES.get(feed["tier"], 240)
    fails = st.get("consecutive_failures", 0)
    # Back off on feeds that keep failing, up to once a day.
    minutes = base if fails < 3 else min(1440, base * 2 ** min(fails - 2, 4))
    last = parse_iso(st["last_checked"])
    return (now - last).total_seconds() >= (minutes - 5) * 60


def entry_time(e):
    for k in ("published_parsed", "updated_parsed"):
        t = e.get(k)
        if t:
            return dt.datetime(*t[:6], tzinfo=dt.timezone.utc)
    return None


def parse_feed_bytes(content: bytes, feed: dict, now: dt.datetime):
    """Turn raw feed XML into article dicts. Returns (articles, error)."""
    parsed = feedparser.parse(content)
    if not parsed.entries:
        return [], ("unparseable" if parsed.bozo else "no items")
    is_gnews = "news.google.com" in feed["url"]
    cutoff = now - dt.timedelta(hours=config.RETAIN_HOURS)
    out = []
    for e in parsed.entries:
        link = (e.get("link") or "").strip()
        title = strip_html(e.get("title") or "", 300)
        if not link or not title:
            continue
        t = entry_time(e)
        if t and t < cutoff:
            continue
        if t and t > now + dt.timedelta(hours=6):
            t = now
        outlet = feed["outlet"]
        if is_gnews:
            src = (e.get("source") or {}).get("title")
            if src:
                outlet = src
                if title.endswith(" - " + src):
                    title = title[: -(len(src) + 3)]
        out.append({
            "id": article_id(link),
            "url": link,
            "title": title,
            "outlet": outlet,
            "feed_id": feed["id"],
            "category": feed["category"],
            "note": feed.get("notes", ""),
            "published": iso(t) if t else None,
            "first_seen": iso(now),
            "summary": strip_html(e.get("summary") or e.get("description") or "", 500),
        })
    return out, None


def fetch(feed):
    try:
        r = requests.get(feed["url"], headers={"User-Agent": config.UA, "Accept": "application/rss+xml, application/xml, text/xml, */*"},
                         timeout=25)
    except requests.RequestException as e:
        return feed, None, type(e).__name__
    if r.status_code != 200:
        return feed, None, f"HTTP {r.status_code}"
    return feed, r.content, None


def run(force=False, now=None, fetcher=fetch):
    now = now or utcnow()
    config.DATA.mkdir(parents=True, exist_ok=True)
    feeds = load_feeds()
    state = load_json(config.DATA / "feed_state.json", {})
    articles = {a["id"]: a for a in load_jsonl(config.DATA / "articles.jsonl")}

    due = [f for f in feeds if force or is_due(f, state.get(f["id"]), now)]
    print(f"{len(feeds)} feeds, {len(due)} due")
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(fetcher, due))

    today = now.strftime("%Y-%m-%d")
    new_total = ok = failed = 0
    for feed, content, err in results:
        st = state.setdefault(feed["id"], {"consecutive_failures": 0, "new_per_day": {}})
        st["last_checked"] = iso(now)
        items = []
        if content is not None:
            items, err = parse_feed_bytes(content, feed, now)
        if err:
            st["consecutive_failures"] = st.get("consecutive_failures", 0) + 1
            st["last_error"] = err
            failed += 1
            continue
        ok += 1
        st["consecutive_failures"] = 0
        st["last_error"] = None
        st["last_ok"] = iso(now)
        st["items_in_feed"] = len(items)
        new = 0
        for a in items:
            if a["id"] not in articles:
                articles[a["id"]] = a
                new += 1
        st["new_per_day"][today] = st["new_per_day"].get(today, 0) + new
        # keep two weeks of daily counts
        for d in sorted(st["new_per_day"])[:-14]:
            del st["new_per_day"][d]
        new_total += new

    # Drop anything older than the retention window.
    cutoff = now - dt.timedelta(hours=config.RETAIN_HOURS)
    kept = []
    for a in articles.values():
        t = parse_iso(a.get("published")) or parse_iso(a.get("first_seen"))
        if t is None or t >= cutoff:
            kept.append(a)
    kept.sort(key=lambda a: a.get("published") or a["first_seen"], reverse=True)

    save_jsonl(config.DATA / "articles.jsonl", kept)
    save_json(config.DATA / "feed_state.json", state)
    print(f"fetched ok={ok} failed={failed}; new articles={new_total}; stored={len(kept)}")
    return new_total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="check every feed regardless of schedule")
    args = ap.parse_args()
    run(force=args.force)
    sys.exit(0)
