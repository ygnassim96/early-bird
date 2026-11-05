import datetime as dt
import json

import numpy as np
import pytest

from collector import collect, config, digest, enrich
from collector.util import canonical_url, article_id, iso, load_jsonl, save_jsonl

NOW = dt.datetime(2026, 11, 5, 12, 0, tzinfo=dt.timezone.utc)

RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
<item><title>Oil falls 9% as Iran talks hopes rise</title><link>https://example.com/a?utm_source=rss</link>
<pubDate>Thu, 05 Nov 2026 10:00:00 GMT</pubDate><description>&lt;p&gt;Brent drops.&lt;/p&gt;</description></item>
<item><title>Old story</title><link>https://example.com/old</link><pubDate>Mon, 01 Jan 2024 10:00:00 GMT</pubDate></item>
<item><title>No date story</title><link>https://example.com/nodate/</link></item>
</channel></rss>"""

GNEWS = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>g</title>
<item><title>Talks resume in Abu Dhabi - Reuters</title><link>https://news.google.com/rss/articles/CBMi123?oc=5</link>
<pubDate>Thu, 05 Nov 2026 09:00:00 GMT</pubDate><source url="https://www.reuters.com">Reuters</source></item>
</channel></rss>"""

FEED = {"id": "f1", "outlet": "Example", "category": "2 Global general", "url": "https://example.com/rss", "tier": 2, "notes": ""}
GFEED = {"id": "g1", "outlet": "Reuters (via Google News)", "category": "1 Wires & aggregators",
         "url": "https://news.google.com/rss/search?q=x", "tier": 1, "notes": ""}


def test_canonical_url_strips_tracking_and_slash():
    assert canonical_url("https://Example.com/a/?utm_source=rss&id=3#frag") == "https://example.com/a?id=3"
    assert article_id("https://example.com/a?utm_source=x") == article_id("https://example.com/a")


def test_parse_feed_filters_old_and_cleans():
    arts, err = collect.parse_feed_bytes(RSS, FEED, NOW)
    assert err is None
    titles = [a["title"] for a in arts]
    assert "Old story" not in titles
    assert len(arts) == 2
    first = arts[0]
    assert first["summary"] == "Brent drops."
    assert first["published"] == "2026-11-05T10:00:00Z"
    assert [a for a in arts if a["title"] == "No date story"][0]["published"] is None


def test_google_news_title_and_outlet():
    arts, _ = collect.parse_feed_bytes(GNEWS, GFEED, NOW)
    assert arts[0]["outlet"] == "Reuters"
    assert arts[0]["title"] == "Talks resume in Abu Dhabi"


def test_empty_and_garbage_feeds_report_error():
    assert collect.parse_feed_bytes(b"<html>nope</html>", FEED, NOW)[1] in ("unparseable", "no items")


def test_is_due_and_backoff():
    assert collect.is_due(FEED, None, NOW)
    st = {"last_checked": iso(NOW - dt.timedelta(minutes=30)), "consecutive_failures": 0}
    assert not collect.is_due(FEED, st, NOW)          # tier 2 = 4h
    st["last_checked"] = iso(NOW - dt.timedelta(hours=5))
    assert collect.is_due(FEED, st, NOW)
    st["consecutive_failures"] = 5                     # backed off to >4h
    st["last_checked"] = iso(NOW - dt.timedelta(hours=5))
    assert not collect.is_due(FEED, st, NOW)


def test_run_dedupes_prunes_and_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA", tmp_path)
    feeds_csv = tmp_path / "feeds.csv"
    feeds_csv.write_text("category,outlet,feed,url,poll_tier,volume,confidence,notes\n"
                         "2 Global general,Example,Main,https://example.com/rss,2,med,likely,\n")
    monkeypatch.setattr(config, "FEEDS_CSV", feeds_csv)
    fetcher = lambda f: (f, RSS, None)
    assert collect.run(now=NOW, fetcher=fetcher) == 2
    # second run later the same day: feed not due yet, nothing new
    assert collect.run(now=NOW + dt.timedelta(minutes=10), fetcher=fetcher) == 0
    # forced run: same articles, no duplicates
    assert collect.run(force=True, now=NOW + dt.timedelta(hours=5), fetcher=fetcher) == 0
    assert len(load_jsonl(tmp_path / "articles.jsonl")) == 2
    # far future: everything pruned
    collect.run(force=True, now=NOW + dt.timedelta(days=5), fetcher=lambda f: (f, GNEWS.replace(b"2026", b"2031"), None))
    state = json.loads((tmp_path / "feed_state.json").read_text())
    assert state[next(iter(state))]["new_per_day"]


def test_tag_batch_maps_by_number(monkeypatch):
    batch = [{"title": "A", "outlet": "X", "summary": "s"}, {"title": "B", "outlet": "Y", "summary": "s"}]
    reply = [{"i": 2, "topic": "Business", "urgency": "breaking", "entities": ["Brent"], "tags": [], "summary": "B happened", "hard_news": True},
             {"i": 9, "topic": "politics"}]  # bad index ignored
    monkeypatch.setattr(enrich.gemini, "generate_json", lambda p: reply)
    assert enrich.tag_batch(batch) == 1
    assert batch[1]["topic"] == "business" and batch[1]["tagged"] and not batch[0].get("tagged")


def _fake_articles(now):
    arts = []
    def add(i, outlet, title, topic="international", urg="routine", hard=True):
        arts.append({"id": str(i), "url": f"https://x/{i}", "title": title, "outlet": outlet, "category": "2 Global general",
                     "published": iso(now - dt.timedelta(hours=i % 5)), "first_seen": iso(now), "tagged": True,
                     "topic": topic, "urgency": urg, "hard_news": hard, "ai_summary": title, "entities": [], "note": ""})
    for i, o in enumerate(["BBC", "NPR", "DW", "France 24", "Guardian"]):
        add(i, o, f"Oil plunges after midterms {i}", urg="breaking")
    for i, o in enumerate(["BBC", "NPR", "DW"], start=10):
        add(i, o, f"Israel coalition talks {i}")
    add(20, "Lone Outlet", "Local bridge reopens")
    add(21, "ESPN", "Team wins title", topic="sports")
    add(22, "Blog", "Ten best puzzles", hard=False)
    return arts


def test_digest_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA", tmp_path)
    feeds_csv = tmp_path / "feeds.csv"
    feeds_csv.write_text("category,outlet,feed,url,poll_tier,volume,confidence,notes\n"
                         "2 Global general,BBC,Main,https://bbc/rss,1,high,likely,\n")
    monkeypatch.setattr(config, "FEEDS_CSV", feeds_csv)
    save_jsonl(tmp_path / "articles.jsonl", _fake_articles(NOW))

    rng = np.random.default_rng(0)
    centres = {"Oil": rng.normal(size=16), "Israel": rng.normal(size=16), "Local": rng.normal(size=16)}

    def embedder(texts):
        out = []
        for t in texts:
            key = "Oil" if "Oil" in t else "Israel" if "Israel" in t else "Local"
            out.append(centres[key] + rng.normal(scale=0.01, size=16))
        return out

    summariser = lambda members: {"headline": f"H{len(members)}", "summary": "S", "region": "Global", "topic": "business",
                                  "coherent": True, "outlet_disagreement": ""}
    d = digest.run(now=NOW, embedder=embedder, summariser=summariser)
    assert d["stats"]["articles_considered"] == 9          # sports + non-hard-news excluded
    sizes = [c["n_articles"] for c in d["clusters"]]
    assert sizes[0] == 5                                    # oil story ranks first (most outlets, breaking)
    assert 3 in sizes
    assert d["clusters"][0]["headline"] == "H5"
    assert any(r["title"] == "Local bridge reopens" for r in d["radar"])
    for name in ("digest.json", "digest_top.md", "feed_health.json"):
        assert (tmp_path / name).exists()
    health = json.loads((tmp_path / "feed_health.json").read_text())
    assert health[0]["status"] == "never checked"


def test_cluster_vectors_small_inputs():
    assert list(digest.cluster_vectors(np.zeros((1, 4)))) == [-1]


def test_feeds_csv_is_wellformed():
    feeds = collect.load_feeds(config.ROOT / "feeds.csv")
    assert len(feeds) > 100
    assert all(f["url"].startswith("http") for f in feeds)
    assert len({f["id"] for f in feeds}) == len(feeds)
    assert {f["tier"] for f in feeds} <= {1, 2, 3, 4}
