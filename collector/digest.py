"""Stage 3: group the last day's tagged articles into stories, rank them, summarise the top ones,
and write digest.json, digest_top.md and feed_health.json."""
import datetime as dt
import math
import re

import numpy as np

from . import brief_files, config, gemini
from .collect import load_feeds
from .util import iso, load_json, load_jsonl, parse_iso, save_json, utcnow

# Read real article text, not just the one-sentence tag summary, so the cluster
# synthesis actually draws on what the decoder/browser-fallback fixes (2026-11-07)
# now give us: full text for a majority of articles instead of a 30-word stub.
SUMMARY_PROMPT = """You are an intelligence analyst. The items below come from different outlets and appear to cover the same story.
Each item includes as much of the article's actual text as was available -- read it, don't just skim the headlines.
Return one JSON object with these fields and nothing else:
- headline: neutral factual headline, at most 14 words
- summary: 3 to 5 sentences with the concrete facts (who, what, numbers, when, where). Draw on the article text, not just the headlines. Use only what is in the items below -- never invent a fact, figure or quote.
- key_facts: 3 to 6 short strings, each a single specific fact, figure, or named detail pulled from the article text that a summary this short would otherwise drop (an exact number, a direct quote, a date, a named official's statement). Skip this field (empty array) if the items don't have enough real text to support it.
- region: e.g. "United States", "Middle East", "Europe", "Asia-Pacific", "Africa", "Latin America", "Global"
- topic: one of politics, international, business, technology, health, science, environment, security, society
- coherent: true if all items are about the same event, false if they are mixed
- outlet_disagreement: one sentence if outlets, especially state-run or partisan ones, frame or report it differently, otherwise an empty string

Items:
{items}
"""


def embed_text(a) -> str:
    parts = [a["title"], a.get("ai_summary") or a.get("summary") or ""]
    if a.get("entities"):
        parts.append(", ".join(a["entities"]))
    return ". ".join(p for p in parts if p)


def cluster_vectors(vectors: np.ndarray) -> np.ndarray:
    """Group similar articles. Returns a label per article; -1 means 'not part of a group'."""
    from sklearn.cluster import HDBSCAN

    v = np.asarray(vectors, dtype=np.float32)
    v = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)  # cosine ~ euclidean on unit vectors
    if len(v) < 3:
        return -np.ones(len(v), dtype=int)
    model = HDBSCAN(min_cluster_size=2, min_samples=1, cluster_selection_method="leaf",
                    cluster_selection_epsilon=0.0, max_cluster_size=40, copy=True)
    return model.fit_predict(v)


def score_cluster(members, now) -> float:
    n = len(members)
    outlets = len({m["outlet"] for m in members})
    cats = len({m["category"] for m in members})
    urg = sum(0.5 for m in members if m.get("urgency") == "breaking") + \
        sum(0.25 for m in members if m.get("urgency") == "developing")
    last = max((parse_iso(m.get("published")) or parse_iso(m["first_seen"]) for m in members))
    age_h = max(0.0, (now - last).total_seconds() / 3600)
    recency = max(0.0, 1 - age_h / config.DIGEST_HOURS)
    return round(2.0 * math.log1p(outlets) + math.log1p(n) + 0.4 * math.log1p(cats) + min(urg, 2.0) + recency, 3)


def link_of(a) -> str:
    return a.get("real_url") or a["url"]


def member_view(a):
    return {
        "title": a["title"], "url": link_of(a), "outlet": a["outlet"],
        "published": a.get("published"), "note": a.get("note", ""),
        "summary": a.get("ai_summary") or a.get("summary", ""),
    }


def _item_text(m, chars=700):
    """The best material we have for this article: real extracted text if we
    have it (now common, post decoder/browser-fallback fixes), else fall back
    to the short tag-time summary for articles that are genuinely paywalled
    or otherwise textless."""
    text = (m.get("text") or "").strip()
    if text:
        return text[:chars]
    return m.get("ai_summary") or m.get("summary", "")


def summarise(members):
    # Fewer members than before (8 instead of 12), but each with real article
    # text rather than one tag-time sentence -- more actual reporting reaches
    # the model, not more items.
    items = "\n\n".join(
        f"[{m['outlet']}{' (' + m['note'] + ')' if m.get('note') else ''}] {m['title']}\n{_item_text(m)}"
        for m in members[:8]
    )
    return gemini.generate_json(SUMMARY_PROMPT.format(items=items))


def build_clusters(articles, labels, now):
    groups = {}
    singles = []
    for a, lab in zip(articles, labels):
        if lab == -1:
            singles.append(a)
        else:
            groups.setdefault(int(lab), []).append(a)
    clusters = []
    for lab, members in groups.items():
        members.sort(key=lambda m: m.get("published") or m["first_seen"], reverse=True)
        clusters.append({"members": members, "score": score_cluster(members, now)})
    clusters.sort(key=lambda c: c["score"], reverse=True)
    singles.sort(key=lambda a: (score_cluster([a], now)), reverse=True)
    return clusters, singles


def feed_health(feeds, state, now):
    rows = []
    for f in feeds:
        st = state.get(f["id"], {})
        days = st.get("new_per_day", {})
        today = now.strftime("%Y-%m-%d")
        full = [n for d, n in days.items() if d != today]
        rows.append({
            "id": f["id"], "category": f["category"], "outlet": f["outlet"], "feed": f["feed"], "url": f["url"],
            "confidence": f.get("confidence"),
            "status": "never checked" if not st.get("last_checked") else ("failing" if st.get("consecutive_failures", 0) else "ok"),
            "last_error": st.get("last_error"), "last_ok": st.get("last_ok"),
            "items_in_feed": st.get("items_in_feed"),
            "avg_new_per_day": round(sum(full) / len(full), 1) if full else None,
        })
    return rows


def write_markdown(path, digest):
    lines = [f"# Daily digest — generated {digest['generated_at']}", "",
             f"Articles considered: {digest['stats']['articles_considered']} · outlets: {digest['stats']['outlets']} · "
             f"stories: {digest['stats']['clusters_total']}", ""]
    for i, c in enumerate(digest["clusters"][:40], 1):
        lines.append(f"## {i}. {c['headline']}  [{c.get('region', '')} · {c.get('topic', '')}]")
        lines.append(f"{c['n_articles']} articles from {c['n_outlets']} outlets: {', '.join(c['outlets'][:8])}")
        lines.append("")
        lines.append(c["summary"])
        for kf in c.get("key_facts", []):
            lines.append(f"- {kf}")
        if c.get("outlet_disagreement"):
            lines.append(f"Framing: {c['outlet_disagreement']}")
        for a in c["articles"][:4]:
            lines.append(f"- {a['outlet']}: {a['url']}")
        lines.append("")
    lines.append("## Radar candidates (single-source items)")
    for r in digest["radar"][:30]:
        lines.append(f"- {r['outlet']}: {r['title']} — {r['url']}")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(now=None, embedder=None, summariser=None):
    now = now or utcnow()
    embedder = embedder or gemini.embed
    summariser = summariser or summarise
    allarts = load_jsonl(config.DATA / "articles.jsonl")
    cutoff = now - dt.timedelta(hours=config.DIGEST_HOURS)

    def when(a):
        return parse_iso(a.get("published")) or parse_iso(a["first_seen"])

    arts = [a for a in allarts
            if a.get("tagged") and a.get("hard_news", True)
            and a.get("topic") not in config.EXCLUDE_TOPICS and when(a) >= cutoff]
    print(f"{len(arts)} usable articles of {len(allarts)} stored")
    if not arts:
        print("nothing to digest")
        return None

    vectors = np.array(embedder([embed_text(a) for a in arts]))
    labels = cluster_vectors(vectors)
    clusters, singles = build_clusters(arts, labels, now)

    out_clusters = []
    for rank, c in enumerate(clusters[: config.MAX_CLUSTERS]):
        members = c["members"]
        info = {"headline": members[0]["title"], "summary": members[0].get("ai_summary", ""),
                "key_facts": [],
                "region": members[0].get("geography", ""), "topic": members[0].get("topic", ""),
                "coherent": True, "outlet_disagreement": ""}
        if rank < config.SUMMARISE_TOP:
            try:
                info.update({k: v for k, v in summariser(members).items() if k in info})
            except Exception as e:  # keep going with the fallback text
                print(f"summary failed for cluster {rank}: {type(e).__name__}")
        entities = []
        for m in members:
            for e in m.get("entities", []):
                if e not in entities:
                    entities.append(e)
        times = [when(m) for m in members]
        out_clusters.append({
            "rank": rank + 1, "score": c["score"], **info,
            "n_articles": len(members), "n_outlets": len({m["outlet"] for m in members}),
            "outlets": sorted({m["outlet"] for m in members}),
            "first_published": iso(min(times)), "last_published": iso(max(times)),
            "entities": entities[:8],
            "articles": [member_view(m) for m in members[:8]],
        })

    radar = [member_view(a) | {"urgency": a.get("urgency"), "topic": a.get("topic")} for a in singles[: config.RADAR_ITEMS]]

    feeds = load_feeds()
    state = load_json(config.DATA / "feed_state.json", {})
    health = feed_health(feeds, state, now)
    digest = {
        "generated_at": iso(now), "window_hours": config.DIGEST_HOURS,
        "stats": {
            "articles_stored": len(allarts), "articles_considered": len(arts),
            "outlets": len({a["outlet"] for a in arts}), "clusters_total": len(clusters),
            "single_source_items": len(singles),
            "feeds_total": len(feeds), "feeds_ok": sum(1 for h in health if h["status"] == "ok"),
            "feeds_failing": sum(1 for h in health if h["status"] == "failing"),
        },
        "clusters": out_clusters, "radar": radar,
    }
    save_json(config.DATA / "digest.json", digest)
    save_json(config.DATA / "feed_health.json", health)
    write_markdown(config.DATA / "digest_top.md", digest)
    try:
        brief_files.write(digest, clusters[: config.MAX_CLUSTERS], singles)
    except Exception as e:  # never lose the main digest because of the split files
        print(f"brief files failed: {type(e).__name__}: {e}")
    (config.DATA / ".nojekyll").write_text("")
    print(f"digest written: {len(out_clusters)} stories, {len(radar)} radar items")
    return digest


if __name__ == "__main__":
    run()
