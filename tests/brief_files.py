"""Stage 3b: split the digest into small files the morning brief can read in one fetch each.

The full digest.json is ~250-300 KB, but the brief's fetch tool only reads about the first 60 KB,
so it saw ~26 of 120 stories and none of the radar. These files are each kept under MAX_BYTES and
are organised by the brief's sections:

  brief/index.json      what's here, stats and which files to read
  brief/top.json        the highest-ranked stories overall (for "what matters now")
  brief/economy.json    economic currents: markets, central banks, energy, trade, currencies
  brief/politics.json   power & politics outside the US, UK and Egypt
  brief/us.json         US monitor
  brief/uk.json         UK monitor
  brief/egypt.json      Egypt monitor
  brief/radar.json      the strongest single-source items not used above

A story can appear in more than one file (e.g. a Bank of England decision is in both uk and economy).
"""
import json
import re

from . import config

MAX_BYTES = int(__import__("os").environ.get("BRIEF_FILE_MAX_BYTES", "40000"))
MAX_SECTION_CLUSTERS = 18   # leave room in each section file for single-source items
BASE_URL = "https://raw.githubusercontent.com/ygnassim96/early-bird/data/brief/"

US_STATES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut", "delaware", "florida",
    "georgia-us", "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine",
    "maryland", "massachusetts", "michigan", "minnesota", "mississippi", "missouri", "montana", "nebraska",
    "nevada", "new-hampshire", "new-jersey", "new-mexico", "new-york", "north-carolina", "north-dakota", "ohio",
    "oklahoma", "oregon", "pennsylvania", "rhode-island", "south-carolina", "south-dakota", "tennessee", "texas",
    "utah", "vermont", "virginia", "washington", "west-virginia", "wisconsin", "wyoming",
}
GEO = {
    "us": {"us", "usa", "u.s.", "united-states", "united states", "america", "washington-dc", "washington dc",
           "new york", "los-angeles", "los angeles", "chicago"} | US_STATES,
    "uk": {"uk", "u.k.", "united-kingdom", "united kingdom", "britain", "great-britain", "england", "scotland",
           "wales", "northern-ireland", "northern ireland", "london"},
    "egypt": {"egypt", "cairo", "alexandria", "suez", "sinai", "giza"},
}
TEXT = {
    "us": re.compile(r"\b(U\.S\.|US|United States|White House|Congress|Senate|Pentagon|Federal Reserve|Trump)\b"),
    "uk": re.compile(r"\b(UK|U\.K\.|Britain|British|England|Scotland|Wales|London|Westminster|Starmer|NHS|"
                     r"Bank of England|Downing Street)\b"),
    "egypt": re.compile(r"\b(Egypt|Egyptian|Cairo|Suez|Sisi|Sinai)\b"),
}
ECON = re.compile(r"market|stock|share|econom|inflation|interest|rate|central-bank|central bank|fed\b|tariff|trade|"
                  r"oil|energy|gas|opec|currenc|pound|dollar|bond|yield|debt|gdp|jobs|employment|earnings|imf|"
                  r"budget|tax|bank", re.I)
POLITICS_TOPICS = {"politics", "international", "security"}


def _geo(a) -> str:
    return (a.get("geography") or "").strip().lower()


def _country(a, key) -> bool:
    if _geo(a) in GEO[key]:
        return True
    text = " ".join([a.get("title", ""), " ".join(a.get("entities") or [])])
    return bool(TEXT[key].search(text)) and key != "us"  # "US" appears in too many global stories to trust text


def _econ(a) -> bool:
    if a.get("topic") == "business":
        return True
    return bool(ECON.search(" ".join([a.get("subtopic", ""), " ".join(a.get("tags") or [])])))


def _share(members, test) -> float:
    return sum(1 for m in members if test(m)) / max(1, len(members))


def _link(a) -> str:
    return a.get("real_url") or a["url"]


def _cut(s, n):
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def compact_cluster(c, members=None):
    arts = members if members is not None else c.get("articles", [])
    return {
        "rank": c["rank"], "headline": c["headline"], "summary": _cut(c.get("summary"), 420),
        "region": c.get("region", ""), "topic": c.get("topic", ""), "coherent": c.get("coherent", True),
        "outlet_disagreement": _cut(c.get("outlet_disagreement"), 300),
        "n_articles": c["n_articles"], "n_outlets": c["n_outlets"], "outlets": c.get("outlets", [])[:6],
        "entities": c.get("entities", [])[:6],
        "articles": [{"title": _cut(a["title"], 120), "outlet": a["outlet"],
                      "url": _link(a)} for a in arts[:3]],
    }


def compact_single(a):
    return {"title": _cut(a["title"], 140), "outlet": a["outlet"], "url": _link(a),
            "summary": _cut(a.get("ai_summary") or a.get("summary"), 200),
            "topic": a.get("topic", ""), "geography": a.get("geography", ""), "urgency": a.get("urgency", "")}


def _size(obj) -> int:
    return len(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _fill(header, clusters, singles):
    """Add clusters, then singles, in order, while the file stays under MAX_BYTES."""
    doc = dict(header, clusters=[], singles=[])
    left_c = left_s = 0
    for c in clusters:
        doc["clusters"].append(c)
        if _size(doc) > MAX_BYTES:
            doc["clusters"].pop()
            left_c = len(clusters) - len(doc["clusters"])
            break
    for s in singles:
        doc["singles"].append(s)
        if _size(doc) > MAX_BYTES:
            doc["singles"].pop()
            left_s = len(singles) - len(doc["singles"])
            break
    doc["omitted"] = {"clusters": left_c, "singles": left_s}
    return doc


def _write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def write(digest, clusters, singles):
    """digest: the dict saved as digest.json. clusters: ranked internal clusters with full members
    (aligned with digest['clusters']). singles: every single-source article, best first."""
    out = config.DATA / "brief"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.json"):
        old.unlink()

    pairs = list(zip(digest["clusters"], clusters))   # (summary view, internal with members)
    view = lambda c, inner: compact_cluster(c, inner["members"])

    def country(key):
        return [view(c, i) for c, i in pairs
                if _share(i["members"], lambda m: _country(m, key)) >= 0.5]

    sections = {
        "top": ("Highest-ranked stories overall, for 'what matters now'.",
                [view(c, i) for c, i in pairs[:25]], []),
        "economy": ("Economic currents: markets, central banks, energy, trade, currencies, companies.",
                    [view(c, i) for c, i in pairs if _share(i["members"], _econ) >= 0.5],
                    [compact_single(a) for a in singles if _econ(a)]),
        "politics": ("Power & politics worldwide, excluding stories mainly about the US, UK or Egypt.",
                     [view(c, i) for c, i in pairs
                      if c.get("topic") in POLITICS_TOPICS
                      and not any(_share(i["members"], lambda m, k=k: _country(m, k)) >= 0.5 for k in GEO)],
                     [compact_single(a) for a in singles
                      if a.get("topic") in POLITICS_TOPICS and not any(_country(a, k) for k in GEO)]),
        "us": ("US monitor: stories mainly about the United States.", country("us"),
               [compact_single(a) for a in singles if _country(a, "us")]),
        "uk": ("UK monitor: stories mainly about the United Kingdom.", country("uk"),
               [compact_single(a) for a in singles if _country(a, "uk")]),
        "egypt": ("Egypt monitor: stories mainly about Egypt.", country("egypt"),
                  [compact_single(a) for a in singles if _country(a, "egypt")]),
    }

    used = set()
    index_files = []
    for name, (desc, cl, sg) in sections.items():
        if name != "top":
            cl = cl[:MAX_SECTION_CLUSTERS]
        doc = _fill({"section": name, "description": desc, "generated_at": digest["generated_at"]}, cl, sg)
        _write(out / f"{name}.json", doc)
        used.update(s["url"] for s in doc["singles"])
        index_files.append({"file": f"{name}.json", "url": BASE_URL + f"{name}.json", "description": desc,
                            "clusters": len(doc["clusters"]), "singles": len(doc["singles"]),
                            "omitted": doc["omitted"]})

    radar = [compact_single(a) for a in singles if _link(a) not in used]
    doc = _fill({"section": "radar", "description": "Strongest single-source items not already listed in a section file.",
                 "generated_at": digest["generated_at"]}, [], radar)
    _write(out / "radar.json", doc)
    index_files.append({"file": "radar.json", "url": BASE_URL + "radar.json", "description": doc["description"],
                        "clusters": 0, "singles": len(doc["singles"]), "omitted": doc["omitted"]})

    index = {
        "generated_at": digest["generated_at"], "window_hours": digest.get("window_hours"),
        "stats": digest["stats"],
        "how_to_read": "Fetch each file listed below; each is small enough to read in full. Clusters are stories "
                       "covered by several outlets, ranked best first (rank is the overall rank). Singles are "
                       "stories from one outlet. A story can appear in more than one file.",
        "files": index_files,
    }
    _write(out / "index.json", index)
    print("brief files: " + ", ".join(f"{f['file']} {f['clusters']}c/{f['singles']}s" for f in index_files))
    return index
