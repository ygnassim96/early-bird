"""Shared settings. Everything can be overridden with environment variables."""
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = pathlib.Path(os.environ.get("DATA_DIR", ROOT / "data"))
FEEDS_CSV = pathlib.Path(os.environ.get("FEEDS_CSV", ROOT / "feeds.csv"))

# Articles older than this are dropped (Meridian keeps 48 hours).
RETAIN_HOURS = int(os.environ.get("RETAIN_HOURS", "48"))
# The digest looks at this many hours of articles.
DIGEST_HOURS = int(os.environ.get("DIGEST_HOURS", "30"))

UA_TOKEN = "EarlyBird"
UA = f"Mozilla/5.0 (compatible; {UA_TOKEN}/1.0)"

# How often each feed tier is checked, in minutes.
TIER_MINUTES = {1: 60, 2: 240, 3: 360, 4: 1440}

# Outlets that hide article text behind a paywall: use the RSS summary only.
PAYWALLED_DOMAINS = {
    "wsj.com", "ft.com", "barrons.com", "haaretz.com", "economist.com",
    "bloomberg.com", "washingtonpost.com", "nytimes.com", "scmp.com",
    "asia.nikkei.com", "nikkei.com", "foreignaffairs.com",
}

# Topics left out of the digest (the brief is about world affairs).
EXCLUDE_TOPICS = set(
    t.strip() for t in os.environ.get("EXCLUDE_TOPICS", "sports,culture,other").split(",") if t.strip()
)

MAX_ENRICH_PER_RUN = int(os.environ.get("MAX_ENRICH_PER_RUN", "500"))
# Decoding is just an HTTP request per link, with no Gemini cost, so this can run
# well ahead of the tag cap above: raised from 150 to clear the Google News backlog
# left by the decoder bug (fixed 2026-11-07) within a few runs instead of a day.
MAX_GNEWS_DECODE_PER_RUN = int(os.environ.get("MAX_GNEWS_DECODE_PER_RUN", "800"))
TAG_BATCH = int(os.environ.get("TAG_BATCH", "12"))
TEXT_CHARS = int(os.environ.get("TEXT_CHARS", "1000"))

MAX_CLUSTERS = int(os.environ.get("MAX_CLUSTERS", "120"))
SUMMARISE_TOP = int(os.environ.get("SUMMARISE_TOP", "60"))
RADAR_ITEMS = int(os.environ.get("RADAR_ITEMS", "60"))
