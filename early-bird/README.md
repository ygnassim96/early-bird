# Early Bird

A Meridian-style daily news pipeline.

Runs free on GitHub Actions. Every hour it checks the feeds in `feeds.csv` that are due, stores new
articles for 48 hours, tags each one with Gemini (topic, region, urgency, entities, one-line summary),
and once a day groups them into stories and writes a digest.

```
feeds.csv ──collect──> articles.jsonl ──enrich (Gemini)──> tagged articles ──digest──> digest.json
 (hourly)                (48h rolling)                      (embeddings + HDBSCAN        digest_top.md
                                                             clusters, top stories       feed_health.json
                                                             summarised)
```

Results live on a single-commit `data` branch (it is replaced on every run, so the repo never grows).

## Setup (about 10 minutes)

1. Create a **public** GitHub repository (public is needed for free GitHub Pages) and upload everything in this folder.
   Only headlines, links and summaries are published. Your key is never in the repo.
2. Get a Google AI key at https://aistudio.google.com/apikey.
3. In the repo: **Settings → Secrets and variables → Actions → New repository secret**.
   Name `GEMINI_API_KEY`, value = your key. Paste it only there.
4. **Settings → Actions → General → Workflow permissions → Read and write permissions**. Save.
5. **Actions** tab → run **collect** manually with "Check every feed now" ticked. Check the log, then run **digest**.
6. After the first run: **Settings → Pages → Build and deployment → Deploy from a branch → `data` / `(root)`**.
   Your digest will be at `https://<you>.github.io/<repo>/digest_top.md` and `digest.json`.
7. Open `feed_health.json` to see which feeds work, which fail, and how many new articles each produces per day.

## Settings (optional, as repository variables)

| Variable | Default | What it does |
|---|---|---|
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model used for tagging and story summaries. Change it if Google retires this one. |
| `GEMINI_EMBED_MODEL` | `gemini-embedding-001` | Model used to group similar articles. |
| `GEMINI_RPM` | `12` | Requests per minute. Lower it if you hit rate limits. |
| `EXCLUDE_TOPICS` | `sports,culture,other` | Topics left out of the digest. |

## Notes

- Article text is only fetched where the site's robots.txt allows it and the outlet is not paywalled.
  Otherwise the RSS summary is used, which is enough for tagging and grouping.
- Reuters, AP and AFP have no public feeds, so they are read through Google News searches.
- Feeds only show their latest 15–25 items. The `poll_tier` column (1 = hourly, 2 = 4 hours, 3 = 6 hours, 4 = daily) keeps busy feeds from scrolling past.
- Feeds that fail three times in a row are checked less often (up to once a day).
- GitHub pauses scheduled workflows in a public repo after 60 days with no repository activity. If the digest stops, open the Actions tab and re-enable it.
- Run the tests with `pip install -r requirements.txt pytest && pytest`.
