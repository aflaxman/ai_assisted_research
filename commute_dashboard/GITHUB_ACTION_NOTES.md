# Future: GitHub Action to publish gas prices

The dashboard's gas panel currently scrapes [gasprices.aaa.com](https://gasprices.aaa.com/?state=WA) at page load. AAA doesn't send CORS headers, so the request goes through [corsproxy.io](https://corsproxy.io). This works but introduces a third-party dependency and is fragile if AAA ever rate-limits the proxy.

The fix is to do the scrape **server-side** in CI, commit the result as a JSON file in the repo, and have the dashboard fetch `./gas.json` — same origin, zero CORS, no proxy. Cost: nothing. Reliability: high.

This file documents the design. **It is not yet implemented** — implement when corsproxy.io flakes out enough to bother fixing.

## Design

Add `commute_dashboard/.github/workflows/scrape-gas.yml` at the repo root (`.github/workflows/scrape-gas.yml`):

```yaml
name: Scrape AAA gas prices

on:
  schedule:
    - cron: '17 * * * *'   # every hour at :17 (avoid the top-of-hour stampede)
  workflow_dispatch:        # manual trigger button

permissions:
  contents: write           # needed to commit the JSON back

jobs:
  scrape:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Scrape and write gas.json
        run: |
          python <<'PY'
          import json, re, urllib.request, datetime
          html = urllib.request.urlopen("https://gasprices.aaa.com/?state=WA").read().decode("utf-8", "ignore")
          def grab(label):
              m = re.search(rf"{label}[\s\S]{{0,400}}?\$([0-9]+\.[0-9]+)", html, re.I)
              return float(m.group(1)) if m else None
          data = {
              "fetched_at": datetime.datetime.utcnow().isoformat() + "Z",
              "state": "WA",
              "regular":  grab("Regular"),
              "midgrade": grab("Mid-?\\s?Grade") or grab("Midgrade"),
              "premium":  grab("Premium"),
              "diesel":   grab("Diesel"),
              "source":   "https://gasprices.aaa.com/?state=WA",
          }
          with open("commute_dashboard/gas.json", "w") as f:
              json.dump(data, f, indent=2)
          print(json.dumps(data, indent=2))
          PY

      - name: Commit if changed
        run: |
          git config user.name  "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add commute_dashboard/gas.json
          git diff --cached --quiet || git commit -m "chore: update gas prices [skip ci]"
          git push
```

## Dashboard change

In `index.html`, replace the body of `loadGas()` so it reads `./gas.json` instead of scraping AAA at runtime:

```js
async function loadGas() {
  try {
    const resp = await fetch('./gas.json', { cache: 'no-store' });
    if (!resp.ok) throw new Error('gas.json ' + resp.status);
    const data = await resp.json();
    // …render same UI as before, using data.regular / .midgrade / .premium / .diesel …
  } catch (err) {
    // existing GasBuddy-links fallback
  }
}
```

The `cache: 'no-store'` matters: GitHub Pages caches aggressively, and we want a fresh read each refresh.

## Tradeoffs

- **Hourly cron is approximate** — GitHub Actions cron is queued, sometimes delayed by 5–15 min during high load. Fine for gas prices, which barely move intraday.
- **Free-tier minutes** — public repos get unlimited Action minutes, so cost is zero for `aflaxman/ai_assisted_research`.
- **Git history bloat** — one tiny commit per hour = ~8,760/year. Negligible repo size, but the commit log gets noisy. If that bothers you, change the cron to every 6h or have the action squash via `--amend --force-push` to a `gas-data` branch instead of `main`.
- **AAA selector drift** — same risk as the current scrape: if AAA redesigns their page, the regex breaks. Difference: with the Action, the failure is silent in the Actions log instead of visible in the dashboard. Worth adding a Slack/email notify on failure.

## Other API-key dependencies you could similarly remove

The same pattern (server-side fetch + commit JSON) works for anything that:
1. Needs an API key you don't want in client JS, or
2. Doesn't send CORS headers.

Possible candidates if you ever want them:
- Real-time traffic (Google Distance Matrix needs a key) → commit `traffic.json` every 10 min during commute hours.
- Detailed weather radar imagery → commit hourly composite.
- Air quality from PurpleAir (their API needs a key) → commit `aqi.json` every 15 min.

For now, the dashboard's existing data sources are all free + CORS-friendly enough that this isn't needed.
