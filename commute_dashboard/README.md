# commute_dashboard

Live commute dashboard for the trip between **800 Hiawatha Pl S** (corner of Hiawatha &amp; Dearborn, Seattle) and the **Hans Rosling Center for Population Health** at the University of Washington. Designed to answer one question every weekday morning: *should I work from home, bike, take a bus, take light rail, or drive?*

**Live:** https://aflaxman.github.io/ai_assisted_research/commute_dashboard/

## What's on it

Above the fold, in order of decision-importance:

1. **Today's verdict** — a one-line recommendation (WFH / bike / bus / 2 Line / drive) computed from weather, arrivals, and travel time.
2. **Live arrivals** — three side-by-side panels:
   - **Rainier Ave NB** — routes 7, 106, 554 merged from stops at Bayview St and Charles St.
   - **23rd Ave NB** — route 48 at Wood Tech Center (S Lane St).
   - **2 Line · Judkins Park** — light rail in both directions, with stops discovered dynamically by OneBusAway location search.
3. **Weather** — current temp + 4-period forecast from the National Weather Service.
4. **Travel** — drive / bike / walk times. Bike includes elevation gain.

Below the fold:

- **Gas prices** — WA state Regular / Mid / Premium / Diesel averages, scraped live from AAA.
- **Air quality** — PM2.5 + PM10 + US AQI from Open-Meteo (no key).
- **Daylight** — sunrise / sunset / day length.
- **Bike, traffic cams, etc.** — curated links.

## Data sources (all free, no keys required for the defaults)

| Datum | Source | CORS-friendly? |
| --- | --- | --- |
| Weather | api.weather.gov (NWS) | yes |
| Drive / walk time | router.project-osrm.org | yes |
| Bike time (with elevation) | brouter.de | yes |
| Bus + light rail arrivals | api.pugetsound.onebusaway.org (`key=TEST` by default) | yes |
| Air quality | air-quality-api.open-meteo.com | yes |
| Daylight | api.sunrise-sunset.org | yes |
| Gas prices | gasprices.aaa.com (HTML scrape via corsproxy.io) | no — uses proxy |

`safeFetchJson()` tries direct fetch first and falls back to [corsproxy.io](https://corsproxy.io) only when CORS blocks the request. This makes the dashboard work both when hosted on GitHub Pages (most direct fetches succeed) and when opened as a local `file://` (where everything routes through the proxy).

## Settings

Click `⚙` in the header to open the drawer. You can override:

- **OBA API key** — `TEST` is shared and rate-limited. Get your own free key by emailing `oba_api_key@soundtransit.org` per the [Sound Transit OTD page](https://www.soundtransit.org/help-contacts/business-information/open-transit-data-otd). Stored in `localStorage`, never transmitted except to OBA.
- **Home and work coordinates** — `lat,lon` pairs. Recomputes all routing and the weather grid point.

## Why this exists

I wanted to know whether to bike or bus before walking out the door, without opening five tabs. Built with Claude.

## Files

- `index.html` — the entire dashboard (self-contained: HTML + CSS + JS, no build step)
- `README.md` — this file
- `GITHUB_ACTION_NOTES.md` — design notes for a future GitHub Action that would replace the AAA scrape with a committed `gas.json` (not yet implemented)
