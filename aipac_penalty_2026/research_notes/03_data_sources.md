# Research Note 3: Data-Source Catalog

*Deep-research report compiled by an AI research agent, 2026-08-12. Confidence flags:
[VERIFIED] = confirmed by direct fetch/search this week; [LIKELY] = strong secondary
evidence; [UNCERTAIN] = needs confirmation before budgeting.*

## 1. Campaign Finance (treatment measurement + fundraising covariates)

### 1.1 FEC — api.open.fec.gov + bulk data (the backbone; free)

**Key committee IDs** [VERIFIED via FEC.gov]:

| Committee | FEC ID | Type | Filing freq. |
|---|---|---|---|
| AIPAC PAC ("American Israel Public Affairs Committee Political Action Committee") | **C00797670** | Qualified membership-org PAC; lobbyist/registrant PAC | Monthly (Form 3X) |
| United Democracy Project (UDP) — AIPAC's super PAC | **C00799031** | Independent-expenditure-only (super PAC), monthly filer | Monthly + 24/48-hr IE reports |
| DMFI PAC (Democratic Majority for Israel) | **C00710848** | Hybrid/Carey committee (contributions **and** IEs — query both schedules) | — |

Committee pages: [AIPAC PAC](https://www.fec.gov/data/committee/C00797670/),
[UDP](https://www.fec.gov/data/committee/C00799031/). 2025–26 cycle to date (verified on
FEC page): AIPAC PAC raised ~$47.5M / distributed ~$44M to other committees; UDP has
raised ~$104M through June 2026 per
[FactCheck.org](https://www.factcheck.org/2026/08/united-democracy-project-3/). Watch
for additional affiliated vehicles this cycle — reporting describes a "pro-Israel super
PAC cinematic universe"
([American Prospect, June 2026](https://prospect.org/2026/06/22/pro-israel-super-pac-cinematic-universe/));
build the treatment from filings by committee ID, then audit new committee names each
month.

- **API**: `https://api.open.fec.gov/developers/` — free key from api.data.gov (1,000
  calls/hr default). Endpoints needed:
  - `/schedules/schedule_a/` (receipts) and `/schedules/schedule_b/` (disbursements,
    incl. AIPAC PAC's transfers to candidate committees).
  - `/schedules/schedule_e/` (processed independent expenditures) and
    `/schedules/schedule_e/efile/` (raw e-filings, near-real-time) — filter
    `committee_id=C00799031` (UDP) and `C00710848` (DMFI), with
    `support_oppose_indicator` (S/O) and `candidate_id`. This is the cleanest continuous
    treatment variable (dollars for/against, by date, so pre-primary exposure is
    measurable).
  - `/committee/{id}/totals/` and `/candidates/totals/` for cycle fundraising covariates.
- **Earmarked/conduit subtlety** [VERIFIED]: Most AIPAC PAC money is **earmarked
  contributions from individuals passed through the PAC as conduit** (capped
  $3,300/donor/candidate, not subject to the $5k PAC limit). On Schedule A/B these
  appear with **memo entries** ("total earmarked through conduit — limit not affected").
  When computing "AIPAC dollars to candidate X," use Schedule B disbursements from
  C00797670 and de-duplicate memo vs. non-memo lines, or you will double count. FEC
  guidance: [earmarked contributions](https://www.fec.gov/help-candidates-and-committees/filing-pac-reports/earmarked-contributions/)
  and [contributions through conduits](https://www.fec.gov/help-candidates-and-committees/filing-reports/contributions-received-through-conduits).
- **Bulk data**: `https://www.fec.gov/data/browse-data/?tab=bulk-data` — pipe-delimited
  files (`itcont`, `itoth`, `itpas2`, `oppexp`, plus IE files), refreshed **nightly**
  for the 2025–2026 cycle; separate header files define columns.
- **Update lag**: 24/48-hour IE reports hit the e-filing endpoints within hours of
  filing (UDP must file 24-hr reports near primaries) — effectively real-time treatment
  timing. Processed/itemized data lag days-to-weeks behind e-filings. AIPAC PAC monthly
  reports post ~20th of the following month; candidate committee totals are quarterly
  (so fundraising covariates for a June primary reflect the April quarterly +
  pre-primary report).
- **Python**: plain `requests` against OpenFEC (no maintained official client; community
  wrappers are stale — write a thin pager); `fecfile` (PyPI) parses raw .fec e-filings;
  bulk files load with `pandas.read_csv(sep='|')`.
- **Cost**: free.

### 1.2 OpenSecrets — pro-Israel sector aggregates

- **URLs**: [Pro-Israel industry summary (code Q05)](https://www.opensecrets.org/industries/indus?ind=Q05),
  [recipients](https://www.opensecrets.org/industries/summary?cycle=All&ind=Q05&recipdetail=S);
  PAC profiles for [UDP](https://www.opensecrets.org/outside-spending/detail/2024?cmte=C00799031&tab=summary)
  and DMFI.
- **What it adds**: the "pro-Israel" *sector* coding (Q05) aggregates many committees
  beyond AIPAC — useful for robustness treatment definitions; recipient tables by cycle
  1998–2024 (2026 cycle pages update on FEC filing schedule).
- **Access**: [bulk data](https://www.opensecrets.org/bulk-data) (free, **educational
  use only**, signup + approval required; compressed CSVs incl. CRP industry codes).
  **API status [UNCERTAIN]**: OpenSecrets laid off ~1/3 of staff after the 2024 cycle
  ([Common Dreams](https://www.commondreams.org/news/opensecrets)) and its API pages
  ([legacy](https://www.opensecrets.org/api),
  [new open-data portal](https://www.opensecrets.org/open-data/api)) suggest the legacy
  REST API has been sunset/replaced — verify current key issuance and rate limits before
  building on it. Treat OpenSecrets as convenience/validation; compute canonical numbers
  from FEC directly.
- **Python**: pandas on bulk CSVs; no reliable maintained API client in 2026.

### 1.3 DIME (Bonica, Stanford) — historical donors + ideology

- **URL**: [data.stanford.edu/dime](https://data.stanford.edu/dime);
  [Bonica's data page](http://web.stanford.edu/~bonica/data.html) [VERIFIED]
- **Coverage**: **v4.0, released early 2025, through the 2024 cycle** — 850M+ itemized
  contributions 1979–2024; CFscores for ~156k candidates, 37k committees, 36M donors.
  **No 2026 CFscores** — for 2026 non-incumbent ideology, either compute recipient
  CFscores from 2025–26 FEC itemized data (project onto DIME donor scores) or use v4.0
  scores for candidates with prior runs.
- **Access**: free bulk download (CSV/SQLite; tens of GB for the contribution DB; the
  candidate/committee "recipients" file is modest). Python: pandas/duckdb.
- **Use**: CFscores are the main challenger-ideology covariate; DIME also supports
  building pre-2026 pro-Israel giving histories for panel/priors.

## 2. Election Returns (outcome variable)

### 2.1 2026 primary results — the hard part (primaries ongoing as of Aug 2026)

- **State Secretary of State / election board sites** — the only *official* source
  in-season. Certified results typically **2–6 weeks after election day**; formats vary
  wildly (HTML, CSV, Clarity/Scytl JSON endpoints in GA/CO/etc., PDFs). Plan a per-state
  scraper with a normalization layer; unavoidable for certified candidate-level primary
  returns in 2026.
- **Associated Press Elections API** — near-real-time candidate-level results incl.
  primaries; JSON; used by ABC/CBS/NBC/CNN this cycle
  ([Yahoo/Variety](https://www.yahoo.com/news/articles/top-news-networks-shift-ap-210111851.html)).
  **Commercial license required; pricing unpublished** (AP stood up a dedicated
  election-data business unit in Jan 2026) — historically five figures for a cycle. The
  old NPR/NYT [`elex`](https://pypi.org/project/elex/) CLI targets AP API v2 and may
  need updating [UNCERTAIN]. Some outlets republish AP tables (e.g.,
  [Cardinal News for VA](https://cardinalnews.org/2026/08/04/primary-2026-election-results-from-associated-press/));
  [NPR's results pages](https://blog.apps.npr.org/) are AP-backed.
- **Ballotpedia** — every 2026 congressional primary has a structured page (candidate
  lists, incumbency, results tables updated election night, typically complete within ~1
  business day for federal races). Options: (a) **paid data services** —
  [Buy Political Data](https://ballotpedia.org/Ballotpedia:Buy_Political_Data), one-time
  CSV dumps from ~$600, [API v3](https://developer.ballotpedia.org/) subscriptions run
  to thousands/month; (b) **scraping** — MediaWiki; page HTML tables parse cleanly with
  `requests` + `pandas.read_html`/BeautifulSoup; respect ToS/rate limits. For a bounded
  universe (~470 congressional primaries), scraping is very feasible and is probably the
  best free candidate-level 2026 source, cross-validated against SoS certified numbers.
- **MEDSL** — will produce cleaned 2026 returns but historically with a **6–18 month
  lag** (precinct-level general-election files appear ~1 year later; no evidence of a
  2026 primary dataset yet [VERIFIED — GitHub/Dataverse show nothing for 2026]). Not
  usable in-season; excellent for later replication.
- **OurCampaigns** ([ourcampaigns.com](https://www.ourcampaigns.com)) — community-entered,
  includes essentially all primaries with candidate-level votes; scrapable HTML; quality
  good for federal races but verify against official sources; no bulk download.
- **Dave Leip's Atlas** ([uselectionatlas.org](https://uselectionatlas.org/)) — sells
  compiled official datasets (store:
  [store_data.php](https://uselectionatlas.org/BOTTOM/store_data.php)); site membership
  ~$435/yr for the 30-seat institutional tier; primary-level congressional data for 2026
  will lag certification [UNCERTAIN on exact product/price]. Several university
  libraries (UVA, MSU, Penn) license it.
- **MultiState** ([multistate.us/elections/primaries-2026](https://www.multistate.us/elections/primaries-2026))
  — free running state-by-state 2026 primary results tables; convenient scrape target
  for a first pass.

### 2.2 Historical primaries 2012–2024 (panel construction)

- **FEC "Federal Elections" biennial publications** — official primary *and* general
  candidate-level results for House/Senate every cycle, **Excel + PDF**, at fec.gov
  (Election results section). Canonical, free, covers 2012–2024. Best single panel
  source.
- **MEDSL Dataverse** ([medsl_election_returns](https://dataverse.harvard.edu/dataverse/medsl_election_returns))
  — includes a U.S. House **primary** results compilation for 2012–2018; precinct-level
  datasets 2016+ (mostly generals). Free CSVs; `pandas` or `pyDataverse`.
- Ballotpedia historical primary pages fill 2020–2024 gaps; DIME's recipients file also
  carries primary vote outcomes for older cycles.

### 2.3 Turnout denominators

- **Certified SoS returns** give total ballots per primary (usually sufficient — the
  vote-share denominator is within-contest).
- **Voter files**: **L2** has an academic program (many universities — Yale, NYU, Penn,
  MSU, WashU — hold institutional licenses; individual-level vote history incl. primary
  participation). **TargetSmart** (progressive-market; 265M+ records, results back to
  2012) — both are negotiated contracts, typically **$10k–$50k+/yr academic**
  [UNCERTAIN — no public pricing; check whether UW/IHME libraries already license L2].
  **Catalist** sells to academics via its analytics products; individual data access is
  restricted. For aggregate eligible-population denominators, Michael McDonald's
  **US Elections Project** (electproject.org) VEP estimates (general elections; primary
  turnout not standardized there).

## 3. Candidate & District Covariates

| Covariate | Source | Access | Notes |
|---|---|---|---|
| Incumbency, candidate lists, party | Ballotpedia pages; FEC candidate master (`cn` bulk file; `candidate_status=I/C/O`) | free | FEC incumbent flag is reliable for federal races |
| Fundraising totals | FEC `/candidates/totals/` | free API | quarterly lag + pre-primary reports |
| Endorsements | Ballotpedia endorsement pages (e.g., [Endorsements by AIPAC](https://ballotpedia.org/Endorsements_by_American_Israel_Public_Affairs_Committee), [by Track AIPAC](https://ballotpedia.org/Endorsements_by_Track_AIPAC)); FiveThirtyEight is **dead** (shut March 2025; [GitHub data archive](https://github.com/fivethirtyeight/data) incl. old [endorsement datasets](https://github.com/fivethirtyeight/data/tree/master/endorsements) still up, plus a [preservation mirror](https://github.com/Turn-Left-Now/FiveThirtyEight-Archive)); successors (Silver Bulletin, Strength in Numbers, [Split Ticket](https://split-ticket.org)) publish no structured 2026 endorsement tracker [LIKELY] | free/scrape | Build the endorsement panel from org press releases + Ballotpedia |
| Incumbent ideology | Voteview DW-NOMINATE — voteview.com/data, free CSVs, updated continuously through the 119th Congress | free | member_ideology CSV |
| Challenger ideology | DIME CFscores (v4.0 through 2024; compute 2026 yourself — §1.3) | free | flag uncertainty for first-time candidates |
| District partisanship | **Cook PVI 2026** [VERIFIED]: [2026 PVI released](https://www.cookpolitical.com/cook-pvi/2026-partisan-voting-index/district-map-and-list) reflecting **mid-decade re-redistricting in AL, CA, FL, LA, MO, NC, OH, TN, TX, UT**; spreadsheet/API for CPR subscribers (~$300–500/yr individual); the map/list page is scrapable | sub/scrape | Use 2026 PVI, not 2025 — boundaries changed |
| District election composites | [Dave's Redistricting](https://davesredistricting.org) — free, per-district election composites + demographics for current (incl. redrawn) maps; export via UI/JSON | free | best free PVI substitute |
| Demographics | Census ACS — api.census.gov (free key), ACS 5-year at congressional-district level. **Caveat**: ACS CD geography = 119th-Congress districts; re-redistricted 2026 districts need block-level rebuilds (Dave's Redistricting or NHGIS crosswalks) | free API | Python: `census`, `censusdis`, or raw requests |
| Jewish population share | **American Jewish Population Project** (Brandeis): [ajpp.brandeis.edu](https://ajpp.brandeis.edu/) — free registration; state/county/metro estimates, plus CD estimates built for the **116th Congress** (2015–2019 data, adjusted to 2020) [VERIFIED]. **No 2026-boundary CD file** — aggregate county estimates to current districts via county-to-CD crosswalks (Census/MCDC geocorr), or request a [custom analysis](https://ajpp.brandeis.edu/analysis) | free (registration) | flag measurement-era mismatch (pre-2020 data) |

## 4. Treatment Coding (AIPAC association)

Recommended: code treatment three ways — (a) AIPAC endorsement (binary), (b) AIPAC PAC
conduit dollars received (continuous, from FEC Sched B of C00797670), (c) UDP/DMFI IE
dollars for/against (continuous with sign, Sched E of C00799031/C00710848) — plus an
"anti-treatment" indicator (Reject AIPAC pledge).

- **AIPAC's own endorsement list**: AIPAC does **not** publish a clean public roster on
  aipac.org; endorsements surface via press releases/social posts ("75+ endorsed
  candidates advanced," etc.). The donor-facing PAC portal is login-gated [LIKELY]. Do
  not plan on scraping aipac.org; reconstruct from the sources below + AIPAC press
  releases.
- **Track AIPAC** — [trackaipac.com/candidates](https://www.trackaipac.com/candidates)
  [VERIFIED by fetch]: per-candidate 2026 pages with total "Israel lobby" funding,
  direct-vs-IE breakdown, and supporting PACs (AIPAC, DMFI, J Street, 19+ orgs), all
  sourced to FEC; includes a public **Google Sheets** PAC tracking spreadsheet; also an
  [endorsements page](https://www.trackaipac.com/endorsements) (anti-AIPAC slate).
  Active on X as @TrackAIPAC, which posted the "full list of 2026 endorsements."
  Scrapable; treat as a coding aid and independently re-derive dollars from FEC. (Domain
  is trackaipac.com — not aipactracker.org.)
- **Reject AIPAC pledge** — [rejectaipac.org/pledge](https://www.rejectaipac.org/pledge)
  and a [rejecters roster](https://www.rejectaipac.org/rejecters); the coalition
  maintains a **pledge-signer tracker spreadsheet**; related:
  [IfNotNow pledge](https://www.ifnotnowmovement.org/aipacpledge), an
  [Action Network form](https://actionnetwork.org/forms/anti-aipac-candidate-pledge),
  and [CAAC pledge](https://citizensagainstaipac.com/pledge). Fragmented; site had
  intermittent availability [UNCERTAIN] — archive pages (Wayback) while coding, and
  distinguish formal pledge signature from rhetorical refusal.
- **Journalistic race trackers**: [Jewish Insider](https://jewishinsider.com) covers
  essentially every AIPAC-relevant primary — narrative, not structured; good for
  adjudicating ambiguous cases.
  [JTA races-to-watch](https://www.jta.org/2026/07/02/politics/races-to-watch-as-staunch-israel-critics-notch-wins-these-candidates-could-be-next),
  [Sludge](https://readsludge.com/2026/03/01/here-is-how-much-aipac-has-funneled-to-every-member-of-congress/)
  (publishes member-level AIPAC dollar tables from FEC),
  [FundVoter](https://fundvoter.org/aipac),
  [Legis1](https://legis1.com/news/aipac-is-the-biggest-follow-the-money-pac-story-of-2026-and-its-getting-complicated).
- **DMFI PAC endorsements**: press releases at
  [dmfipac.org](https://dmfipac.org/news-updates/press-release/dmfi-pac-announces-2026-majority-project-endorses-first-slate-of-non-incumbent-candidates-for-the-2026-cycle/)
  — scrape the press-release archive for dated endorsement slates (gives endorsement
  *timing*, valuable for identification).

## 5. Polling / Survey / Market-Implied Counterfactuals

- **CES (Cooperative Election Study)** — [cces.gov.harvard.edu](https://cces.gov.harvard.edu/):
  the 2026 study runs a ~60k-sample pre-election wave (Oct) + post-election wave (Nov);
  **team data delivered March 2027, voter-file-matched summer 2027; public Common
  Content later in 2027** — too late for in-season analysis, right for the final paper.
  Team module buy-in (~$1k/question historically) is the vehicle for a bespoke AIPAC
  survey experiment — but 2026 module commitments may already be closed as of Aug 2026
  [UNCERTAIN — contact CES]. Past CES files (through 2024) on Harvard Dataverse. Python:
  `pyreadstat`/pandas.
- **Data for Progress** — [dataforprogress.org/latest-polling](https://www.dataforprogress.org/latest-polling):
  free crosstab PDFs/CSVs; directly relevant 2026 items:
  [July 2026 national poll on AIPAC/corporate-PAC funded Democrats](https://www.dataforprogress.org/blog/2026/7/28/aipac-ai-and-crypto-funded-democrats-underperform-democrats-who-reject-corporate-pac-money)
  (n=1,207 LVs, July 17–19) and an April MI-Sen primary poll with AIPAC questions
  ([Drop Site coverage](https://www.dropsitenews.com/p/michigan-senate-democratic-primary-poll-aipac));
  also the [Jewish Electorate Institute Spring 2026 poll](https://www.jewishelectorateinstitute.org/jei-s-spring-2026-latest-poll).
  Crosstabs only — microdata by request.
- **Prediction markets** (market-implied counterfactual trajectories around UDP ad-buy
  events — a natural event-study design):
  - **Kalshi**: official REST/WebSocket API (free key; regulated exchange); hosts
    district-level 2026 nominee markets. Full tick history via
    [Predexon](https://predexon.com/data/kalshi) (Parquet orderbook history, paid).
  - **Polymarket**: free public APIs (CLOB + Gamma) and the Polymarket Subgraph on The
    Graph (free <100k queries/mo); district-level House primary markets exist (e.g.,
    TX-23, TX-33, NC-04, NY-12 noted this cycle).
  - Aggregated historical dumps: [Kingsets](https://kingsets.com/) (daily CSV/BigQuery
    for both venues), [pmxt archive](https://archive.pmxt.dev/) (free JSON). Caveats:
    thin liquidity in House primary markets; **PredictIt** persists in reduced form
    post-CFTC settlement but is now third-choice [UNCERTAIN].
- **MRP inputs**: ACS PUMS via api.census.gov for poststratification frames; the
  re-redistricted CD geography caveat (§3) applies here too.

## Practical Priorities / Gaps to Flag

1. **Everything treatment-side is free and current**: FEC API + bulk gives real-time IEs
   (24/48-hr reports) and monthly AIPAC PAC conduit flows. Beware memo-entry double
   counting on earmarks.
2. **The binding constraint is 2026 primary returns at candidate level**: no free
   canonical dataset exists in-season. Cheapest defensible path: Ballotpedia scrape (or
   ~$600 one-time CSV) validated against SoS certified results; AP API only if budget
   allows.
3. **CFscores end at 2024** — plan to estimate 2026 challenger scores from itemized
   receipts.
4. **Geography churn**: mid-decade redistricting in 10 states means every district
   covariate (PVI, ACS, AJPP Jewish share) must be checked against *2026* boundaries;
   use the 2026 Cook PVI and Dave's Redistricting for redrawn maps.
5. **OpenSecrets API and rejectaipac.org availability are the two access details most in
   need of direct confirmation.**
