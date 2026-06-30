# Can the Current Population Survey measure who *leaves* the country?

> **The short answer:** The CPS sees who leaves a household. It never sees where
> they go. That one gap is the whole story.

The Current Population Survey (CPS) re-interviews the same addresses month after
month, so it is tempting to think you could spot international out-migration
directly: here is a person in the household in January, gone in February —
surely some of them moved abroad. I chased that idea, read what others have
done, and then computed the numbers for Washington State from public data.

![Why the CPS panel cannot isolate emigration](outputs/figure_departure.png)

## TL;DR

- **Yes, the CPS has a longitudinal design** you can exploit: addresses stay in
  sample for 4 months, rest 8, return for 4 more. Consecutive months link by
  household ID + person line number.
- **No, you cannot read international out-migration off the monthly panel
  directly.** The CPS follows *addresses*, not *people*. When someone leaves, the
  survey simply stops seeing them — the basic monthly file and the recurring
  March supplement have **no destination field, no "moved abroad" code.** For a
  departure, a move to the next neighborhood and a move to Mumbai look identical.
  (The Census Bureau *has* asked directly, in one-off **emigration supplements**
  — 1987–1989 and August 2008 — but none became an ongoing series, and all share
  a fatal blind spot, below.)
- **Researchers have used the CPS for emigration anyway — indirectly.** The
  landmark method (Van Hook, Zhang, Bean & Passel, *Demography* 2006) links
  March CPS files a year apart and *statistically decomposes* the people who
  vanish into deaths, internal movers, and emigrants. It estimates a proportion;
  it never points at an individual and says "emigrant."
- **I computed it for Washington (2024).** Linking all 11 consecutive
  month-pairs of the 2024 basic monthly CPS, **1.14% of people in a continuing
  WA household leave it each month** (~13%/year). Foreign-born residents leave at
  **2.1%/month (~23%/year)** — but published foreign-born *emigration* is only
  ~1–3%/year (residual methods ~1–1.5%; the matched-CPS method ~2.9%).
  **The observed signal is roughly an order of magnitude (~8–20×) too large to be
  emigration.** It is dominated by ordinary domestic moves, deaths, and household
  change, and none of it carries a destination.
- **The honest method for a state-level emigration number is the residual /
  matched-CPS approach, not direct panel observation.**

## The problem: emigration is the hardest number in demography

A country's population changes through four channels: births, deaths,
immigration, and emigration. Three of them are well measured. The fourth is
not. As Van Hook and colleagues put it, "of the four components, emigration,
especially of the foreign-born, has proved the most difficult to gauge."

The reason is structural. People who immigrate arrive and can be counted.
People who emigrate *leave* — they are no longer here to fill out a survey. The
United States has no exit register and asks no one to check out. The Census
Bureau has said for decades that it lacks "a believable estimate of that elusive
process: emigration."

So when I noticed the CPS panel structure, I wondered: does the rotating design
quietly record emigration as attrition? Could a state like Washington be
estimated from the public files?

## How the CPS panel actually works

The CPS uses a **4-8-4 rotation**. A sampled address is interviewed for 4
consecutive months (month-in-sample, MIS, 1–4), left alone for 8 months, then
interviewed for 4 more (MIS 5–8). Consecutive monthly files therefore share most
of their addresses, and you can link a person from month *t* to month *t+1*.

The standard linkage keys (the same ones IPUMS uses to build `CPSIDP`):

| Level | Key | Validation |
|---|---|---|
| Household | `HRHHID` + `HRHHID2` | re-interview status `HRINTSTA` |
| Person | line number `PULINENO` within household | sex `PESEX`, race `PTDTRACE`, age `PRTAGE` (±) |

I verified this works: **98.6% of eligible Washington households reappear in the
next month's file**, and household + line keys are unique within a month. If the
keys were unstable, that match rate would be near zero.

The catch is what the file *does not* contain. I searched the full 2024 basic
monthly record layout. There is a country-of-**birth** field (`PENATVTY`), a
citizenship field (`PRCITSHP`), and an immigrant **year-of-entry** field
(`PRINUYER`) — all about where people *came from*. There is **nothing about
where a departing person goes.** When a household stops responding, the file
records a "Type A noninterview" (`HRINTSTA = 2`) — a bucket that lumps refusals,
no-one-home, and temporary absence together with families that actually moved.
A household that emigrated and a household that slammed the door are the same
code.

One precision worth making: the CPS *does* tell domestic from international moves
— but only for arrivals. The March (ASEC) migration battery asks current
residents where they lived a year ago, and "abroad" is one of the answers. That
is an *in*-migration flag; there is no symmetric *out*-migration flag, because
the person who would tick it has already left the sampling frame. The asymmetry
is the whole problem: you can survey the arrived, never the departed.

Even the one field that tracks roster changes, `PUCHINHH` ("change in household
composition"), does not help. It flags people *added* (codes 1–3) and
demographic edits (code 9), but in the public file a departing person gets no
record at all: across the entire nation in February 2024 it carried just **5**
"deleted" codes and **zero** "died" codes. A person who leaves is simply
**gone** — no reason, no destination, not even a death flag. The disappearance
*is* the only signal.

## What the panel shows for Washington

I linked all 11 consecutive month-pairs of the 2024 basic monthly CPS (Jan→Feb
… Nov→Dec), filtered to Washington (state FIPS 53), and classified every person
in a *continuing, re-interviewed* household as a **stayer** or a **leaver**.
Everything is weighted by the final person weight `PWSSWGT`. (Whole-household
departures are analyzed separately, because when a household stops responding we
cannot tell a move from a refusal.)

**Washington, 2024 (pooled over 11 month-pairs, 14,337 person-pairs):**

| Group | Monthly departure rate | Annualized | Leavers / person-pairs |
|---|---|---|---|
| All persons | **1.14%/mo** (weighted) | ~12.9%/yr | 155 / 14,337 |
| U.S.-born | 0.96%/mo | ~10.9%/yr | 110 / 12,162 |
| Foreign-born | **2.14%/mo** | ~22.8%/yr | 45 / 2,175 |

Washington tracks the nation almost exactly (US: 1.17%/mo overall, foreign-born
1.75%/mo), which is reassuring for such a small state sample.

At the household level, **7.9% of Washington households scheduled to return were
not cleanly re-interviewed** the next month (Type A noninterviews dominate, plus
a few vacancies and disappearances). Whole-household moves abroad live inside
that 7.9% — indistinguishable from the refusals that make up most of it.

### Why this cannot be emigration

The foreign-born leave their household at ~2.1%/month, more than twice the
U.S.-born rate. The tempting story writes itself: the extra departures are
people going home. The data refuse that story.

- **Magnitude.** A 2.1%/month household-departure rate annualizes to ~23%/year.
  The published foreign-born *emigration* rate is ~1–3%/year — residual methods
  put it near 0.9–1.5% (e.g., Mulder 2003 ≈ 0.9%), the matched-CPS method higher
  at ~2.9% on average and up to ~3.8% for the most recent arrivals (Van Hook et
  al.). Even taking the high end, emigration could account for only **about a
  tenth** of the foreign-born departures the panel sees; on the residual figure,
  a twentieth.
- **Confounding.** The foreign-born are younger, more likely to rent, and more
  residentially mobile *within* the U.S. — exactly the people who change address
  often, domestically. The CPS cannot separate that from an international move.
- **It is happening right now, with the same caveat.** In August 2025 Pew
  Research used CPS monthly tabulations to report the U.S. foreign-born
  population fell by ~1.4 million in the first half of 2025 — and warned the drop
  "may in part be due to technical reasons such as declining CPS survey
  participation among immigrants." That is the Type A noninterview problem,
  live, in the most-watched migration number of the year.

So the panel gives you a real, weightable, state-level number — the rate at
which people leave their households — and that number is *gross residential
churn*, not emigration. International out-migration is in there. You just cannot
get it out.

## They tried asking directly — and gave up

Before turning to indirect methods, it is worth knowing that the Census Bureau
*did* try the direct route. The CPS has carried **emigration supplements** — in
July 1987, June 1988, and November 1989 (analyzed with a "multiplicity" design),
and again in the **August 2008 Immigration/Emigration Supplement**. The 2008
supplement asked surviving households to name members and prior co-residents now
*living outside the United States*, including the **destination country** and the
emigrant's demographics. Census documentation called it the only nationally
representative source on emigration from the U.S.

So why isn't it the answer? Because of one structural flaw, the same one that
sinks the monthly panel: it is a **"left-behind reporter"** design. It can only
find an emigrant who left behind a household with someone still here to report
them. A household that emigrated *in full* leaves no one to answer the door —
exactly the case most central to measuring out-migration. Add that it was
one-off (no continuing series) and too small for a state like Washington, and
the direct approach collapses back into estimation.

## How the pros do it: indirect estimation

If you cannot observe emigrants leaving, you infer them from who is *missing*.

- **Residual method (Census Bureau).** Take the foreign-born population counted
  at two points in time. Add known new arrivals, subtract deaths. Whatever stock
  is *missing* relative to that bookkeeping is attributed to emigration. The
  Bureau's modern population estimates build net international migration this way
  from the American Community Survey, with the CPS monthly files as a more-timely
  benchmark and administrative data (e.g., DHS) as adjustments.
- **Matched-CPS decomposition (Van Hook, Zhang, Bean & Passel, *Demography*
  2006, 43(2):361–382).** This is the closest anyone has come to the idea that
  started this post — and it is essentially my Washington exercise done right.
  They link the March CPS to the next year's March CPS and treat everyone present
  in the first but absent in the second as a *mixture* of deaths, internal
  movers, and emigrants. The attrition rate `a` decomposes as
  `a = m + d + e + r` (internal migration + death + emigration + other
  nonresponse), so emigration is the residual `e = a − m − d − r`: internal
  migration `m` comes from the "lived elsewhere a year ago" question, mortality
  `d` from NHIS-linked death models, and `r` from a same-rate assumption against
  the U.S.-born second generation. It matches residual methods for long-term
  residents and runs *higher* for recent arrivals — where the authors argue it
  is more accurate. Applied to 1995–2009 data it puts foreign-born emigration at
  about **2.9%/year on average** (≈3.8%/year in the first five years after
  arrival, falling toward ~0.8% later). Crucially, it works at the **annual**
  link and never identifies an individual emigrant; it estimates a proportion.

Both methods share a feature my month-to-month exercise lacks: they do not
pretend the survey saw the emigration. They model it — and they get the monthly
churn out of the way first, exactly because, as Panel B shows, that churn would
otherwise swamp the signal.

### An order-of-magnitude figure for Washington

Putting the honest method to work at the back of an envelope: Washington's
foreign-born population is roughly **1.25–1.3 million** (~16% of the state,
weighted from the same CPS files; the monthly estimate bounces between ~14% and
~18% on sampling noise). Applying the published foreign-born emigration rate —
~1–1.5%/year (residual) up to ~2.9%/year (matched-CPS) — gives a rough
**13,000–37,000 foreign-born emigrants per year from Washington**, plus a much
smaller, lower-rate flow of U.S.-born emigrants. That is a *literature-anchored
estimate*, not something the CPS panel measured. And even its high end is
dwarfed by the ~290,000 foreign-born WA residents (~23% of ~1.27M) the panel
sees leave their households each year for reasons it cannot label.

## Reproduce it

No API key needed; the public-use files are open.

```bash
# 1. set up an isolated environment (uv)
uv venv .venv && uv pip install -r requirements.txt

# 2. download the 2024 basic monthly files (~120 MB total, ZIP-wrapped .dat)
.venv/bin/python download_data.py --year 2024 --out cps_data

# 3. run the full Washington + national analysis
.venv/bin/python analyze_wa.py cps_data

# 4. (re)build the figure, run the tests
.venv/bin/python make_figure.py
.venv/bin/python test_link_months.py
```

Outputs land in `outputs/`: per-pair and pooled summaries, and the figure above.

| File | What it does |
|---|---|
| [`cps_parse.py`](cps_parse.py) | Read a monthly fixed-width file into the columns needed |
| [`link_months.py`](link_months.py) | Link month *t* → *t+1*; classify stayers/leavers/households |
| [`analyze_wa.py`](analyze_wa.py) | Pool all 2024 month-pairs; weighted rates by nativity |
| [`make_figure.py`](make_figure.py) | The two-panel figure |
| [`test_link_months.py`](test_link_months.py) | Synthetic-fixture tests of the linkage logic |

## Caveats (read these)

- **No destination field — the whole point.** Every "leaver" here has an unknown
  destination. This is a measure of *gross household departure*, not emigration.
- **No design-based standard errors.** The public basic monthly file omits the
  design variables (`SDMVSTRA`/`SDMVPSU`); proper variance needs replicate
  weights or a generalized variance function. The unweighted counts are shown so
  small cells are visible; treat the foreign-born WA rate as indicative.
- **Pooled month-pairs are not independent.** The rotation design means the same
  household appears in several consecutive pairs, so the pooled rate is a
  descriptive average, not 11 independent samples.
- **Annualizing is illustrative.** `1-(1-m)^12` assumes a constant monthly hazard
  and is for intuition only.
- **Temporary absence ≈ departure here.** A usual resident away for one month can
  look like a leaver, inflating the rate — another reason it overstates true
  out-mobility, let alone emigration.

## Further reading

- Van Hook, J., Zhang, W., Bean, F. D., & Passel, J. S. (2006). *Foreign-born
  emigration: A new approach and estimates based on matched CPS files.*
  **Demography**, 43(2), 361–382. doi:10.1353/dem.2006.0013
- U.S. Census Bureau population estimates methodology for **net international
  migration** (ACS-based residual approach; CPS as benchmark).
- Fernandez, E. W. (1995). *Estimation of the annual emigration of U.S.-born
  persons.* Census Bureau Population Division Working Paper POP-twps0010.
- U.S. Census Bureau (2008). **August 2008 CPS Immigration/Emigration
  Supplement** — the one-off direct emigration instrument (microdata, technical
  documentation `cpsaug08`, and debriefing report POP-twps0099). Earlier
  emigration supplements: July 1987, June 1988, November 1989 (multiplicity
  method; cf. Woodrow-Lapham).
- Pew Research Center (Aug 2025), CPS-based estimates of the 2025 immigrant
  population decline (with nonresponse caveats).
- IPUMS-CPS documentation on linking the CPS over time (`CPSIDP`); Drew, Flood &
  Warren (2014) on the CPS longitudinal design.

---

*Built with public Census Bureau microdata and `uv`. Data files are not checked
in; `download_data.py` fetches them.*
