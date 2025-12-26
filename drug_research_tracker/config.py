"""
Configuration for the Drug Research News Tracker.
Customize keywords and sources here.
"""

# Keywords to track - articles matching these will be included
# More specific terms get higher relevance scores
KEYWORDS = {
    # Core simulation/modeling terms (high weight)
    "high": [
        "simulation",
        "computational model",
        "agent-based model",
        "microsimulation",
        "system dynamics",
        "mathematical model",
        "predictive model",
        "markov model",
        "monte carlo",
        "discrete event simulation",
    ],
    # Drug/opioid terms (medium weight - combined with above)
    "medium": [
        "opioid",
        "opioids",
        "fentanyl",
        "heroin",
        "methadone",
        "buprenorphine",
        "naloxone",
        "narcan",
        "overdose",
        "substance use disorder",
        "opioid use disorder",
        "OUD",
        "medication-assisted treatment",
        "MAT",
        "harm reduction",
        "drug policy",
        "drug epidemic",
    ],
    # Broader context terms (lower weight)
    "low": [
        "addiction",
        "substance abuse",
        "drug use",
        "public health",
        "epidemiology",
        "health policy",
        "treatment",
        "prevention",
        "intervention",
    ],
}

# RSS/Atom feed sources
FEEDS = {
    "pubmed": {
        "name": "PubMed - Opioid Simulation",
        "url": "https://pubmed.ncbi.nlm.nih.gov/rss/search/1234/?limit=50&utm_campaign=pubmed-2&fc=20231201000000",
        # We'll construct search URLs dynamically
        "category": "research",
        "search_terms": "opioid AND (simulation OR modeling OR computational)",
    },
    "arxiv_q-bio": {
        "name": "arXiv - Quantitative Biology",
        "url": "https://rss.arxiv.org/rss/q-bio",
        "category": "research",
    },
    "arxiv_stat": {
        "name": "arXiv - Statistics Applications",
        "url": "https://rss.arxiv.org/rss/stat.AP",
        "category": "research",
    },
    "cdc_mmwr": {
        "name": "CDC MMWR",
        "url": "https://tools.cdc.gov/api/v2/resources/media/rss/mmwr.rss",
        "category": "policy",
    },
    "samhsa": {
        "name": "SAMHSA News",
        "url": "https://www.samhsa.gov/rss/samhsa-newsroom.xml",
        "category": "policy",
    },
    "nih_news": {
        "name": "NIH News",
        "url": "https://www.nih.gov/news-events/news-releases/feed",
        "category": "policy",
    },
    "drugabuse_news": {
        "name": "NIDA News",
        "url": "https://nida.nih.gov/news-events/news-releases/feed",
        "category": "policy",
    },
}

# PubMed search query (constructed dynamically)
PUBMED_SEARCHES = [
    {
        "name": "Opioid Simulation Models",
        "query": '(opioid OR fentanyl OR "opioid use disorder") AND (simulation OR "mathematical model" OR "computational model" OR "agent-based")',
    },
    {
        "name": "Drug Policy Modeling",
        "query": '"drug policy" AND (simulation OR modeling OR "cost-effectiveness")',
    },
    {
        "name": "Overdose Prevention Models",
        "query": '(overdose OR naloxone) AND (simulation OR "predictive model" OR forecasting)',
    },
]

# How often to refresh feeds (in minutes)
REFRESH_INTERVAL = 60

# Database path
DATABASE_PATH = "drug_research.db"

# Number of days to keep articles
RETENTION_DAYS = 90
