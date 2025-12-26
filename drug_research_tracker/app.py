"""
Drug Research News Tracker - A web app to stay current on
computer simulation research in the drug use and opioid disorder space.
"""

import sqlite3
import hashlib
import re
from datetime import datetime, timedelta
from urllib.parse import quote_plus
import feedparser
from flask import Flask, render_template, request, jsonify, g

from config import KEYWORDS, FEEDS, PUBMED_SEARCHES, DATABASE_PATH, RETENTION_DAYS

app = Flask(__name__)


def get_db():
    """Get database connection for current request."""
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(error):
    """Close database connection at end of request."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    """Initialize the database schema."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS articles (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            link TEXT NOT NULL,
            summary TEXT,
            source TEXT,
            category TEXT,
            published TEXT,
            fetched TEXT,
            relevance_score REAL DEFAULT 0,
            is_read INTEGER DEFAULT 0,
            is_starred INTEGER DEFAULT 0,
            is_hidden INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_published ON articles(published DESC);
        CREATE INDEX IF NOT EXISTS idx_relevance ON articles(relevance_score DESC);
        CREATE INDEX IF NOT EXISTS idx_category ON articles(category);
        CREATE INDEX IF NOT EXISTS idx_is_read ON articles(is_read);
        CREATE INDEX IF NOT EXISTS idx_is_starred ON articles(is_starred);
    """
    )
    conn.commit()
    conn.close()


def calculate_relevance(title, summary):
    """
    Calculate relevance score based on keyword matches.
    Higher score = more relevant to opioid simulation research.
    """
    text = f"{title} {summary}".lower()
    score = 0

    # Check high-weight keywords (simulation/modeling terms)
    high_matches = sum(1 for kw in KEYWORDS["high"] if kw.lower() in text)

    # Check medium-weight keywords (opioid/drug terms)
    medium_matches = sum(1 for kw in KEYWORDS["medium"] if kw.lower() in text)

    # Check low-weight keywords (broader context)
    low_matches = sum(1 for kw in KEYWORDS["low"] if kw.lower() in text)

    # Scoring: best when both simulation AND opioid terms present
    if high_matches > 0 and medium_matches > 0:
        score = 100 + (high_matches * 10) + (medium_matches * 5)
    elif high_matches > 0:
        score = 50 + (high_matches * 5) + (low_matches * 2)
    elif medium_matches > 0:
        score = 30 + (medium_matches * 3) + (low_matches * 2)
    else:
        score = low_matches * 2

    return min(score, 200)  # Cap at 200


def generate_article_id(link, title):
    """Generate a unique ID for an article."""
    content = f"{link}{title}"
    return hashlib.md5(content.encode()).hexdigest()


def parse_date(date_str):
    """Try to parse various date formats."""
    if not date_str:
        return datetime.now().isoformat()

    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).isoformat()
        except ValueError:
            continue

    return datetime.now().isoformat()


def fetch_feed(feed_config):
    """Fetch and parse a single RSS/Atom feed."""
    articles = []
    try:
        feed = feedparser.parse(feed_config["url"])
        for entry in feed.entries:
            title = entry.get("title", "No title")
            link = entry.get("link", "")
            summary = entry.get("summary", entry.get("description", ""))

            # Clean HTML from summary
            summary = re.sub(r"<[^>]+>", "", summary)[:500]

            published = entry.get("published", entry.get("updated", ""))

            article = {
                "id": generate_article_id(link, title),
                "title": title,
                "link": link,
                "summary": summary,
                "source": feed_config["name"],
                "category": feed_config.get("category", "news"),
                "published": parse_date(published),
                "fetched": datetime.now().isoformat(),
                "relevance_score": calculate_relevance(title, summary),
            }
            articles.append(article)
    except Exception as e:
        print(f"Error fetching {feed_config['name']}: {e}")

    return articles


def fetch_pubmed_search(search_config):
    """Fetch results from a PubMed search via RSS."""
    base_url = "https://pubmed.ncbi.nlm.nih.gov/rss/search/"
    # PubMed RSS feeds require encoded search terms
    query = quote_plus(search_config["query"])
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={query}&retmax=20&sort=date"

    feed_config = {
        "name": f"PubMed: {search_config['name']}",
        "url": f"https://pubmed.ncbi.nlm.nih.gov/rss/search/?term={query}&limit=20",
        "category": "research",
    }

    return fetch_feed(feed_config)


def refresh_all_feeds():
    """Fetch all configured feeds and store new articles."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    new_count = 0

    # Fetch regular RSS feeds
    for feed_id, feed_config in FEEDS.items():
        articles = fetch_feed(feed_config)
        for article in articles:
            # Only insert if relevance > 0 (has some keyword match)
            if article["relevance_score"] > 0:
                try:
                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO articles
                        (id, title, link, summary, source, category, published, fetched, relevance_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            article["id"],
                            article["title"],
                            article["link"],
                            article["summary"],
                            article["source"],
                            article["category"],
                            article["published"],
                            article["fetched"],
                            article["relevance_score"],
                        ),
                    )
                    if cursor.rowcount > 0:
                        new_count += 1
                except Exception as e:
                    print(f"Error inserting article: {e}")

    # Fetch PubMed searches
    for search in PUBMED_SEARCHES:
        articles = fetch_pubmed_search(search)
        for article in articles:
            try:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO articles
                    (id, title, link, summary, source, category, published, fetched, relevance_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        article["id"],
                        article["title"],
                        article["link"],
                        article["summary"],
                        article["source"],
                        article["category"],
                        article["published"],
                        article["fetched"],
                        article["relevance_score"],
                    ),
                )
                if cursor.rowcount > 0:
                    new_count += 1
            except Exception as e:
                print(f"Error inserting article: {e}")

    # Clean up old articles
    cutoff = (datetime.now() - timedelta(days=RETENTION_DAYS)).isoformat()
    cursor.execute(
        "DELETE FROM articles WHERE published < ? AND is_starred = 0", (cutoff,)
    )

    conn.commit()
    conn.close()
    return new_count


# Flask Routes


@app.route("/")
def index():
    """Main page showing article feed."""
    return render_template("index.html")


@app.route("/api/articles")
def get_articles():
    """API endpoint to get articles with filtering."""
    db = get_db()

    category = request.args.get("category", "all")
    filter_type = request.args.get("filter", "all")  # all, unread, starred
    search = request.args.get("search", "")
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 50))

    query = "SELECT * FROM articles WHERE is_hidden = 0"
    params = []

    if category != "all":
        query += " AND category = ?"
        params.append(category)

    if filter_type == "unread":
        query += " AND is_read = 0"
    elif filter_type == "starred":
        query += " AND is_starred = 1"

    if search:
        query += " AND (title LIKE ? OR summary LIKE ?)"
        search_term = f"%{search}%"
        params.extend([search_term, search_term])

    query += " ORDER BY relevance_score DESC, published DESC"
    query += f" LIMIT {per_page} OFFSET {(page - 1) * per_page}"

    cursor = db.execute(query, params)
    articles = [dict(row) for row in cursor.fetchall()]

    # Get counts
    count_query = "SELECT COUNT(*) FROM articles WHERE is_hidden = 0"
    total = db.execute(count_query).fetchone()[0]
    unread = db.execute(count_query + " AND is_read = 0").fetchone()[0]
    starred = db.execute(count_query + " AND is_starred = 1").fetchone()[0]

    return jsonify(
        {
            "articles": articles,
            "total": total,
            "unread": unread,
            "starred": starred,
            "page": page,
            "per_page": per_page,
        }
    )


@app.route("/api/articles/important")
def get_important_articles():
    """Get high-relevance articles that shouldn't be missed."""
    db = get_db()
    cursor = db.execute(
        """
        SELECT * FROM articles
        WHERE is_hidden = 0 AND relevance_score >= 100 AND is_read = 0
        ORDER BY relevance_score DESC, published DESC
        LIMIT 10
    """
    )
    articles = [dict(row) for row in cursor.fetchall()]
    return jsonify({"articles": articles})


@app.route("/api/articles/<article_id>/read", methods=["POST"])
def mark_read(article_id):
    """Mark an article as read."""
    db = get_db()
    db.execute("UPDATE articles SET is_read = 1 WHERE id = ?", (article_id,))
    db.commit()
    return jsonify({"success": True})


@app.route("/api/articles/<article_id>/unread", methods=["POST"])
def mark_unread(article_id):
    """Mark an article as unread."""
    db = get_db()
    db.execute("UPDATE articles SET is_read = 0 WHERE id = ?", (article_id,))
    db.commit()
    return jsonify({"success": True})


@app.route("/api/articles/<article_id>/star", methods=["POST"])
def toggle_star(article_id):
    """Toggle starred status of an article."""
    db = get_db()
    db.execute(
        "UPDATE articles SET is_starred = NOT is_starred WHERE id = ?", (article_id,)
    )
    db.commit()
    cursor = db.execute("SELECT is_starred FROM articles WHERE id = ?", (article_id,))
    row = cursor.fetchone()
    return jsonify({"success": True, "is_starred": row["is_starred"] if row else False})


@app.route("/api/articles/<article_id>/hide", methods=["POST"])
def hide_article(article_id):
    """Hide an article from the feed."""
    db = get_db()
    db.execute("UPDATE articles SET is_hidden = 1 WHERE id = ?", (article_id,))
    db.commit()
    return jsonify({"success": True})


@app.route("/api/refresh", methods=["POST"])
def refresh_feeds():
    """Manually trigger a feed refresh."""
    new_count = refresh_all_feeds()
    return jsonify({"success": True, "new_articles": new_count})


@app.route("/api/stats")
def get_stats():
    """Get statistics about the article collection."""
    db = get_db()

    stats = {}

    # Total counts
    stats["total"] = db.execute(
        "SELECT COUNT(*) FROM articles WHERE is_hidden = 0"
    ).fetchone()[0]
    stats["unread"] = db.execute(
        "SELECT COUNT(*) FROM articles WHERE is_hidden = 0 AND is_read = 0"
    ).fetchone()[0]
    stats["starred"] = db.execute(
        "SELECT COUNT(*) FROM articles WHERE is_hidden = 0 AND is_starred = 1"
    ).fetchone()[0]
    stats["high_relevance"] = db.execute(
        "SELECT COUNT(*) FROM articles WHERE is_hidden = 0 AND relevance_score >= 100"
    ).fetchone()[0]

    # By category
    cursor = db.execute(
        """
        SELECT category, COUNT(*) as count
        FROM articles WHERE is_hidden = 0
        GROUP BY category
    """
    )
    stats["by_category"] = {row["category"]: row["count"] for row in cursor.fetchall()}

    # By source
    cursor = db.execute(
        """
        SELECT source, COUNT(*) as count
        FROM articles WHERE is_hidden = 0
        GROUP BY source
        ORDER BY count DESC
        LIMIT 10
    """
    )
    stats["by_source"] = {row["source"]: row["count"] for row in cursor.fetchall()}

    return jsonify(stats)


@app.route("/api/mark-all-read", methods=["POST"])
def mark_all_read():
    """Mark all visible articles as read."""
    db = get_db()
    category = request.json.get("category", "all") if request.json else "all"

    if category == "all":
        db.execute("UPDATE articles SET is_read = 1 WHERE is_hidden = 0")
    else:
        db.execute(
            "UPDATE articles SET is_read = 1 WHERE is_hidden = 0 AND category = ?",
            (category,),
        )

    db.commit()
    return jsonify({"success": True})


if __name__ == "__main__":
    init_db()
    print("Fetching initial feeds...")
    new_articles = refresh_all_feeds()
    print(f"Found {new_articles} new relevant articles")
    print("Starting web server at http://localhost:5000")
    app.run(debug=True, port=5000)
