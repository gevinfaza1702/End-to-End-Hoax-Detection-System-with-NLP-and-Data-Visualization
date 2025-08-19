"""
social_media_agent.py
======================

This module provides a high‑level implementation of an intelligent agent that
collects posts from Twitter and Reddit, classifies their content as hoax or
non‑hoax using natural language processing (NLP), optionally verifies claims
against Google's Fact Check Tools API, stores results in a database and
exposes a simple Streamlit dashboard for visualising the collected data.

The code is organised into several classes to keep responsibilities
separated:

* **TwitterScraper** – fetches tweets matching a list of keywords using
  `snscrape`.
* **RedditScraper** – fetches Reddit submissions matching the keywords using
  `praw`.  Credentials for Reddit API should be provided via environment
  variables or passed explicitly when instantiating the scraper.
* **NewsClassifier** – wraps a HuggingFace transformer model and exposes a
  `classify` method which assigns each piece of text a label (`hoax` or
  `not_hoax`) along with a confidence score.
* **FactChecker** – optionally queries the Google Fact Check Tools API to
  search for fact‑checked claims.  The `search_claim` method returns a
  dictionary with information about the fact check if available.
* **Database** – uses SQLAlchemy to define a simple schema and persist
  scraped posts along with classification and fact check metadata.
* **Scheduler** – orchestrates periodic execution of scraping, classification
  and fact checking tasks.  A simple daily schedule is implemented using
  the `schedule` package.
* **Streamlit dashboard** – defined in `dashboard.py` (see below) and
  provides an interactive web interface for exploring the data stored in
  the database.

This script is intended as a starting point rather than a production‑ready
system.  Before running it in production you should consider adding robust
error handling, authentication/authorisation, persistence and rate‑limiting
to comply with the terms of service for the underlying data sources.

"""

from __future__ import annotations

import dataclasses
import datetime as dt
import logging
import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Dict, Any

from dotenv import load_dotenv
load_dotenv()


# Third‑party imports – these modules need to be installed in the
# environment where this script runs.  They are commented here so that
# the code will still import cleanly in environments where the packages
# are not available (e.g. during static analysis or documentation build).
try:
    import snscrape.modules.twitter as sntwitter  # type: ignore
except ImportError:
    sntwitter = None  # type: ignore

try:
    import praw  # type: ignore
except ImportError:
    praw = None  # type: ignore

try:
    from transformers import pipeline  # type: ignore
except ImportError:
    pipeline = None  # type: ignore

try:
    import requests
except ImportError:
    requests = None  # type: ignore

try:
    from sqlalchemy import (create_engine, Column, Integer, String, Text,
                            Float, Boolean, DateTime)
    from sqlalchemy.orm import declarative_base, sessionmaker
except ImportError:
    # If SQLAlchemy isn't installed the Database class will not work.
    create_engine = None  # type: ignore
    Column = Integer = String = Text = Float = Boolean = DateTime = None  # type: ignore
    declarative_base = None  # type: ignore
    sessionmaker = None  # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


###############################################################################
# Configuration
###############################################################################

DEFAULT_KEYWORDS: List[str] = [
    "vaksin", "pemilu", "konflik", "Israel", "Palestina", "covid",
    "konspirasi", "hoaks", "buzzer"
]


###############################################################################
# Data structures
###############################################################################

@dataclass
class Post:
    """Represents a social media post scraped from Twitter or Reddit."""

    platform: str
    keyword: str
    content: str
    url: str
    created_at: dt.datetime
    author: Optional[str] = None
    predicted_label: Optional[str] = None
    prediction_score: Optional[float] = None
    fact_check_url: Optional[str] = None
    fact_check_rating: Optional[str] = None
    fact_check_publisher: Optional[str] = None


###############################################################################
# Scrapers
###############################################################################

class TwitterScraper:
    """
    Scrape tweets matching a list of keywords using snscrape.

    snscrape is a popular open‑source tool for scraping Twitter data without
    using the Twitter API.  It scrapes publicly available tweets and
    therefore does not require API keys.  See the project's documentation for
    details on usage.

    Note: When scraping large volumes of data you should consider Twitter's
    terms of service and ensure your usage complies with them.
    """

    def __init__(self, keywords: Iterable[str], max_tweets_per_keyword: int = 100) -> None:
        if sntwitter is None:
            raise ImportError("snscrape is required for TwitterScraper but it's not installed.")
        self.keywords = list(keywords)
        self.max_tweets = max_tweets_per_keyword

    def fetch(self) -> List[Post]:
        """Fetch tweets for all configured keywords.

        Returns a list of Post objects.
        """
        posts: List[Post] = []
        for keyword in self.keywords:
            logger.info(f"Scraping Twitter for keyword: {keyword}")
            # Use snscrape to search for tweets containing the keyword
            query = f"{keyword} lang:id"  # restrict to Indonesian language
            scraper = sntwitter.TwitterSearchScraper(query)
            for i, tweet in enumerate(scraper.get_items()):
                if i >= self.max_tweets:
                    break
                post = Post(
                    platform="twitter",
                    keyword=keyword,
                    content=tweet.content,
                    url=f"https://twitter.com/{tweet.user.username}/status/{tweet.id}",
                    created_at=tweet.date,
                    author=tweet.user.username
                )
                posts.append(post)
        return posts


class RedditScraper:
    """
    Scrape Reddit submissions matching a set of keywords using PRAW.

    PRAW requires a registered Reddit application and therefore needs
    credentials: client_id, client_secret and user_agent.  You can pass
    these via environment variables (REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT) or as arguments to the constructor.
    """

    def __init__(self, keywords: Iterable[str], client_id: Optional[str] = None,
                 client_secret: Optional[str] = None, user_agent: Optional[str] = None,
                 max_posts_per_keyword: int = 100) -> None:
        if praw is None:
            raise ImportError("praw is required for RedditScraper but it's not installed.")
        self.keywords = list(keywords)
        self.max_posts = max_posts_per_keyword
        self.client_id = client_id or os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET")
        self.user_agent = user_agent or os.getenv("REDDIT_USER_AGENT", "social_media_agent")
        if not all([self.client_id, self.client_secret, self.user_agent]):
            raise ValueError(
                "Reddit credentials must be provided via parameters or environment variables."  # noqa: E501
            )
        self.reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent
        )

    def fetch(self) -> List[Post]:
        """Fetch Reddit submissions for all configured keywords.

        Returns a list of Post objects.
        """
        posts: List[Post] = []
        for keyword in self.keywords:
            logger.info(f"Scraping Reddit for keyword: {keyword}")
            # search all of Reddit (subreddit='all') and sort by new
            submissions = self.reddit.subreddit("all").search(keyword, sort="new", limit=self.max_posts)
            for submission in submissions:
                post = Post(
                    platform="reddit",
                    keyword=keyword,
                    content=submission.title + "\n\n" + (submission.selftext or ""),
                    url=submission.url,
                    created_at=dt.datetime.fromtimestamp(submission.created_utc),
                    author=submission.author.name if submission.author else None
                )
                posts.append(post)
        return posts


###############################################################################
# Classification
###############################################################################

class NewsClassifier:
    """
    Classify text into `hoax` or `not_hoax` using a HuggingFace model.

    The classifier uses a sequence classification pipeline.  You can specify
    which underlying model to load (default is `bert-base-uncased`).  If you
    have a fine‑tuned model saved locally or on HuggingFace Hub, pass its
    identifier to `model_name`.

    The pipeline returns a dictionary with a label and a score.  We normalise
    the label names to `hoax` and `not_hoax` by mapping model output labels
    (e.g. `LABEL_0`, `LABEL_1`) to our semantic names.  You should fine‑tune
    the model yourself on a hoax detection dataset for best results.
    """

    def __init__(self, model_name: str = "bert-base-uncased") -> None:
        if pipeline is None:
            raise ImportError("transformers is required for NewsClassifier but it's not installed.")
        logger.info(f"Loading NLP model: {model_name}")
        # The pipeline will download the model if not present locally.  This may
        # take a while on the first run and require internet access.
        self.pipeline = pipeline(
            "text-classification",
            model=model_name,
            return_all_scores=True,
            truncation=True
        )
        # When using off‑the‑shelf models, you need to know how labels are
        # mapped.  For demonstration we assume a binary classifier with
        # `LABEL_0` = not hoax and `LABEL_1` = hoax.
        self.label_map = {
            "LABEL_0": "not_hoax",
            "LABEL_1": "hoax",
        }

    def classify(self, text: str) -> Dict[str, Any]:
        """Classify a piece of text and return label and score."""
        if not text.strip():
            return {"label": None, "score": None}
        result = self.pipeline(text)[0]
        best = max(result, key=lambda x: x['score'])
        label = self.label_map.get(best['label'], best['label'])

        # ✅ Apply threshold adjustment to reduce false positives
        if label == "hoax" and best["score"] < 0.65:
            label = "not_hoax"

        return {"label": label, "score": best['score']}



###############################################################################
# Fact checking
###############################################################################

class FactChecker:
    """
    Interface to Google's Fact Check Tools API to verify claims found in
    social media posts.

    The API endpoint used is `/v1alpha1/claims:search`, which allows searching
    for fact‑checked claims via a textual query.  According to the
    documentation, the request uses the GET method and accepts parameters
    such as `query`, `languageCode`, `maxAgeDays`, `pageSize` etc.  The
    response contains a list of claims with associated claim reviews,
    including review URLs, titles and textual ratings【853238102903704†L85-L145】【763788042889412†L84-L178】.

    To use this class you must supply a valid API key.  Create an API key
    through Google Cloud Console and enable the Fact Check Tools API on
    your project.  Then either pass the API key directly when creating
    the FactChecker or set the `FACT_CHECK_API_KEY` environment variable.
    """

    BASE_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    def __init__(self, api_key: Optional[str] = None, language_code: str = "id") -> None:
        if requests is None:
            raise ImportError("requests is required for FactChecker but it's not installed.")
        self.api_key = api_key or os.getenv("FACT_CHECK_API_KEY")
        if not self.api_key:
            logger.warning("No API key provided for FactChecker. Fact checking will be disabled.")
        self.language_code = language_code

    def search_claim(self, text: str, max_age_days: int = 1000) -> Optional[Dict[str, Any]]:
        """
        Search for fact‑checked claims matching the given text.

        Returns a dictionary containing the first matching claim review (if
        any), or None if no match or if API key is absent.
        """
        if not self.api_key:
            return None
        params = {
            "query": text,
            "languageCode": self.language_code,
            "maxAgeDays": max_age_days,
            "pageSize": 10,
            "key": self.api_key,
        }
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
        except Exception as exc:
            logger.error(f"Fact check API request failed: {exc}")
            return None
        data = response.json()
        claims = data.get("claims", [])
        if not claims:
            return None
        claim = claims[0]
        # Each claim may have one or more claim reviews
        reviews = claim.get("claimReview", [])
        if not reviews:
            return None
        review = reviews[0]
        return {
            "url": review.get("url"),
            "title": review.get("title"),
            "textual_rating": review.get("textualRating"),
            "publisher": review.get("publisher", {}).get("name"),
            "review_date": review.get("reviewDate"),
        }


###############################################################################
# Database
###############################################################################

Base = declarative_base() if declarative_base is not None else None

class PostModel(Base):  # type: ignore
    """
    SQLAlchemy model representing a social media post.  Fields mirror the
    attributes of the Post dataclass but are persisted in a relational
    database.
    """

    __tablename__ = "posts"
    id = Column(Integer, primary_key=True)
    platform = Column(String, nullable=False)
    keyword = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    url = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)
    author = Column(String, nullable=True)
    predicted_label = Column(String, nullable=True)
    prediction_score = Column(Float, nullable=True)
    fact_check_url = Column(String, nullable=True)
    fact_check_rating = Column(String, nullable=True)
    fact_check_publisher = Column(String, nullable=True)
    inserted_at = Column(DateTime, nullable=False, default=dt.datetime.utcnow)


class Database:
    """
    Handle persistence of posts using SQLAlchemy.

    Supports both SQLite (default) and PostgreSQL (or any other database
    supported by SQLAlchemy).  The database URL should be provided in the
    form accepted by SQLAlchemy, e.g. `sqlite:///data.db` or
    `postgresql+psycopg2://user:password@localhost/dbname`.
    """

    def __init__(self, db_url: str = "sqlite:///data.db") -> None:
        if create_engine is None:
            raise ImportError("SQLAlchemy is required for Database but it's not installed.")
        self.engine = create_engine(db_url, echo=False, future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def insert_posts(self, posts: Iterable[Post]) -> None:
        """
        Insert a list of Post objects into the database.  Existing rows with
        the same URL will be ignored to avoid duplicates.
        """
        session = self.Session()
        try:
            for post in posts:
                # Check for duplicates
                exists = session.query(PostModel).filter_by(url=post.url).first()
                if exists:
                    exists.predicted_label = post.predicted_label
                    exists.prediction_score = post.prediction_score
                    exists.fact_check_url = post.fact_check_url
                    exists.fact_check_rating = post.fact_check_rating
                    exists.fact_check_publisher = post.fact_check_publisher
                    continue
                model = PostModel(
                    platform=post.platform,
                    keyword=post.keyword,
                    content=post.content,
                    url=post.url,
                    created_at=post.created_at,
                    author=post.author,
                    predicted_label=post.predicted_label,
                    prediction_score=post.prediction_score,
                    fact_check_url=post.fact_check_url,
                    fact_check_rating=post.fact_check_rating,
                    fact_check_publisher=post.fact_check_publisher,
                    inserted_at=dt.datetime.utcnow(),
                )
                session.add(model)
            session.commit()
        except Exception:
            session.rollback()
            logger.exception("Failed to insert posts into database")
        finally:
            session.close()

    def get_posts(self, limit: int = 1000) -> List[PostModel]:
        """
        Return the most recent posts from the database up to the specified limit.
        """
        session = self.Session()
        try:
            posts = session.query(PostModel).order_by(PostModel.inserted_at.desc()).limit(limit).all()
            return posts
        finally:
            session.close()


###############################################################################
# Scheduler
###############################################################################

class Scheduler:
    """
    Orchestrate periodic scraping, classification and fact checking.

    Uses the `schedule` library to run tasks every day at a specified time.
    When executed as a script (`python social_media_agent.py`) the scheduler
    starts immediately and runs indefinitely.  To integrate into another
    application you can instantiate the Scheduler and call its methods
    manually or schedule them using your own job scheduler.
    """

    def __init__(self,
                 keywords: Iterable[str] = DEFAULT_KEYWORDS,
                 sources: Iterable[str] | str = ("google",),
                 twitter_max: int = 50,
                 reddit_max: int = 50,
                 google_max: int = 10,
                 db_url: str = "sqlite:///data.db",
                 model_name: str = "bert-base-uncased",
                 fact_check: bool = False) -> None:
        """
        Initialise the scheduler.

        Parameters
        ----------
        keywords : iterable of str
            Kata kunci yang akan digunakan untuk scraping.
        sources : iterable atau string
            Sumber data yang diaktifkan.  Bisa berisi kombinasi 'google',
            'twitter', dan 'reddit'.  Jika string tunggal diberikan,
            otomatis dikonversi ke list.
        twitter_max : int
            Jumlah maksimum tweet per kata kunci.
        reddit_max : int
            Jumlah maksimum kiriman Reddit per kata kunci.
        google_max : int
            Jumlah maksimum artikel Google News per kata kunci.
        db_url : str
            URL database SQLAlchemy.
        model_name : str
            Nama model HuggingFace untuk klasifikasi.
        fact_check : bool
            Aktifkan integrasi Fact Check Tools API.
        """
        self.keywords = list(keywords)
        # Normalise sources into a list
        if isinstance(sources, str):
            sources_list = [sources.lower()]
        else:
            sources_list = [s.lower() for s in sources]
        self.sources = sources_list
        # Initialise scrapers conditionally
        self.twitter_scraper = None
        self.reddit_scraper = None
        self.google_scraper = None
        if "twitter" in self.sources:
            try:
                self.twitter_scraper = TwitterScraper(self.keywords, max_tweets_per_keyword=twitter_max)
            except Exception as exc:
                logger.warning(f"Twitter scraper initialisation failed: {exc}")
        if "reddit" in self.sources:
            try:
                self.reddit_scraper = RedditScraper(self.keywords, max_posts_per_keyword=reddit_max)
            except Exception as exc:
                logger.warning(f"Reddit scraper initialisation failed: {exc}")
        if "google" in self.sources:
            try:
                from news_scraper import GoogleNewsScraper  # local import to avoid circular dependency
                self.google_scraper = GoogleNewsScraper(self.keywords, max_results=google_max)
            except Exception as exc:
                logger.warning(f"Google News scraper initialisation failed: {exc}")
        # Initialise classifier and optional fact checker
        self.classifier = NewsClassifier(model_name=model_name)
        self.fact_checker = FactChecker() if fact_check else None
        self.db = Database(db_url=db_url)

    def extract_claim_keywords(self, text: str) -> str:
        import re
        keywords = [
            "vaksin", "covid", "chip", "autisme",
            "pemilu", "kecurangan", "konspirasi", "hoaks",
            "buzzer", "Israel", "Palestina"
        ]
        # Ambil keyword yang muncul di dalam teks
        found = [kw for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE)]
        return " ".join(found)

    def run_job(self) -> None:

        logger.info("Starting scheduled job: scrape, classify, fact check")
        posts: List[Post] = []

    # Step 1: Scraping
        if self.google_scraper is not None:
            try:
                posts.extend(self.google_scraper.fetch())
            except Exception as exc:
                logger.error(f"Google News scraping failed: {exc}")

            if self.twitter_scraper is not None:
                try:
                    posts.extend(self.twitter_scraper.fetch())
                except Exception as exc:
                    logger.error(f"Twitter scraping failed: {exc}")

        if self.reddit_scraper is not None:
            try:
                posts.extend(self.reddit_scraper.fetch())
            except Exception as exc:
                logger.error(f"Reddit scraping failed: {exc}")

        logger.info(f"Fetched {len(posts)} posts")

    # Step 2: Klasifikasi + Fact Checking
        for post in posts:
            result = self.classifier.classify(post.content)
            post.predicted_label = result.get("label")
            post.prediction_score = result.get("score")
            logger.info(f"🔍 Label: {post.predicted_label} | Score: {post.prediction_score:.2f}")

            if self.fact_checker is not None and post.predicted_label == "hoax":
            # 💡 Optimasi: gunakan keyword pendek untuk fact check
                claim_query = self.extract_claim_keywords(post.content).strip()
                if claim_query:
                    claim_query += " " + post.keyword  # tambahkan keyword dari scraping
                else:
                    claim_query = post.keyword


                    # 🔁 Fallback pakai keyword asli dari scraping
                    claim_query = post.keyword

                logger.info(f"🔎 Fact-checking with query: {claim_query}")

                fc_result = self.fact_checker.search_claim(claim_query)
                if fc_result:
                    logger.info(f"✅ Found fact-check: {fc_result.get('title')} ({fc_result.get('url')})")
                    post.fact_check_url = fc_result.get("url")
                    post.fact_check_rating = fc_result.get("textual_rating")
                    post.fact_check_publisher = fc_result.get("publisher")
                else:
                    logger.info(f"❗ No fact-check found for: {claim_query}")


        # Step 3: Simpan ke database
        self.db.insert_posts(posts)
        logger.info("✅ Job completed")

        # Statistik ringkas
        total_hoax = sum(1 for p in posts if p.predicted_label == "hoax")
        fact_checked = sum(1 for p in posts if p.predicted_label == "hoax" and p.fact_check_url)
        not_found = total_hoax - fact_checked

        logger.info(f"📊 {total_hoax} prediksi hoax, {fact_checked} ditemukan faktanya, {not_found} tidak ditemukan")



    def start(self, schedule_time: str = "02:00") -> None:
        """
        Start the scheduler which runs `run_job` every day at `schedule_time`.

        `schedule_time` should be a string in 24‑hour HH:MM format (Asia/Jakarta
        timezone assumed).  The method will block indefinitely.
        """
        try:
            import schedule  # type: ignore
        except ImportError:
            raise ImportError("schedule package is required for the Scheduler")
        # Convert schedule_time to 24‑hour time
        logger.info(f"Scheduling job daily at {schedule_time} (Asia/Jakarta)")
        schedule.every().day.at(schedule_time).do(self.run_job)
        # Run once at startup
        self.run_job()
        while True:
            schedule.run_pending()
            time.sleep(60)


###############################################################################
# Streamlit Dashboard
###############################################################################

# The dashboard is defined in a separate file to keep the core agent code
# independent of the web interface.  See `dashboard.py` in the repository
# (provided below) for the implementation.  The dashboard uses Streamlit to
# display posts from the database with filters and summary statistics.


###############################################################################
# Main entry point
###############################################################################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the social media agent")
    parser.add_argument("--once", action="store_true",
                        help="Run scraping/classification once and exit")
    parser.add_argument("--daily", action="store_true",
                        help="Run scraping/classification every day at a scheduled time (default 02:00)")
    parser.add_argument("--time", default="02:00",
                        help="Schedule time for daily run (HH:MM, 24h) in Asia/Jakarta timezone")
    parser.add_argument("--db", default="sqlite:///data.db", help="Database URL")
    parser.add_argument("--model", default="bert-base-uncased", help="HuggingFace model name to use for classification")
    parser.add_argument("--fact-check", action="store_true", help="Enable fact checking via Google Fact Check Tools API")
    parser.add_argument("--source", default="google", choices=["google", "twitter", "reddit", "all", "social"],
                        help="Select data source: google (Google News), twitter, reddit, social (twitter+reddit), or all (google+twitter+reddit)")
    parser.add_argument("--twitter-max", type=int, default=50, help="Maximum tweets per keyword")
    parser.add_argument("--reddit-max", type=int, default=50, help="Maximum Reddit posts per keyword")
    parser.add_argument("--google-max", type=int, default=10, help="Maximum Google News articles per keyword")
    args = parser.parse_args()
    # Determine sources list
    if args.source == "social":
        sources = ["twitter", "reddit"]
    elif args.source == "all":
        sources = ["google", "twitter", "reddit"]
    else:
        sources = [args.source]
    agent = Scheduler(keywords=DEFAULT_KEYWORDS,
                      sources=sources,
                      twitter_max=args.twitter_max,
                      reddit_max=args.reddit_max,
                      google_max=args.google_max,
                      db_url=args.db,
                      model_name=args.model,
                      fact_check=args.fact_check)
    if args.once:
        agent.run_job()
    elif args.daily:
        agent.start(schedule_time=args.time)
    else:
        parser.print_help()