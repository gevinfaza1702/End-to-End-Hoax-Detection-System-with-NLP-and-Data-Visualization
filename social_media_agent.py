"""
social_media_agent.py
======================

This module provides a high‑level implementation of an intelligent agent that
collects posts from Twitter, Reddit, and Google News, classifies their content
as hoax or non‑hoax using natural language processing (NLP), optionally verifies
claims against Google's Fact Check Tools API, stores results in a database.

Refactored to be modular and cleaner.
"""

from __future__ import annotations

import logging
import time
from typing import Iterable, List

from dotenv import load_dotenv

# Local imports
import config
from structures import Post
from database import Database
from classifier import NewsClassifier
from fact_checker import FactChecker
from scrapers import TwitterScraper, RedditScraper, GoogleNewsScraper

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class Scheduler:
    """
    Orchestrate periodic scraping, classification and fact checking.
    """

    def __init__(self,
                 keywords: Iterable[str] = config.DEFAULT_KEYWORDS,
                 sources: Iterable[str] | str = ("google",),
                 twitter_max: int = 50,
                 reddit_max: int = 50,
                 google_max: int = 10,
                 db_url: str = config.DEFAULT_DB_URL,
                 model_name: str = config.DEFAULT_MODEL,
                 fact_check: bool = False) -> None:
        """
        Initialise the scheduler.
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
                self.google_scraper = GoogleNewsScraper(self.keywords, max_results=google_max)
            except Exception as exc:
                logger.warning(f"Google News scraper initialisation failed: {exc}")

        # Initialise classifier and optional fact checker
        self.classifier = NewsClassifier(model_name=model_name)
        self.fact_checker = FactChecker() if fact_check else None
        self.db = Database(db_url=db_url)

    def extract_claim_keywords(self, text: str) -> str:
        import re
        # Use keywords from config plus some claim-specific terms
        base_keywords = config.DEFAULT_KEYWORDS
        extra_keywords = ["chip", "autisme", "kecurangan"]
        keywords = list(set(base_keywords + extra_keywords))

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

    def start(self, schedule_time: str = config.DEFAULT_SCHEDULE_TIME) -> None:
        """
        Start the scheduler which runs `run_job` every day at `schedule_time`.
        """
        try:
            import schedule  # type: ignore
        except ImportError:
            raise ImportError("schedule package is required for the Scheduler")

        logger.info(f"Scheduling job daily at {schedule_time} (Asia/Jakarta)")
        schedule.every().day.at(schedule_time).do(self.run_job)
        # Run once at startup
        self.run_job()
        while True:
            schedule.run_pending()
            time.sleep(60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the social media agent")
    parser.add_argument("--once", action="store_true",
                        help="Run scraping/classification once and exit")
    parser.add_argument("--daily", action="store_true",
                        help="Run scraping/classification every day at a scheduled time (default 02:00)")
    parser.add_argument("--time", default=config.DEFAULT_SCHEDULE_TIME,
                        help="Schedule time for daily run (HH:MM, 24h) in Asia/Jakarta timezone")
    parser.add_argument("--db", default=config.DEFAULT_DB_URL, help="Database URL")
    parser.add_argument("--model", default=config.DEFAULT_MODEL, help="HuggingFace model name to use for classification")
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

    agent = Scheduler(keywords=config.DEFAULT_KEYWORDS,
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
