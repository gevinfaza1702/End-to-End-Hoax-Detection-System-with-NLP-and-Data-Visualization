from __future__ import annotations

import datetime as dt
import logging
import os
import time
import random
from typing import Iterable, List, Optional, Callable, Any

# Third‑party imports
try:
    import snscrape.modules.twitter as sntwitter  # type: ignore
except ImportError:
    sntwitter = None  # type: ignore

try:
    import praw  # type: ignore
except ImportError:
    praw = None  # type: ignore

try:
    from gnews import GNews  # type: ignore
except ImportError:
    GNews = None  # type: ignore

from structures import Post

logger = logging.getLogger(__name__)

def random_sleep(min_seconds: float = 1.0, max_seconds: float = 3.0) -> None:
    """Sleep for a random amount of time to avoid rate limiting."""
    sleep_time = random.uniform(min_seconds, max_seconds)
    time.sleep(sleep_time)

def retry_request(max_retries: int = 3, delay: float = 2.0) -> Callable:
    """Decorator to retry a function call upon exception."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    logger.warning(f"Error in {func.__name__}: {e}. Retrying ({retries}/{max_retries})...")
                    time.sleep(delay)
            logger.error(f"Failed {func.__name__} after {max_retries} retries.")
            return [] # Return empty list on failure for fetch methods
        return wrapper
    return decorator


class TwitterScraper:
    """
    Scrape tweets matching a list of keywords using snscrape.
    """

    def __init__(self, keywords: Iterable[str], max_tweets_per_keyword: int = 100) -> None:
        if sntwitter is None:
            raise ImportError("snscrape is required for TwitterScraper but it's not installed.")
        self.keywords = list(keywords)
        self.max_tweets = max_tweets_per_keyword

    def fetch(self) -> List[Post]:
        """Fetch tweets for all configured keywords."""
        posts: List[Post] = []
        for keyword in self.keywords:
            logger.info(f"Scraping Twitter for keyword: {keyword}")
            query = f"{keyword} lang:id"
            try:
                # Add retry logic manually or via helper if needed.
                # snscrape generator is hard to retry cleanly with a simple decorator.
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
                    # Anti-blocking sleep every 10 tweets
                    if i % 10 == 0:
                        random_sleep(0.5, 1.5)

            except Exception as e:
                logger.error(f"Error scraping Twitter for {keyword}: {e}")

            random_sleep(2.0, 5.0) # Sleep between keywords

        return posts


class RedditScraper:
    """
    Scrape Reddit submissions matching a set of keywords using PRAW.
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
                "Reddit credentials must be provided via parameters or environment variables."
            )
        self.reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent
        )

    def fetch(self) -> List[Post]:
        """Fetch Reddit submissions for all configured keywords."""
        posts: List[Post] = []
        for keyword in self.keywords:
            logger.info(f"Scraping Reddit for keyword: {keyword}")
            try:
                # PRAW handles rate limiting internally, but we can add retries
                self._fetch_reddit_keyword(keyword, posts)
            except Exception as e:
                logger.error(f"Error scraping Reddit for {keyword}: {e}")

            random_sleep(1.0, 3.0) # Sleep between keywords
        return posts

    @retry_request(max_retries=3, delay=5.0)
    def _fetch_reddit_keyword(self, keyword: str, posts_list: List[Post]) -> None:
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
            posts_list.append(post)


class GoogleNewsScraper:
    """
    Scrapes news articles from Google News using the gnews package.
    """

    def __init__(self,
                 keywords: Iterable[str],
                 language: str = "id",
                 country: str = "ID",
                 period: Optional[str] = None,
                 max_results: int = 10) -> None:
        if GNews is None:
            raise ImportError("The gnews package must be installed to use GoogleNewsScraper. Run `pip install gnews`. ")
        self.keywords = list(keywords)
        self.language = language
        self.country = country
        self.period = period
        self.max_results = max_results
        self.client = GNews(language=self.language, country=self.country)
        if self.period:
            self.client.period = self.period
        self.client.max_results = self.max_results

    def fetch(self) -> List[Post]:
        """Fetch news articles from Google News for all configured keywords."""
        posts: List[Post] = []
        for keyword in self.keywords:
            logger.info(f"Scraping Google News for keyword: {keyword}")
            try:
                self._fetch_google_keyword(keyword, posts)
            except Exception as exc:
                logger.error(f"Error fetching Google News for {keyword}: {exc}")
                continue

            random_sleep(1.0, 3.0) # Sleep between keywords
        return posts

    @retry_request(max_retries=3, delay=2.0)
    def _fetch_google_keyword(self, keyword: str, posts_list: List[Post]) -> None:
        results = self.client.get_news(keyword)  # type: ignore
        if not results:
             return

        for item in results:
            title = item.get('title', '') or ''
            description = item.get('description', '') or ''
            content = f"{title}\n\n{description}".strip()
            url = item.get('url', '')
            published_str = item.get('published date') or item.get('published_date')
            created_at: dt.datetime
            if published_str:
                try:
                    created_at = dt.datetime.strptime(published_str, '%a, %d %b %Y %H:%M:%S %Z')
                except Exception:
                    try:
                        created_at = dt.datetime.strptime(published_str, '%Y-%m-%dT%H:%M:%SZ')
                    except Exception:
                        created_at = dt.datetime.utcnow()
            else:
                created_at = dt.datetime.utcnow()

            publisher = item.get("publisher")
            if isinstance(publisher, dict):
                author = publisher.get("title")
            elif isinstance(publisher, str):
                author = publisher
            else:
                author = None

            post = Post(
                platform="google",
                keyword=keyword,
                content=content,
                url=url,
                created_at=created_at,
                author=author
            )
            posts_list.append(post)
