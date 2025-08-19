"""
news_scraper.py
================

Provides a scraper for collecting news articles from Google News using the
`gnews` package.  The scraper wraps `gnews.GNews` and produces a list of
`Post` objects compatible with the pipeline defined in `social_media_agent.py`.

Usage example::

    from news_scraper import GoogleNewsScraper
    scraper = GoogleNewsScraper(keywords=["hoaks", "vaksin"], max_results=10)
    posts = scraper.fetch()
    for post in posts:
        print(post.title, post.created_at)

Note: the gnews package retrieves data from Google News RSS feeds.  The
publisher field may be present in the response but is not used here.

"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Iterable, List, Optional

try:
    from gnews import GNews  # type: ignore
except ImportError:
    GNews = None  # type: ignore

try:
    from social_media_agent import Post
except Exception:
    # Define a lightweight Post dataclass fallback to avoid circular imports
    from dataclasses import dataclass

    @dataclass
    class Post:
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

logger = logging.getLogger(__name__)


class GoogleNewsScraper:
    """
    Scrapes news articles from Google News using the gnews package.

    Parameters
    ----------
    keywords: iterable of str
        List of keywords to search for.
    language: str, optional
        Two‑letter language code (default `'id'` for Indonesian).  See
        https://pypi.org/project/gnews/ for supported languages.
    country: str, optional
        Two‑letter country code (default `'ID'` for Indonesia).
    period: str, optional
        Restrict articles to a time window, e.g. `'7d'` for past seven days.
    max_results: int, optional
        Maximum number of articles per keyword (default 10).
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
        # Configure GNews client
        self.client = GNews(language=self.language, country=self.country)
        if self.period:
            # Set the period on the client.  GNews uses `period` property to
            # filter by timeframe (e.g. '7d').  See library docs for details.
            self.client.period = self.period
        self.client.max_results = self.max_results

    def fetch(self) -> List[Post]:
        """
        Fetch news articles from Google News for all configured keywords.

        Returns
        -------
        List[Post]
            A list of Post objects containing the combined title and
            description of each article, along with its URL and publish date.
        """
        posts: List[Post] = []
        for keyword in self.keywords:
            logger.info(f"Scraping Google News for keyword: {keyword}")
            try:
                results = self.client.get_news(keyword)  # type: ignore
            except Exception as exc:
                logger.error(f"Error fetching Google News for {keyword}: {exc}")
                continue
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

    # 🔧 Amanin bagian publisher
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
                posts.append(post)
        return posts
