from __future__ import annotations
import dataclasses
import datetime as dt
from dataclasses import dataclass
from typing import Optional

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
