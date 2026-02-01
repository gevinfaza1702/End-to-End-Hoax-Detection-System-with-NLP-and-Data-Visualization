import logging
import os
from typing import Optional, Dict, Any

try:
    import requests
except ImportError:
    requests = None  # type: ignore

logger = logging.getLogger(__name__)

class FactChecker:
    """
    Interface to Google's Fact Check Tools API to verify claims found in
    social media posts.

    The API endpoint used is `/v1alpha1/claims:search`.
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
