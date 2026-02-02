import logging
import os
from typing import Optional, Dict, Any

try:
    import requests
except ImportError:
    requests = None  # type: ignore

try:
    from thefuzz import fuzz
except ImportError:
    fuzz = None

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

    def search_claim(self, text: str, max_age_days: int = 1000, similarity_threshold: int = 50) -> Optional[Dict[str, Any]]:
        """
        Search for fact‑checked claims matching the given text.

        Returns a dictionary containing the first matching claim review (if
        any), or None if no match or if API key is absent.

        Uses Fuzzy Matching to ensure the returned claim is actually relevant
        to the input text.
        """
        if not self.api_key:
            logger.warning("Skipping fact check: No API Key provided.")
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

        # Iterate through claims to find the best match
        best_match = None
        best_score = 0

        for claim in claims:
            # Calculate similarity between query text and claim text
            claim_text = claim.get("text", "")
            # If claim text is empty, try using the title of the first review
            reviews = claim.get("claimReview", [])
            if not claim_text and reviews:
                claim_text = reviews[0].get("title", "")

            if not claim_text:
                continue

            if fuzz:
                score = fuzz.token_set_ratio(text, claim_text)
            else:
                score = 100 # Fallback if thefuzz is not installed

            if score > best_score and score >= similarity_threshold:
                best_score = score
                # Select the first review for this claim
                if reviews:
                     review = reviews[0]
                     best_match = {
                        "url": review.get("url"),
                        "title": review.get("title"),
                        "textual_rating": review.get("textualRating"),
                        "publisher": review.get("publisher", {}).get("name"),
                        "review_date": review.get("reviewDate"),
                        "similarity_score": score
                    }

        if best_match:
            logger.info(f"Fact Check Match Found! Score: {best_match['similarity_score']} - Title: {best_match['title']}")
            return best_match

        return None
