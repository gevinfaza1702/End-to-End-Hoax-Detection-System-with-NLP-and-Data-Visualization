from typing import List

DEFAULT_KEYWORDS: List[str] = [
    "vaksin", "pemilu", "konflik", "Israel", "Palestina", "covid",
    "konspirasi", "hoaks", "buzzer"
]

# Default model for the classifier
# Using a zero-shot model that supports Indonesian for better accuracy without fine-tuning
DEFAULT_MODEL: str = "joeddav/xlm-roberta-large-xnli"

# Database URL
DEFAULT_DB_URL: str = "sqlite:///data.db"

# Schedule time (Asia/Jakarta)
DEFAULT_SCHEDULE_TIME: str = "02:00"
