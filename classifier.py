import logging
from typing import Dict, Any, List, Optional

try:
    from transformers import pipeline  # type: ignore
except ImportError:
    pipeline = None  # type: ignore

logger = logging.getLogger(__name__)

class NewsClassifier:
    """
    Classify text into `hoax` or `not_hoax` using a HuggingFace model.

    Supports two modes:
    1. Zero-Shot Classification (default): Uses a model trained on NLI (Natural Language Inference)
       to classify text against candidate labels ["hoaks", "fakta"] without specific fine-tuning.
       Default model: `joeddav/xlm-roberta-large-xnli` (multilingual).

    2. Text Classification: Uses a standard fine-tuned model for binary classification.
       Used if a non-NLI model is provided.
    """

    DEFAULT_MODEL = "joeddav/xlm-roberta-large-xnli"
    CANDIDATE_LABELS = ["hoaks", "fakta"]

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        if pipeline is None:
            raise ImportError("transformers is required for NewsClassifier but it's not installed.")

        self.model_name = model_name
        self.is_zero_shot = "xnli" in model_name or "mnli" in model_name or "zero-shot" in model_name

        logger.info(f"Loading NLP model: {model_name} (Zero-shot: {self.is_zero_shot})")

        task = "zero-shot-classification" if self.is_zero_shot else "text-classification"

        try:
            self.pipeline = pipeline(
                task,
                model=model_name,
                return_all_scores=True if task == "text-classification" else None,
                truncation=True
            )
        except ValueError as e:
            if "sentencepiece" in str(e).lower() or "tiktoken" in str(e).lower():
                logger.error(
                    "Missing dependency for model tokenizer. "
                    "Please install required packages:\n\n"
                    "    pip install sentencepiece protobuf\n"
                )
            raise e
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise e

        # Legacy label map for standard text classification models
        self.label_map = {
            "LABEL_0": "not_hoax",
            "LABEL_1": "hoax",
            "fakta": "not_hoax",
            "hoaks": "hoax"
        }

    def classify(self, text: str) -> Dict[str, Any]:
        """Classify a piece of text and return label and score."""
        if not text.strip():
            return {"label": None, "score": None}

        try:
            if self.is_zero_shot:
                # Zero-shot output format: {'labels': ['hoaks', 'fakta'], 'scores': [0.9, 0.1]}
                result = self.pipeline(text, candidate_labels=self.CANDIDATE_LABELS)
                best_label = result['labels'][0]
                best_score = result['scores'][0]

                # Map "hoaks"/"fakta" to standard system labels "hoax"/"not_hoax"
                mapped_label = self.label_map.get(best_label, "not_hoax")

                # If mapped_label is already correct key (like if we used English labels), good.
                # Here: "hoaks" -> "hoax", "fakta" -> "not_hoax"

                return {"label": mapped_label, "score": best_score}

            else:
                # Standard text classification
                result = self.pipeline(text)[0] # List of dicts [{'label': 'LABEL_0', 'score': 0.9}, ...]
                best = max(result, key=lambda x: x['score'])
                label = self.label_map.get(best['label'], best['label'])

                # Threshold adjustment (legacy logic)
                if label == "hoax" and best["score"] < 0.65:
                    label = "not_hoax"

                return {"label": label, "score": best['score']}

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return {"label": None, "score": 0.0}
