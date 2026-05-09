"""
NLP sentiment analysis using FinBERT.

From PRD §10.4:
  - Model: FinBERT (pre-trained financial sentiment BERT)
  - Input: daily aggregated news articles filtered by carbon/climate keywords
  - Output: daily sentiment score from -1 (bearish) to +1 (bullish)
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np


# Carbon market keyword filters (from PRD §9.3)
POLICY_KEYWORDS = [
    "carbon tax", "CBAM", "EU ETS reform", "Article 6",
    "emissions trading directive",
]
MARKET_KEYWORDS = [
    "carbon credit", "offset", "Verra", "Gold Standard",
    "voluntary carbon market",
]
CLIMATE_KEYWORDS = [
    "net zero", "climate policy", "UNFCCC", "COP", "Paris Agreement",
]
ALL_KEYWORDS = POLICY_KEYWORDS + MARKET_KEYWORDS + CLIMATE_KEYWORDS


class SentimentAnalyzer:
    """FinBERT-based sentiment scorer for carbon market news."""

    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        """Load FinBERT model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_NAME
        ).to(self.device)
        self.model.eval()

    def score(self, text: str) -> float:
        """
        Compute sentiment score for a single text.

        Returns float in [-1, +1] where:
          -1 = strongly bearish
           0 = neutral
          +1 = strongly bullish
        """
        if self.model is None:
            raise RuntimeError("Call .load() before scoring")

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        # FinBERT labels: [positive, negative, neutral]
        positive, negative, neutral = probs
        return float(np.round(positive - negative, 4))

    def score_batch(self, texts: list[str]) -> float:
        """Compute aggregate daily sentiment from multiple articles."""
        if not texts:
            return 0.0
        scores = [self.score(t) for t in texts]
        return float(np.round(np.mean(scores), 4))
