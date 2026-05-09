"""
Data loaders for CarbonLens ML pipeline.

Handles ingestion from:
  - EU ETS price data (EEA / DataHub)
  - Verra registry (Berkeley Carbon Trading Project)
  - News / policy feeds (GDELT / NewsAPI)
"""

from pathlib import Path

import pandas as pd


def load_eu_ets_prices(data_dir: Path) -> pd.DataFrame:
    """Load and clean EU ETS historical price data."""
    # TODO: implement EEA API / DataHub CSV ingestion
    raise NotImplementedError("EU ETS data loader not yet implemented")


def load_verra_registry(data_dir: Path) -> pd.DataFrame:
    """Load Verra project registry data from Berkeley dataset."""
    # TODO: implement Berkeley CSV / verra-scraper integration
    raise NotImplementedError("Verra data loader not yet implemented")


def load_news_feed(data_dir: Path) -> pd.DataFrame:
    """Load news articles for NLP sentiment analysis."""
    # TODO: implement GDELT / NewsAPI ingestion
    raise NotImplementedError("News feed loader not yet implemented")
