"""
XGBoost model for long-term quality-adjusted credit valuation.

Architecture (from PRD §10.3):
  - Input: project type, vintage year, co-benefits, certification standard,
           region, policy sentiment score, EUA price trend
  - Feature importance via SHAP values
  - Output: quality-adjusted fair value estimate per credit category
"""

from pathlib import Path

import xgboost as xgb
import numpy as np


class CarbonXGBoost:
    """XGBoost regressor for carbon credit quality valuation."""

    def __init__(self, params: dict | None = None):
        self.params = params or {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
        self.model = xgb.XGBRegressor(**self.params)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the XGBoost model."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate fair-value predictions."""
        return self.model.predict(X)

    def save(self, path: Path) -> None:
        """Serialize model to disk."""
        self.model.save_model(str(path))

    def load(self, path: Path) -> None:
        """Load model from disk."""
        self.model.load_model(str(path))
