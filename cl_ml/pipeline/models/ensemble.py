"""
Weighted ensemble combining LSTM + XGBoost predictions.

From PRD §10.5:
  - LSTM provides the price trend signal
  - XGBoost provides the quality valuation signal
  - Weights tuned on held-out validation set
"""

import numpy as np


class CarbonEnsemble:
    """Weighted ensemble of LSTM and XGBoost predictions."""

    def __init__(self, lstm_weight: float = 0.4, xgb_weight: float = 0.6):
        assert abs(lstm_weight + xgb_weight - 1.0) < 1e-6, "Weights must sum to 1"
        self.lstm_weight = lstm_weight
        self.xgb_weight = xgb_weight

    def predict(
        self,
        lstm_pred: np.ndarray,
        xgb_pred: np.ndarray,
    ) -> dict:
        """
        Combine predictions into a fair-value estimate with confidence range.

        Returns dict matching the PRD output format.
        """
        fair_value = (self.lstm_weight * lstm_pred) + (self.xgb_weight * xgb_pred)

        # Simple confidence range based on model disagreement
        spread = np.abs(lstm_pred - xgb_pred)
        confidence_low = fair_value - spread
        confidence_high = fair_value + spread

        return {
            "fair_value": float(np.round(fair_value, 2)),
            "confidence_low": float(np.round(confidence_low, 2)),
            "confidence_high": float(np.round(confidence_high, 2)),
        }
