"""
Training orchestrator for CarbonLens ML models.

Orchestrates the full training pipeline:
  1. Load data (EU ETS, Verra, News)
  2. Feature engineering
  3. Train LSTM on price sequences
  4. Train XGBoost on quality features
  5. Tune ensemble weights
  6. Evaluate against baseline (30-day MA)
  7. Export model artifacts
"""

from pathlib import Path


def run_training_pipeline(
    data_dir: Path,
    output_dir: Path,
    window_size: int = 30,
) -> dict:
    """
    Execute the full training pipeline.

    Returns evaluation metrics dict.
    """
    # TODO: implement full training pipeline
    # Steps:
    #   1. data = load_eu_ets_prices(data_dir)
    #   2. verra = load_verra_registry(data_dir)
    #   3. news = load_news_feed(data_dir)
    #   4. features = engineer_features(data, verra, news)
    #   5. lstm_model = train_lstm(features.price_sequences)
    #   6. xgb_model = train_xgboost(features.quality_features)
    #   7. ensemble = tune_ensemble_weights(lstm_model, xgb_model, val_data)
    #   8. metrics = evaluate_against_baseline(ensemble, test_data)
    #   9. save_artifacts(output_dir, lstm_model, xgb_model, ensemble)
    raise NotImplementedError("Training pipeline not yet implemented")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train CarbonLens ML models")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("./artifacts"))
    parser.add_argument("--window-size", type=int, default=30)
    args = parser.parse_args()

    metrics = run_training_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        window_size=args.window_size,
    )
    print(f"Training complete. Metrics: {metrics}")
