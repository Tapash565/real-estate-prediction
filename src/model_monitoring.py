"""Model performance monitoring and drift detection."""

import json
import os
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from api.caching import PredictionCache, cache
from api.logging_config import get_logger

logger = get_logger(__name__)

# Monitoring configuration
MONITORING_DB_PATH = os.getenv("MONITORING_DB_PATH", "./monitoring")
PREDICTION_LOG_PATH = os.path.join(MONITORING_DB_PATH, "predictions")
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.05"))


class PredictionLogger:
    """Logger for prediction requests and results."""

    def __init__(self) -> None:
        """Initialize prediction logger."""
        os.makedirs(PREDICTION_LOG_PATH, exist_ok=True)
        self.current_date = datetime.now().date()
        self.log_file: Optional[Any] = None
        self._open_log_file()

    def _open_log_file(self) -> None:
        """Open log file for current date."""
        filename = f"predictions_{self.current_date.isoformat()}.jsonl"
        filepath = os.path.join(PREDICTION_LOG_PATH, filename)
        self.log_file = open(filepath, "a")

    def _check_rotation(self) -> None:
        """Check if we need to rotate to a new day's file."""
        today = datetime.now().date()
        if today != self.current_date:
            if self.log_file:
                self.log_file.close()
            self.current_date = today
            self._open_log_file()

    def log_prediction(
        self,
        features: dict[str, Any],
        prediction: float,
        actual: Optional[float] = None,
        latency: float = 0.0,
        cached: bool = False,
        model_version: str = "unknown",
        user_id: Optional[str] = None,
    ) -> None:
        """Log a prediction.

        Args:
            features: Input features.
            prediction: Predicted value.
            actual: Actual value if available.
            latency: Prediction latency in seconds.
            cached: Whether prediction was cached.
            model_version: Model version used.
            user_id: User ID.
        """
        self._check_rotation()

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "features": features,
            "prediction": prediction,
            "actual": actual,
            "latency": latency,
            "cached": cached,
            "model_version": model_version,
            "user_id": user_id,
        }

        if self.log_file:
            self.log_file.write(json.dumps(log_entry) + "\n")
            self.log_file.flush()

        logger.info(
            "prediction_logged",
            prediction=prediction,
            model_version=model_version,
            latency=latency,
        )

    def close(self) -> None:
        """Close log file."""
        if self.log_file:
            self.log_file.close()
            self.log_file = None


class DriftDetector:
    """Detector for model and data drift."""

    def __init__(self, reference_data_path: Optional[str] = None) -> None:
        """Initialize drift detector.

        Args:
            reference_data_path: Path to reference data distribution.
        """
        self.reference_data: Optional[pd.DataFrame] = None
        self.reference_stats: dict[str, dict[str, Any]] = {}

        if reference_data_path and os.path.exists(reference_data_path):
            self.reference_data = pd.read_csv(reference_data_path)
            self._compute_reference_stats()

    def _compute_reference_stats(self) -> None:
        """Compute statistics from reference data."""
        if self.reference_data is None:
            return

        for column in self.reference_data.select_dtypes(include=[np.number]).columns:
            self.reference_stats[column] = {
                "mean": self.reference_data[column].mean(),
                "std": self.reference_data[column].std(),
                "min": self.reference_data[column].min(),
                "max": self.reference_data[column].max(),
                "quantiles": self.reference_data[column].quantile([0.25, 0.5, 0.75]).to_dict(),
            }

    def compute_drift_score(
        self,
        current_data: pd.DataFrame,
        feature: str,
    ) -> dict[str, Any]:
        """Compute drift score for a single feature using Kolmogorov-Smirnov test.

        Args:
            current_data: Current data distribution.
            feature: Feature name.

        Returns:
            Dictionary with drift score and statistics.
        """
        if self.reference_data is None or feature not in self.reference_data.columns:
            return {"error": "Reference data or feature not available"}

        reference_values = self.reference_data[feature].dropna()
        current_values = current_data[feature].dropna()

        if len(reference_values) == 0 or len(current_values) == 0:
            return {"error": "Insufficient data"}

        # Perform KS test
        statistic, p_value = stats.ks_2samp(reference_values, current_values)

        # Compute effect size (Cohen's d)
        mean_diff = current_values.mean() - reference_values.mean()
        pooled_std = np.sqrt(
            (current_values.std() ** 2 + reference_values.std() ** 2) / 2
        )
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

        return {
            "ks_statistic": statistic,
            "p_value": p_value,
            "cohens_d": cohens_d,
            "drift_detected": p_value < DRIFT_THRESHOLD,
            "reference_mean": reference_values.mean(),
            "current_mean": current_values.mean(),
            "mean_shift_percent": (mean_diff / reference_values.mean()) * 100
            if reference_values.mean() != 0 else 0,
        }

    def detect_drift(
        self,
        current_data: pd.DataFrame,
        features: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Detect drift across all features.

        Args:
            current_data: Current data distribution.
            features: List of features to check. If None, check all numerical features.

        Returns:
            Dictionary with drift results.
        """
        if self.reference_data is None:
            return {"error": "Reference data not available"}

        if features is None:
            features = list(self.reference_data.select_dtypes(include=[np.number]).columns)

        feature_drift = {}
        drift_detected = False
        total_drift_score = 0.0

        for feature in features:
            if feature in current_data.columns:
                result = self.compute_drift_score(current_data, feature)
                feature_drift[feature] = result
                if result.get("drift_detected", False):
                    drift_detected = True
                total_drift_score += result.get("ks_statistic", 0)

        return {
            "drift_detected": drift_detected,
            "drift_score": total_drift_score / len(features) if features else 0,
            "feature_drift": feature_drift,
            "threshold": DRIFT_THRESHOLD,
            "timestamp": datetime.now().isoformat(),
        }


class PerformanceMonitor:
    """Monitor model performance metrics."""

    def __init__(self) -> None:
        """Initialize performance monitor."""
        self.prediction_logger = PredictionLogger()

    def log_prediction(
        self,
        features: dict[str, Any],
        prediction: float,
        actual: Optional[float] = None,
        latency: float = 0.0,
        cached: bool = False,
        model_version: str = "unknown",
        user_id: Optional[str] = None,
    ) -> None:
        """Log a prediction."""
        self.prediction_logger.log_prediction(
            features=features,
            prediction=prediction,
            actual=actual,
            latency=latency,
            cached=cached,
            model_version=model_version,
            user_id=user_id,
        )

    def get_prediction_stats(
        self,
        hours: int = 24,
    ) -> dict[str, Any]:
        """Get prediction statistics for the last N hours.

        Args:
            hours: Number of hours to look back.

        Returns:
            Dictionary with statistics.
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        predictions = []

        # Read log files
        for filename in os.listdir(PREDICTION_LOG_PATH):
            if filename.startswith("predictions_") and filename.endswith(".jsonl"):
                filepath = os.path.join(PREDICTION_LOG_PATH, filename)
                try:
                    with open(filepath) as f:
                        for line in f:
                            entry = json.loads(line)
                            entry_time = datetime.fromisoformat(entry["timestamp"])
                            if entry_time >= cutoff:
                                predictions.append(entry)
                except Exception as e:
                    logger.error("error_reading_prediction_log", error=str(e))

        if not predictions:
            return {
                "period_hours": hours,
                "total_predictions": 0,
                "message": "No predictions in period",
            }

        # Compute statistics
        latencies = [p["latency"] for p in predictions if "latency" in p]
        cached_count = sum(1 for p in predictions if p.get("cached", False))
        predictions_values = [p["prediction"] for p in predictions]

        # Compute accuracy if actuals available
        actuals = [(p["prediction"], p["actual"]) for p in predictions if p.get("actual")]
        mae = None
        if actuals:
            mae = sum(abs(pred - act) for pred, act in actuals) / len(actuals)

        return {
            "period_hours": hours,
            "total_predictions": len(predictions),
            "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
            "max_latency": max(latencies) if latencies else 0,
            "cache_hit_rate": cached_count / len(predictions) if predictions else 0,
            "prediction_value_mean": sum(predictions_values) / len(predictions_values) if predictions_values else 0,
            "prediction_value_std": np.std(predictions_values) if len(predictions_values) > 1 else 0,
            "mean_absolute_error": mae,
        }

    def should_retrain(
        self,
        mae_threshold: float = 50000.0,
        min_predictions: int = 100,
    ) -> dict[str, Any]:
        """Check if model should be retrained based on performance.

        Args:
            mae_threshold: MAE threshold for triggering retrain.
            min_predictions: Minimum predictions before retrain decision.

        Returns:
            Dictionary with retrain recommendation.
        """
        stats = self.get_prediction_stats(hours=24 * 7)  # Last week

        if stats["total_predictions"] < min_predictions:
            return {
                "should_retrain": False,
                "reason": "Insufficient predictions for decision",
                "predictions_count": stats["total_predictions"],
                "min_required": min_predictions,
            }

        current_mae = stats.get("mean_absolute_error")
        if current_mae is None:
            return {
                "should_retrain": False,
                "reason": "No ground truth data available",
            }

        should_retrain = current_mae > mae_threshold

        return {
            "should_retrain": should_retrain,
            "current_mae": current_mae,
            "threshold": mae_threshold,
            "predictions_count": stats["total_predictions"],
        }


# Global instances
prediction_logger = PredictionLogger()
performance_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor."""
    return performance_monitor


def get_drift_detector(reference_data_path: Optional[str] = None) -> DriftDetector:
    """Create a drift detector."""
    return DriftDetector(reference_data_path)
