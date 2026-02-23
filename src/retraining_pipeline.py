"""Model retraining pipeline."""

import os
from datetime import datetime
from typing import Any, Optional

from api.logging_config import get_logger
from src.model_monitoring import get_performance_monitor

logger = get_logger(__name__)


class RetrainingPipeline:
    """Pipeline for automated model retraining."""

    def __init__(self) -> None:
        """Initialize retraining pipeline."""
        self.retraining_triggered = False
        self.last_retrain_time: Optional[datetime] = None

    def check_retraining_needed(
        self,
        mae_threshold: float = 50000.0,
        min_days_between_retrains: int = 1,
    ) -> dict[str, Any]:
        """Check if retraining is needed based on monitoring metrics.

        Args:
            mae_threshold: MAE threshold for triggering retrain.
            min_days_between_retrains: Minimum days between retraining attempts.

        Returns:
            Dictionary with retrain decision.
        """
        # Check if enough time has passed since last retrain
        if self.last_retrain_time:
            days_since_retrain = (datetime.now() - self.last_retrain_time).days
            if days_since_retrain < min_days_between_retrains:
                return {
                    "retrain_needed": False,
                    "reason": f"Too soon since last retrain ({days_since_retrain} days)",
                }

        # Check performance metrics
        monitor = get_performance_monitor()
        recommendation = monitor.should_retrain(mae_threshold=mae_threshold)

        return {
            "retrain_needed": recommendation["should_retrain"],
            "metrics": recommendation,
            "timestamp": datetime.now().isoformat(),
        }

    def trigger_retraining(
        self,
        data_path: Optional[str] = None,
        model_type: str = "random_forest",
        notify: bool = False,
    ) -> dict[str, Any]:
        """Trigger model retraining.

        Args:
            data_path: Path to training data.
            model_type: Type of model to train.
            notify: Whether to send notifications.

        Returns:
            Dictionary with retraining results.
        """
        if self.retraining_triggered:
            return {
                "status": "already_in_progress",
                "message": "Retraining already in progress",
            }

        self.retraining_triggered = True
        start_time = datetime.now()

        try:
            logger.info("retraining_started", model_type=model_type)

            # In production, this would:
            # 1. Fetch new data from data warehouse
            # 2. Run data validation checks
            # 3. Train new model
            # 4. Evaluate against holdout set
            # 5. Compare with current model
            # 6. If better, deploy to staging
            # 7. Run A/B test if configured
            # 8. Promote to production

            # For now, just run the existing training pipeline
            import subprocess

            result = subprocess.run(
                ["python", "run_all.py", "--mode", "train"],
                capture_output=True,
                text=True,
            )

            self.last_retrain_time = datetime.now()
            self.retraining_triggered = False

            success = result.returncode == 0

            if notify:
                self._send_notification(success)

            return {
                "status": "completed" if success else "failed",
                "start_time": start_time.isoformat(),
                "end_time": self.last_retrain_time.isoformat(),
                "duration_seconds": (self.last_retrain_time - start_time).total_seconds(),
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        except Exception as e:
            self.retraining_triggered = False
            logger.error("retraining_failed", error=str(e))
            return {
                "status": "failed",
                "error": str(e),
            }

    def _send_notification(self, success: bool) -> None:
        """Send notification about retraining completion."""
        # In production, integrate with Slack, email, PagerDuty, etc.
        logger.info(
            "retraining_notification",
            success=success,
            timestamp=datetime.now().isoformat(),
        )

    def get_status(self) -> dict[str, Any]:
        """Get retraining pipeline status."""
        return {
            "retraining_in_progress": self.retraining_triggered,
            "last_retrain_time": self.last_retrain_time.isoformat()
            if self.last_retrain_time
            else None,
        }


# Global pipeline instance
retraining_pipeline = RetrainingPipeline()


def get_retraining_pipeline() -> RetrainingPipeline:
    """Get the global retraining pipeline."""
    return retraining_pipeline
