"""MLflow integration for model versioning and experiment tracking."""

import os
from datetime import datetime
from typing import Any, Optional

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from api.logging_config import get_logger

logger = get_logger(__name__)

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "real-estate-prediction")


class MLflowTracker:
    """MLflow experiment tracker."""

    def __init__(self) -> None:
        """Initialize MLflow tracker."""
        self._initialized = False
        self._client: Optional[MlflowClient] = None

    def initialize(self) -> bool:
        """Initialize MLflow connection."""
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

            # Create client for model registry
            self._client = MlflowClient(MLFLOW_TRACKING_URI)

            self._initialized = True
            logger.info(
                "mlflow_initialized",
                tracking_uri=MLFLOW_TRACKING_URI,
                experiment=MLFLOW_EXPERIMENT_NAME,
            )
            return True
        except Exception as e:
            logger.warning("mlflow_initialization_failed", error=str(e))
            self._initialized = False
            return False

    def is_initialized(self) -> bool:
        """Check if MLflow is initialized."""
        return self._initialized

    def log_model(
        self,
        model: Any,
        model_name: str,
        metrics: dict[str, float],
        params: dict[str, Any],
        artifacts: Optional[list[str]] = None,
        tags: Optional[dict[str, str]] = None,
        register: bool = False,
    ) -> Optional[str]:
        """Log a model to MLflow.

        Args:
            model: The trained model.
            model_name: Name of the model.
            metrics: Dictionary of metrics to log.
            params: Dictionary of parameters to log.
            artifacts: List of artifact file paths to log.
            tags: Dictionary of tags to log.
            register: Whether to register the model.

        Returns:
            Run ID if successful, None otherwise.
        """
        if not self._initialized:
            logger.warning("mlflow_not_initialized")
            return None

        try:
            with mlflow.start_run() as run:
                run_id = run.info.run_id

                # Log parameters
                for key, value in params.items():
                    mlflow.log_param(key, value)

                # Log metrics
                for key, value in metrics.items():
                    mlflow.log_metric(key, value)

                # Log tags
                if tags:
                    for key, value in tags.items():
                        mlflow.set_tag(key, value)

                # Log model
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=model_name if register else None,
                )

                # Log artifacts
                if artifacts:
                    for artifact in artifacts:
                        if os.path.exists(artifact):
                            mlflow.log_artifact(artifact)

                logger.info(
                    "model_logged_to_mlflow",
                    run_id=run_id,
                    model_name=model_name,
                    metrics=metrics,
                )

                return run_id

        except Exception as e:
            logger.error("mlflow_log_failed", error=str(e))
            return None

    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> Optional[Any]:
        """Load a model from MLflow.

        Args:
            model_name: Name of the model.
            version: Specific version to load.
            stage: Stage to load (e.g., "Production", "Staging").

        Returns:
            The loaded model or None.
        """
        if not self._initialized:
            logger.warning("mlflow_not_initialized")
            return None

        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
            elif stage:
                model_uri = f"models:/{model_name}/{stage}"
            else:
                model_uri = f"models:/{model_name}/latest"

            model = mlflow.sklearn.load_model(model_uri)
            logger.info("model_loaded_from_mlflow", model_name=model_name, version=version)
            return model

        except Exception as e:
            logger.error("mlflow_load_failed", error=str(e))
            return None

    def get_model_versions(self, model_name: str) -> list[dict[str, Any]]:
        """Get all versions of a model.

        Args:
            model_name: Name of the model.

        Returns:
            List of version information dictionaries.
        """
        if not self._initialized or self._client is None:
            logger.warning("mlflow_not_initialized")
            return []

        try:
            versions = self._client.get_latest_versions(model_name)
            return [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "status": v.status,
                    "run_id": v.run_id,
                    "creation_timestamp": datetime.fromtimestamp(
                        v.creation_timestamp / 1000
                    ).isoformat(),
                }
                for v in versions
            ]
        except Exception as e:
            logger.error("mlflow_get_versions_failed", error=str(e))
            return []

    def transition_model_version_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
    ) -> bool:
        """Transition a model version to a new stage.

        Args:
            model_name: Name of the model.
            version: Version number.
            stage: New stage (e.g., "Staging", "Production").

        Returns:
            True if successful.
        """
        if not self._initialized or self._client is None:
            logger.warning("mlflow_not_initialized")
            return False

        try:
            self._client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
            )
            logger.info(
                "model_stage_transitioned",
                model_name=model_name,
                version=version,
                stage=stage,
            )
            return True
        except Exception as e:
            logger.error("mlflow_transition_stage_failed", error=str(e))
            return False


# Global tracker instance
mlflow_tracker = MLflowTracker()


def initialize_mlflow() -> bool:
    """Initialize MLflow tracking."""
    return mlflow_tracker.initialize()


def log_model_with_tracking(
    model: Any,
    model_name: str,
    metrics: dict[str, float],
    params: dict[str, Any],
    artifacts: Optional[list[str]] = None,
    register: bool = False,
) -> Optional[str]:
    """Log model to MLflow with tracking."""
    return mlflow_tracker.log_model(
        model=model,
        model_name=model_name,
        metrics=metrics,
        params=params,
        artifacts=artifacts,
        register=register,
    )
