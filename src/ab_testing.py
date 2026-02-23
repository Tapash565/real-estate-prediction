"""A/B testing framework for model comparison."""

import hashlib
import os
import random
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from api.logging_config import get_logger

logger = get_logger(__name__)


class ExperimentStatus(Enum):
    """Experiment status."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


@dataclass
class Variant:
    """Experiment variant."""

    id: str
    name: str
    model_version: str
    weight: float
    traffic_percentage: float


@dataclass
class Experiment:
    """A/B test experiment."""

    id: str
    name: str
    description: str
    status: ExperimentStatus
    variants: list[Variant]
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    target_metric: str = "prediction_accuracy"


class ABTestManager:
    """Manager for A/B testing experiments."""

    def __init__(self) -> None:
        """Initialize A/B test manager."""
        self.experiments: dict[str, Experiment] = {}
        self.user_assignments: dict[str, str] = {}  # user_id -> variant_id
        self._load_experiments()

    def _load_experiments(self) -> None:
        """Load experiments from persistent storage."""
        # In production, load from database
        logger.info("experiments_loaded", count=len(self.experiments))

    def create_experiment(
        self,
        name: str,
        description: str,
        variants: list[tuple[str, str, float]],  # (id, model_version, traffic_pct)
        target_metric: str = "prediction_accuracy",
    ) -> str:
        """Create a new experiment.

        Args:
            name: Experiment name.
            description: Experiment description.
            variants: List of (variant_id, model_version, traffic_percentage).
            target_metric: Metric to track.

        Returns:
            Experiment ID.
        """
        experiment_id = f"exp-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"

        # Create variants
        variant_objects = []
        for variant_id, model_version, traffic_pct in variants:
            variant_objects.append(
                Variant(
                    id=variant_id,
                    name=f"Variant {variant_id}",
                    model_version=model_version,
                    weight=traffic_pct / 100.0,
                    traffic_percentage=traffic_pct,
                )
            )

        experiment = Experiment(
            id=experiment_id,
            name=name,
            description=description,
            status=ExperimentStatus.DRAFT,
            variants=variant_objects,
            target_metric=target_metric,
        )

        self.experiments[experiment_id] = experiment
        logger.info(
            "experiment_created",
            experiment_id=experiment_id,
            name=name,
            variants=[v.id for v in variant_objects],
        )

        return experiment_id

    def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment."""
        if experiment_id not in self.experiments:
            logger.error("experiment_not_found", experiment_id=experiment_id)
            return False

        experiment = self.experiments[experiment_id]
        experiment.status = ExperimentStatus.RUNNING
        experiment.start_date = datetime.now()

        logger.info("experiment_started", experiment_id=experiment_id)
        return True

    def assign_user_to_variant(
        self,
        experiment_id: str,
        user_id: str,
    ) -> Optional[Variant]:
        """Assign a user to a variant using consistent hashing.

        Args:
            experiment_id: Experiment ID.
            user_id: User identifier.

        Returns:
            Assigned variant or None.
        """
        if experiment_id not in self.experiments:
            return None

        experiment = self.experiments[experiment_id]

        if experiment.status != ExperimentStatus.RUNNING:
            return None

        # Check if user already assigned
        assignment_key = f"{experiment_id}:{user_id}"
        if assignment_key in self.user_assignments:
            variant_id = self.user_assignments[assignment_key]
            for variant in experiment.variants:
                if variant.id == variant_id:
                    return variant

        # Use consistent hashing for assignment
        hash_input = f"{experiment_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = hash_value % 100

        # Find variant based on traffic percentage
        cumulative = 0
        for variant in experiment.variants:
            cumulative += variant.traffic_percentage
            if bucket < cumulative:
                self.user_assignments[assignment_key] = variant.id
                logger.info(
                    "user_assigned_to_variant",
                    experiment_id=experiment_id,
                    user_id=user_id,
                    variant=variant.id,
                )
                return variant

        # Default to last variant
        return experiment.variants[-1] if experiment.variants else None

    def get_experiment_results(
        self,
        experiment_id: str,
    ) -> dict[str, Any]:
        """Get experiment results.

        Args:
            experiment_id: Experiment ID.

        Returns:
            Dictionary with experiment results.
        """
        if experiment_id not in self.experiments:
            return {"error": "Experiment not found"}

        experiment = self.experiments[experiment_id]

        # In production, query metrics from analytics database
        return {
            "experiment_id": experiment_id,
            "name": experiment.name,
            "status": experiment.status.value,
            "variants": [
                {
                    "id": v.id,
                    "model_version": v.model_version,
                    "traffic_percentage": v.traffic_percentage,
                    # These would come from actual metrics
                    "predictions_count": 0,
                    "mean_absolute_error": 0.0,
                    "r2_score": 0.0,
                }
                for v in experiment.variants
            ],
        }

    def get_active_experiments(self) -> list[Experiment]:
        """Get all running experiments."""
        return [
            exp for exp in self.experiments.values()
            if exp.status == ExperimentStatus.RUNNING
        ]

    def stop_experiment(self, experiment_id: str) -> bool:
        """Stop an experiment."""
        if experiment_id not in self.experiments:
            return False

        experiment = self.experiments[experiment_id]
        experiment.status = ExperimentStatus.COMPLETED
        experiment.end_date = datetime.now()

        logger.info("experiment_stopped", experiment_id=experiment_id)
        return True


# Global A/B test manager
ab_manager = ABTestManager()


def get_ab_manager() -> ABTestManager:
    """Get the global A/B test manager."""
    return ab_manager


def get_variant_for_user(experiment_id: str, user_id: str) -> Optional[Variant]:
    """Get assigned variant for a user."""
    return ab_manager.assign_user_to_variant(experiment_id, user_id)
