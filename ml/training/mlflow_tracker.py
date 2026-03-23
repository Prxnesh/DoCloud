"""MLflow tracking wrapper for CloudInsight pipeline runs."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class MLflowTracker:
    """Thin wrapper that keeps MLflow usage optional and isolated."""

    def __init__(
        self,
        enabled: bool,
        experiment_name: str,
        tracking_uri: str | None = None,
        artifact_dir: str | Path | None = None,
    ) -> None:
        self.enabled = enabled
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.artifact_dir = Path(artifact_dir).expanduser().resolve() if artifact_dir else None
        self._mlflow = None

        if self.enabled:
            try:
                import mlflow
            except ImportError:
                self.enabled = False
            else:
                self._mlflow = mlflow
                if self.tracking_uri:
                    mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment_name)

    @contextmanager
    def run(self, run_name: str) -> Iterator[None]:
        """Start an MLflow run when tracking is enabled."""

        if not self.enabled or self._mlflow is None:
            yield
            return

        with self._mlflow.start_run(run_name=run_name):
            yield

    def log_params(self, params: dict[str, object]) -> None:
        """Log run parameters when tracking is enabled."""

        if self.enabled and self._mlflow is not None:
            self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log numeric metrics when tracking is enabled."""

        if self.enabled and self._mlflow is not None:
            self._mlflow.log_metrics(metrics)

    def log_text(self, text: str, artifact_file: str) -> None:
        """Persist text artifacts when tracking is enabled."""

        if self.enabled and self._mlflow is not None:
            self._mlflow.log_text(text, artifact_file)
