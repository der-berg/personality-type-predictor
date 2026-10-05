"""Load the fitted champion pipeline from this project's local MLflow store.

The app does not train a model or duplicate preprocessing. Imputation,
scaling, categorical encoding, and classification are stored together in the
logged scikit-learn pipeline.
"""

from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline


# A fresh modeling run creates a new ID in this machine's MLflow store.
# After verifying the intended champion run, paste its ID here.
CHAMPION_RUN_ID = ""


def get_project_root() -> Path:
    """Return this module's directory, independent of the current shell path.

    ``__file__`` remains tied to this module even if Streamlit starts from a
    different working directory.
    """

    return Path(__file__).resolve().parent


def get_tracking_uri() -> str:
    """Point MLflow at the project-local SQLite database used by the notebook."""

    database_path = get_project_root() / "mlflow.db"
    # SQLite URIs use forward slashes on Windows as well.
    return f"sqlite:///{database_path.as_posix()}"


def get_model_uri(run_id: str = CHAMPION_RUN_ID) -> str:
    """Return the model-artifact URI for one explicitly selected MLflow run.

    The ``/model`` suffix names the fitted pipeline artifact logged by
    ``modeling.ipynb``. An empty run ID is rejected before attempting to load.
    """

    if not run_id.strip():
        raise ValueError(
            "Champion run ID is missing. Run modeling.ipynb, verify the "
            "champion in MLflow, and set CHAMPION_RUN_ID in model_loader.py."
        )

    return f"runs:/{run_id}/model"


def load_champion_pipeline(run_id: str = CHAMPION_RUN_ID) -> Pipeline:
    """Load the fitted preprocessing-plus-classifier pipeline from MLflow.

    Predictions must pass a DataFrame with the same 22 raw features used for
    training. The loaded pipeline applies the learned transformations itself.
    """

    mlflow.set_tracking_uri(get_tracking_uri())
    model_uri = get_model_uri(run_id)
    return mlflow.sklearn.load_model(model_uri)
