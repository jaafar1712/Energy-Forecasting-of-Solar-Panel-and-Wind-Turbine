"""Home Assistant to AI inference bridge.

This script keeps the Home Assistant data acquisition layer separate from the
local inference engine:
- Home Assistant is responsible for live sensor data collection.
- The Python model is responsible for prediction.
- Home Assistant receives only the resulting forecast value.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import requests

try:
    import tensorflow as tf
except Exception:  # TensorFlow is optional because some models are joblib-only.
    tf = None


LOGGER = logging.getLogger("ha_ai_bridge")

DEFAULT_FEATURE_ENTITY_MAP = {
    "solar_voltage": "sensor.solar_voltage",
    "wind_speed": "sensor.wind_speed",
}


def _load_json_env(name: str, default: Any) -> Any:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Environment variable {name} must contain valid JSON") from exc


def _load_feature_map() -> dict[str, str]:
    feature_map = _load_json_env("FEATURE_ENTITY_MAP", DEFAULT_FEATURE_ENTITY_MAP)
    if not isinstance(feature_map, dict) or not feature_map:
        raise ValueError("FEATURE_ENTITY_MAP must be a non-empty JSON object")
    return {str(key): str(value) for key, value in feature_map.items()}


@dataclass
class BridgeConfig:
    ha_url: str = os.getenv("HA_URL", "http://127.0.0.1:8123").rstrip("/")
    ha_token: str = os.getenv("HA_TOKEN", "").strip()
    model_path: Path = Path(os.getenv("MODEL_PATH", "power_model_rf.joblib"))
    model_backend: str = os.getenv("MODEL_BACKEND", "auto").strip().lower()
    prediction_entity: str = os.getenv("PREDICTION_ENTITY", "input_number.ai_prediction").strip()
    poll_interval_sec: float = float(os.getenv("POLL_INTERVAL_SEC", "5"))
    request_timeout_sec: float = float(os.getenv("REQUEST_TIMEOUT_SEC", "5"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    retry_backoff_sec: float = float(os.getenv("RETRY_BACKOFF_SEC", "2"))
    feature_entity_map: dict[str, str] = field(default_factory=_load_feature_map)
    feature_order: list[str] = field(default_factory=lambda: _load_json_env("FEATURE_ORDER", []))


@dataclass
class LoadedModel:
    model: Any
    backend: str
    feature_order: list[str] | None = None


class HomeAssistantClient:
    def __init__(self, config: BridgeConfig) -> None:
        if not config.ha_token:
            raise ValueError("HA_TOKEN is required")
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {config.ha_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    def _request(self, method: str, path: str, *, json_payload: dict[str, Any] | None = None) -> requests.Response:
        url = f"{self.config.ha_url}{path}"
        last_error: Exception | None = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    json=json_payload,
                    timeout=self.config.request_timeout_sec,
                )
                if response.status_code == 401:
                    response.raise_for_status()
                if response.status_code >= 400:
                    raise requests.HTTPError(
                        f"{method} {path} failed with status {response.status_code}: {response.text}",
                        response=response,
                    )
                return response
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                sleep_for = self.config.retry_backoff_sec * attempt
                LOGGER.warning(
                    "HA request failed on attempt %s/%s. Retrying in %.1f seconds: %s",
                    attempt,
                    self.config.max_retries,
                    sleep_for,
                    exc,
                )
                time.sleep(sleep_for)

        raise RuntimeError(f"Unable to complete HA request {method} {path}") from last_error

    def get_state(self, entity_id: str) -> float:
        response = self._request("GET", f"/api/states/{entity_id}")
        payload = response.json()
        raw_state = str(payload.get("state", "")).strip()
        if raw_state.lower() in {"unknown", "unavailable", "none", ""}:
            raise ValueError(f"Entity {entity_id} is unavailable")
        try:
            return float(raw_state)
        except ValueError as exc:
            raise ValueError(f"Entity {entity_id} did not return a numeric state: {raw_state}") from exc

    def set_input_number(self, entity_id: str, value: float) -> None:
        self._request(
            "POST",
            "/api/services/input_number/set_value",
            json_payload={"entity_id": entity_id, "value": float(value)},
        )


def load_model_bundle(model_path: Path, model_backend: str) -> LoadedModel:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    artifact: Any = None
    if model_backend in {"auto", "joblib"}:
        try:
            artifact = joblib.load(model_path)
        except Exception:
            if model_backend == "joblib":
                raise
        else:
            if isinstance(artifact, dict) and "model" in artifact:
                return LoadedModel(
                    model=artifact["model"],
                    backend="joblib",
                    feature_order=list(artifact.get("feature_cols") or []),
                )
            if model_backend == "joblib":
                return LoadedModel(model=artifact, backend="joblib")

    if model_backend in {"auto", "tensorflow", "tf"}:
        if tf is None:
            raise RuntimeError("TensorFlow is not installed, but MODEL_BACKEND requests it")
        tf_model = tf.keras.models.load_model(str(model_path))
        return LoadedModel(model=tf_model, backend="tensorflow")

    if artifact is not None:
        return LoadedModel(model=artifact, backend="joblib")

    raise ValueError(f"Unsupported MODEL_BACKEND: {model_backend}")


def resolve_feature_order(bundle: LoadedModel, config: BridgeConfig) -> list[str]:
    if config.feature_order:
        return [str(feature) for feature in config.feature_order]
    if bundle.feature_order:
        return bundle.feature_order
    return list(config.feature_entity_map.keys())


def collect_features(client: HomeAssistantClient, config: BridgeConfig, feature_order: list[str]) -> dict[str, float]:
    features: dict[str, float] = {}
    for feature_name in feature_order:
        entity_id = config.feature_entity_map.get(feature_name)
        if not entity_id:
            raise KeyError(
                f"Missing Home Assistant entity mapping for feature '{feature_name}'. Set FEATURE_ENTITY_MAP."
            )
        features[feature_name] = client.get_state(entity_id)
    return features


def _build_model_input(bundle: LoadedModel, features: dict[str, float], feature_order: list[str]) -> Any:
    ordered_values = [features[name] for name in feature_order]
    if bundle.backend == "tensorflow":
        return np.asarray([ordered_values], dtype=np.float32)

    try:
        import pandas as pd

        return pd.DataFrame([features], columns=feature_order)
    except Exception:
        return np.asarray([ordered_values], dtype=np.float32)


def run_inference(bundle: LoadedModel, features: dict[str, float], feature_order: list[str]) -> float:
    model_input = _build_model_input(bundle, features, feature_order)
    prediction = bundle.model.predict(model_input)

    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    if hasattr(prediction, "numpy"):
        prediction = prediction.numpy()

    array = np.asarray(prediction).reshape(-1)
    if array.size == 0:
        raise RuntimeError("Model returned an empty prediction")
    return float(array[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Bridge Home Assistant sensors to a local AI model")
    parser.add_argument("--once", action="store_true", help="Run one fetch/infer/update cycle and exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    config = BridgeConfig()
    client = HomeAssistantClient(config)
    bundle = load_model_bundle(config.model_path, config.model_backend)
    feature_order = resolve_feature_order(bundle, config)

    LOGGER.info("Loaded %s model from %s", bundle.backend, config.model_path)
    LOGGER.info("Using feature order: %s", ", ".join(feature_order))

    while True:
        try:
            features = collect_features(client, config, feature_order)
            prediction = run_inference(bundle, features, feature_order)
            client.set_input_number(config.prediction_entity, prediction)
            LOGGER.info("Published prediction %.3f to %s", prediction, config.prediction_entity)
        except KeyboardInterrupt:
            LOGGER.info("Bridge stopped by user")
            return
        except Exception as exc:
            LOGGER.exception("Bridge cycle failed: %s", exc)

        if args.once:
            return

        time.sleep(config.poll_interval_sec)


if __name__ == "__main__":
    main()