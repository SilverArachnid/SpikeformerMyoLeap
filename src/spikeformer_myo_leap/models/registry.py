"""Model registry for training and evaluation entry points."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from torch import nn


MODEL_REGISTRY: dict[str, str] = {
    "spikeformer": "spikeformer_myo_leap.models.spikeformer.EMGSpikeformerWindowRegressor",
    "transformer": "spikeformer_myo_leap.models.transformer.EMGTransformerWindowRegressor",
    "cnn_lstm": "spikeformer_myo_leap.models.cnn_lstm.EMGCNNLSTMRegressor",
    "cnn": "spikeformer_myo_leap.models.cnn.EMGCNNRegressor",
    "spiking_cnn": "spikeformer_myo_leap.models.spiking_cnn.SpikingCNNRegressor",
}


def model_names() -> list[str]:
    """Return the supported model names in stable sorted order."""

    return sorted(MODEL_REGISTRY)


def get_model_class(model_name: str) -> type[nn.Module]:
    """Look up a model class by registry name."""

    try:
        module_path, class_name = MODEL_REGISTRY[model_name].rsplit(".", 1)
    except KeyError as exc:
        raise KeyError(f"Unknown model '{model_name}'. Available models: {', '.join(model_names())}") from exc
    module = import_module(module_path)
    return getattr(module, class_name)


def create_model(model_name: str, **kwargs: Any) -> nn.Module:
    """Instantiate a model from the registry."""

    return get_model_class(model_name)(**kwargs)
