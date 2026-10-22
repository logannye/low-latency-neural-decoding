"""Feature-only recorded-data replay. Scoring is a separate caller's operation."""

from pathlib import Path

import numpy as np

from .data import load_feature_file
from .models import LinearDecoder, load_model


def replay_features(model_path, feature_path, output, kind="linear"):
    """No target/mask file is opened, and one invocation is one reset file stream."""
    features = load_feature_file(feature_path)
    if kind == "linear":
        model = load_model(model_path)
        decoder, channels = LinearDecoder(model), len(model.mean)
    elif kind == "tcn":
        from .temporal import TCNStream, load_tcn

        decoder, channels = TCNStream(load_tcn(model_path)), 176
    else:
        raise ValueError("unknown decoder kind")
    if not np.array_equal(features["channel_ids"], np.arange(channels)):
        raise ValueError("channel map disagrees with model")
    predictions = decoder.predict_chunk(features["counts"])
    with Path(output).open("xb") as stream:
        np.save(stream, predictions, allow_pickle=False)
    return predictions
