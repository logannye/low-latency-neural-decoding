"""Batch-one H1 adapter for FALCON's pinned public decoder interface.

Source: snel-repo/falcon-challenge@770c9b9597b8f58bf67b5a9ec6401ef705a2cb0f,
falcon_challenge/interface.py and evaluator.py. H1 uses continual=True: reset at
each file, then predict every bin including unscored bins; it does not call
observe or on_done. Those two compatibility hooks below never reset history.

This class implements the required methods without inheriting BCIDecoder or
importing FALCON's optional dependency stack. Tests establish synthetic interface
and state compatibility only, not hidden submission or container validation.
"""

from pathlib import Path

import numpy as np

from .models import LinearDecoder
from .temporal import TCNStream

FALCON_SOURCE_PIN = "770c9b9597b8f58bf67b5a9ec6401ef705a2cb0f"
FALCON_INTERFACE_SHA256 = "781dae89351d2c5e5914c8c9b47654254edd4f5a308a3bda85b4d91a755063e6"


class H1DecoderAdapter:
    """Wrap one LinearDecoder or TCNStream for independent H1 file replay.

    Construct one adapter per independent stream. The official evaluator calls
    reset before each file. Dataset tags identify that boundary only: they never
    select a model, retrieve files or grant access to calibration labels.
    """

    def __init__(self, decoder: LinearDecoder | TCNStream, *, batch_size: int = 1) -> None:
        self.set_batch_size(batch_size)
        if isinstance(decoder, LinearDecoder):
            decoder.model.validate()
            if decoder.model.mean.shape != (176,) or decoder.model.intercept.shape != (7,):
                raise ValueError("H1 requires exactly 176 input channels and seven outputs")
        elif not isinstance(decoder, TCNStream):
            raise TypeError("H1 adapter requires LinearDecoder or TCNStream")
        self._decoder = decoder

    @property
    def batch_size(self) -> int:
        return 1

    def set_batch_size(self, batch_size: int) -> None:
        if isinstance(batch_size, (bool, np.bool_)) or batch_size != 1:
            raise ValueError("H1 adapter supports batch size one only")

    def reset(self, dataset_tags: list[str | Path] | None = None) -> None:
        """Reset at a new file; tags remain opaque and are never opened or parsed."""
        if dataset_tags is not None:
            if not isinstance(dataset_tags, (list, tuple)) or len(dataset_tags) != 1:
                raise ValueError("batch-one reset requires exactly one dataset tag")
            if not isinstance(dataset_tags[0], (str, Path)):
                raise ValueError("dataset tag must be a string or Path")
        self._decoder.reset()

    def predict(self, neural_observations: np.ndarray) -> np.ndarray:
        """Consume one [1,176] count bin and return [1,7] in target units."""
        values = np.asarray(neural_observations)
        if values.shape != (1, 176) or values.dtype.kind not in "fiu":
            raise ValueError("H1 prediction requires numeric observations of shape (1, 176)")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("H1 counts must be finite and nonnegative")
        prediction = np.asarray(self._decoder.step(values[0]))
        if prediction.shape != (7,) or prediction.dtype.kind not in "fiu":
            raise ValueError("H1 decoder must return seven numeric outputs")
        if not np.isfinite(prediction).all():
            raise FloatingPointError("H1 decoder produced nonfinite outputs")
        return prediction.reshape(1, 7)

    def observe(self, neural_observations: np.ndarray) -> None:
        """Match the base interface: consume one bin and discard its prediction.

        H1's pinned movement loop uses predict for all bins instead. Calling both
        hooks on the same observation would intentionally consume it twice.
        """
        self.predict(neural_observations)

    def on_done(self, dones: np.ndarray) -> None:
        """Trial completion never resets H1 file history or changes predictions."""
        dones = np.asarray(dones)
        if dones.shape != (1,) or dones.dtype.kind not in "biu":
            raise ValueError("batch-one trial completion requires one boolean or integer flag")
        if not np.isin(dones, [0, 1]).all():
            raise ValueError("trial completion flag must be zero or one")
