"""Inference and Grad-CAM for the trained CardioLens-IoT model.

Used by the IoT cloud API (iot/cloud/api.py) and from the command line:

    python -m cardiolens.inference path/to/ecg.jpg --model models/ecg_efficientnetb0.keras --gradcam out.png
"""
from __future__ import annotations

import argparse
import io
from dataclasses import dataclass, field

import numpy as np

from .config import ALERT_LEVELS, CLASS_NAMES, DEFAULT_MODEL_PATH, IMG_SIZE


def _tf():
    import tensorflow as tf  # imported lazily so light tools do not need TensorFlow
    return tf


def load_image(data: bytes) -> np.ndarray:
    """Decode image bytes exactly as in training: RGB, 384x384, anti-aliased, uint8."""
    tf = _tf()
    img = tf.io.decode_image(data, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), antialias=True)
    return tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8).numpy()


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict = field(default_factory=dict)
    alert_level: str = "none"

    def to_dict(self) -> dict:
        return {"label": self.label, "confidence": round(self.confidence, 4),
                "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
                "alert_level": self.alert_level}


class ECGClassifier:
    """Wraps the saved Keras model (EfficientNetB0 backbone + 'gap' + 'classifier' layers)."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        tf = _tf()
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.base = self.model.get_layer("efficientnetb0")
        self.gap = self.model.get_layer("gap")
        self.classifier = self.model.get_layer("classifier")

    def predict(self, image: np.ndarray) -> Prediction:
        probs = self.model.predict(image[None].astype("float32"), verbose=0)[0].astype(float)
        idx = int(np.argmax(probs))
        label = CLASS_NAMES[idx]
        return Prediction(label=label, confidence=float(probs[idx]),
                          probabilities=dict(zip(CLASS_NAMES, probs.tolist())),
                          alert_level=ALERT_LEVELS[label])

    def gradcam(self, image: np.ndarray) -> np.ndarray:
        """Grad-CAM heatmap (IMG_SIZE x IMG_SIZE, values 0-1) for the predicted class."""
        tf = _tf()
        x = tf.convert_to_tensor(image[None], tf.float32)
        with tf.GradientTape() as tape:
            conv = tf.cast(self.base(x, training=False), tf.float32)
            tape.watch(conv)
            probs = self.classifier(self.gap(conv))
            score = probs[:, int(tf.argmax(probs[0]))]
        grads = tape.gradient(score, conv)
        weights = tf.reduce_mean(grads, axis=(1, 2))[:, None, None, :]
        cam = tf.nn.relu(tf.reduce_sum(conv * weights, -1))[0]
        cam = cam / (tf.reduce_max(cam) + 1e-8)
        return tf.image.resize(cam[..., None], (IMG_SIZE, IMG_SIZE)).numpy()[..., 0]

    @staticmethod
    def overlay_png(image: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> bytes:
        """Render the heatmap over the ECG image and return PNG bytes."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(image); ax.imshow(cam, cmap="jet", alpha=alpha); ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        plt.close(fig)
        return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser(description="Classify a 12-lead ECG image.")
    ap.add_argument("image", help="path to an ECG image (jpg/png)")
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--gradcam", metavar="OUT_PNG", help="also save a Grad-CAM overlay")
    args = ap.parse_args()

    clf = ECGClassifier(args.model)
    img = load_image(open(args.image, "rb").read())
    pred = clf.predict(img)
    print(pred.to_dict())
    if args.gradcam:
        open(args.gradcam, "wb").write(clf.overlay_png(img, clf.gradcam(img)))
        print("Grad-CAM saved to", args.gradcam)


if __name__ == "__main__":
    main()
