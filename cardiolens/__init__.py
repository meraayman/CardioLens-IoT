"""CardioLens-IoT: explainable 12-lead ECG image classification for IoT cardiac monitoring."""

from .config import CLASS_NAMES, SHORT_NAMES, IMG_SIZE, ALERT_LEVELS

__version__ = "1.0.0"
__all__ = ["CLASS_NAMES", "SHORT_NAMES", "IMG_SIZE", "ALERT_LEVELS", "__version__"]
