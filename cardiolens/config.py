"""Shared constants. These must match the training notebook."""

IMG_SIZE = 384

CLASS_NAMES = ["Abnormal_Heartbeat", "MI_history", "Myocardial_Infarction", "Normal"]
SHORT_NAMES = ["AHB", "PMI", "MI", "Normal"]

# Alert level raised by the IoT disease-alert module for each predicted class.
ALERT_LEVELS = {
    "Myocardial_Infarction": "critical",   # possible acute MI: notify clinicians immediately
    "Abnormal_Heartbeat":    "warning",    # abnormal rhythm: schedule review
    "MI_history":            "warning",    # previous MI: follow-up
    "Normal":                "none",
}

DEFAULT_MODEL_PATH = "models/ecg_efficientnetb0.keras"
