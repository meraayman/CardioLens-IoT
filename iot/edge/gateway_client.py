"""CardioLens-IoT edge gateway client.

Runs on the patient-side gateway (Raspberry Pi, phone, bedside PC). It pre-processes an ECG image
(resize to the model's input size, JPEG re-encoding to reduce bandwidth) and uploads it to the cloud.

    python iot/edge/gateway_client.py --image ecg.jpg --server http://localhost:8000 \
        --patient-id P001 --device-id GW-01 --explain
"""
from __future__ import annotations

import argparse
import base64
import io
import sys

import requests
from PIL import Image

IMG_SIZE = 384  # must match cardiolens.config.IMG_SIZE


def preprocess(path: str, quality: int = 90) -> bytes:
    """Resize to IMG_SIZE x IMG_SIZE RGB and JPEG-encode. Cuts upload size for low-bandwidth links."""
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description="Send an ECG image to the CardioLens-IoT cloud.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--patient-id", default="unknown")
    ap.add_argument("--device-id", default="gateway-01")
    ap.add_argument("--explain", action="store_true", help="request a Grad-CAM explanation")
    ap.add_argument("--gradcam-out", default="gradcam.png")
    args = ap.parse_args()

    payload = preprocess(args.image)
    print(f"Uploading {len(payload) / 1024:.1f} KB to {args.server}/analyze ...")
    resp = requests.post(f"{args.server}/analyze",
                         files={"file": ("ecg.jpg", payload, "image/jpeg")},
                         data={"patient_id": args.patient_id, "device_id": args.device_id,
                               "explain": str(args.explain).lower()},
                         timeout=60)
    if resp.status_code != 200:
        print("Error:", resp.status_code, resp.text)
        return 1

    result = resp.json()
    pred = result["prediction"]
    print(f"Prediction : {pred['label']} ({pred['confidence']:.1%})")
    print(f"Alert level: {pred['alert_level']}")
    if "gradcam_png_base64" in result:
        open(args.gradcam_out, "wb").write(base64.b64decode(result["gradcam_png_base64"]))
        print("Grad-CAM saved to", args.gradcam_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
