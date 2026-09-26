# CardioLens-IoT: IoT layer

This folder contains a runnable reference implementation of the IoT framework: how an ECG travels from the patient to the physician.

```
 ECG sensing ──► IoT network ──► IoT cloud ───────────────────────────► User interface
 (patient)       (routing)       collection ─► analysis ─► alert        (dashboard, app,
                                  (storage)    (CNN +       (webhook)     notifications)
                                               Grad-CAM)
```

| Component | File | What it does |
|---|---|---|
| Edge gateway | [`edge/gateway_client.py`](edge/gateway_client.py) | Resizes the ECG image to 384 × 384, JPEG-compresses it, uploads it with patient and device IDs, and prints the result. |
| Network | [`network/hybrid_routing.py`](network/hybrid_routing.py) | Simulates a battery-powered sensor network and compares DSR, REL and hybrid routing (delivery ratio, hops, retransmissions, control messages, energy). |
| Cloud: collection | [`cloud/api.py`](cloud/api.py) | Stores every analysis as a JSON line in `records.jsonl`. |
| Cloud: analysis | [`cloud/api.py`](cloud/api.py) + [`../cardiolens/inference.py`](../cardiolens/inference.py) | Classifies the ECG with the trained EfficientNetB0 and, on request, returns a Grad-CAM overlay. |
| Cloud: alert | [`cloud/api.py`](cloud/api.py) | Assigns an alert level (critical for MI, warning for AHB or PMI, none for normal) and posts critical and warning alerts to a webhook. |
| User interface feed | `GET /records` | Latest analyses, optionally filtered by patient, for a dashboard or app. |

## Cloud API

Start it from the repository root (the trained model must be at `models/ecg_efficientnetb0.keras`):

```bash
uvicorn iot.cloud.api:app --host 0.0.0.0 --port 8000
```

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service status and model path |
| `POST` | `/analyze` | Form fields: `file` (image), `patient_id`, `device_id`, `explain` (true/false) |
| `GET` | `/records?patient_id=P001&limit=20` | Most recent analyses |

Example response of `/analyze`:

```json
{
  "record_id": "5b1f...",
  "timestamp": "2026-09-26T10:15:02+0200",
  "patient_id": "P001",
  "device_id": "GW-01",
  "prediction": {
    "label": "Myocardial_Infarction",
    "confidence": 0.9731,
    "probabilities": {"Abnormal_Heartbeat": 0.0102, "MI_history": 0.0121,
                      "Myocardial_Infarction": 0.9731, "Normal": 0.0046},
    "alert_level": "critical"
  },
  "inference_ms": 312.4,
  "alert_sent": true
}
```

Configuration through environment variables: `CARDIOLENS_MODEL` (model path), `CARDIOLENS_RECORDS` (record store) and `CARDIOLENS_WEBHOOK` (alert URL).

## Edge gateway

```bash
python iot/edge/gateway_client.py --image ecg.jpg --server http://localhost:8000 \
    --patient-id P001 --device-id GW-01 --explain
```

## Routing simulator

```bash
python iot/network/hybrid_routing.py --nodes 80 --area 140 --packets 10000 --seed 11
```

Options: `--nodes`, `--area` (metres), `--packets`, `--p-gateway` (share of traffic to the gateway), `--seed`.

The simulator uses a simplified model: distance-dependent link quality, a first-order radio energy model, up to 3 link-layer retransmissions per hop, and one control message per node for each route request, reply or proactive beacon. Its results illustrate the trade-offs between the policies. They are not measurements from real hardware.

## Security note

This is a research prototype. A production deployment would need authentication, TLS, encrypted storage of health data, and compliance with medical-data regulations.
