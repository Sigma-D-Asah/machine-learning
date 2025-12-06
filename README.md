# Machine Failure Prediction API

FastAPI untuk memprediksi kegagalan mesin menggunakan Machine Learning dengan dua model terpisah:
- **Binary Model**: Prediksi failure (0=not failed, 1=failed) 
- **Multiclass Model**: Prediksi tipe failure (string output)

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. **PENTING: Generate Scaler**
Scaler diperlukan untuk normalisasi data sesuai dengan training. Jalankan script ini sekali:

```bash
python notebooks/create_scaler.py
```

Output yang diharapkan:
```
✓ Data loaded: 10000 rows
✓ Scaler saved to: /path/to/src/models/scaler.pkl
```

### 3. Jalankan Server
```bash
# Opsi 1: Menggunakan uvicorn
uvicorn src.server:app --reload --host 0.0.0.0 --port 8000

# Machine Failure Prediction API — API Contract

Short, consumable contract for backend use.

Base URL
- http://localhost:8000

Endpoints
- GET /api/v1/failure/health
  - Response (200): {status_code, message, data: {binary_model_loaded, failure_type_model_loaded}}

- POST /api/v1/failure/predict/binary
  - Request (JSON): {product_id, type, air_temperature, process_temperature, rotational_speed, torque, tool_wear}
  - Response (200): {status_code, message, data: {prediction: 0|1, prediction_label, probability, confidence, input_data}}
  - Threshold: model uses probability >= 0.05 -> `failed` (1)

- POST /api/v1/failure/predict/type
  - Behavior: Checks binary prediction first. If binary==0, returns `No Failure` without running multiclass.
  - Response (200) when ran: {status_code, message, data: {prediction, probabilities, confidence, ambiguous, top_k, suggested_override, input_data}}
  - Response (200) when binary==0: returns `No Failure` with `probabilities` set accordingly and `ambiguous=false`.

Field overview
- `ambiguous`: set true when multiclass confidence < 0.3
- `top_k`: top 3 class predictions as [{label, prob}, ...]
- `suggested_override`: optional rule-based suggestion (e.g., high `tool_wear` suggests `Tool Wear Failure`)

Quick start
- Install: `pip install -r requirements.txt`
- Create scaler: `python notebooks/create_scaler.py` (creates `src/models/scaler.pkl`)
- Start server: `uvicorn src.server:app --reload --host 0.0.0.0 --port 8000`

Notes
- Binary and multiclass models are independent; use either or both.
- Tuning thresholds and rule override logic in `src/services/failure_prediction.py` is recommended for production behavior.
- Keep `scaler.pkl` synced with training preprocessing to avoid mispredictions.

Contact
- Update this contract and inform consumer teams for any breaking changes.

---

This README is intentionally concise — it focuses on API contract and important operational notes.
```

**Response:**
```json
{
  "status_code": 200,
  "message": "Failure type prediction successful",
  "data": {
    "prediction": "Tool Wear Failure",
    "probabilities": {
      "Heat Dissipation Failure": 0.0234,
      "No Failure": 0.8123,
      "Overstrain Failure": 0.0456,
      "Power Failure": 0.0234,
      "Random Failures": 0.0123,
      "Tool Wear Failure": 0.0830
    },
    "confidence": 0.8123,
    "input_data": { ... }
  }
}
```

**Note:** Model ini adalah prediksi independen dari binary model. Bisa digunakan terpisah atau bersamaan dengan binary prediction untuk analisis yang lebih lengkap.

---

## Input Parameters

| Parameter | Type | Required | Description | Range/Values |
|-----------|------|----------|-------------|--------------|
| `product_id` | string | ✅ | ID produk mesin | Any string |
| `type` | string | ✅ | Tipe mesin | `"L"`, `"M"`, `"H"` |
| `air_temperature` | float | ✅ | Suhu udara (Kelvin) | ≥ 0 |
| `process_temperature` | float | ✅ | Suhu proses (Kelvin) | ≥ 0 |
| `rotational_speed` | float | ✅ | Kecepatan rotasi (RPM) | ≥ 0 |
| `torque` | float | ✅ | Torsi (Nm) | ≥ 0 |
| `tool_wear` | float | ✅ | Keausan alat (menit) | ≥ 0 |

---

## Struktur Project

```
machine-learning/
├── src/
│   ├── server.py                         # FastAPI app
│   ├── controllers/
│   │   └── failure_prediction_controller.py
│   ├── models/
│   │   ├── model.py                      # Pydantic models & DTOs
│   │   ├── binaryfailure.keras           # Binary model
│   │   ├── multiclassfailtype.keras      # Multiclass model
│   │   └── scaler.pkl                    # MinMaxScaler (PENTING!)
│   └── services/
│       └── failure_prediction.py         # Business logic
├── notebooks/
│   ├── model.ipynb                       # Training notebook
│   ├── create_scaler.py                  # Generate scaler
│   └── predictive_maintenance_new.csv    # Training data
├── requirements.txt
├── test_api.py
└── README.md
```

---

## Technical Details

### Preprocessing Pipeline

1. **Label Encoding untuk `type`:**
   - `H` → 0
   - `L` → 1  
   - `M` → 2

2. **Feature Order** (PENTING - harus sesuai training):
   ```python
   ['type', 'air_temperature', 'process_temperature', 
    'rotational_speed', 'torque', 'tool_wear']
   ```

3. **MinMaxScaler Normalization:**
   - Semua 6 features di-normalize dengan `MinMaxScaler(feature_range=(0, 1))`
   - Scaler di-fit pada training data (`X_train`)
   - API menggunakan scaler yang sama untuk consistency

### Model Architecture

**Binary Model:**
```
Input (6 features)
  ↓
Dense(64, relu)
  ↓
Dense(32, relu)
  ↓
Dense(1, sigmoid) → probability of failure
```

**Multiclass Model:**
```
Input (6 features)
  ↓
Dense(64, relu)
  ↓
Dense(32, relu)
  ↓
Dense(6, softmax) → probabilities for 6 classes
```

---

## Important Notes

### 1. Scaler adalah Critical
API **TIDAK AKAN BERFUNGSI** tanpa `scaler.pkl`. Jika file hilang:
```bash
python notebooks/create_scaler.py
```

### 2. Model Performance
Model di-train dengan data yang **highly imbalanced**:
- No Failure: 9,652 samples (96.5%)
- Heat Dissipation Failure: 112 samples
- Power Failure: 95 samples
- Overstrain Failure: 78 samples
- Tool Wear Failure: 45 samples
- Random Failures: 18 samples

**Binary Model Threshold:** Diturunkan dari 0.5 ke **0.3** untuk meningkatkan sensitivity terhadap failure detection. Dengan threshold lebih rendah, model akan lebih agresif dalam memprediksi failure (lebih banyak true positives, tapi juga mungkin lebih banyak false positives).

### 3. Independensi Model
Binary dan multiclass model adalah **dua model terpisah** yang di-train secara independen. Keduanya bisa digunakan:
- **Terpisah**: Panggil binary atau multiclass saja sesuai kebutuhan
- **Bersamaan**: Panggil keduanya untuk analisis lengkap (failure probability + failure type)

---

## Contoh Penggunaan

### Python
```python
import requests

# Binary prediction
url_binary = "http://localhost:8000/api/v1/failure/predict/binary"
url_type = "http://localhost:8000/api/v1/failure/predict/type"

data = {
    "product_id": "L47257",
    "type": "L",
    "air_temperature": 298.8,
    "process_temperature": 308.9,
    "rotational_speed": 1455.0,
    "torque": 41.3,
    "tool_wear": 208.0
}

# Get binary prediction
binary_response = requests.post(url_binary, json=data)
binary_result = binary_response.json()['data']
print(f"Binary: {binary_result['prediction']} ({binary_result['prediction_label']})")
print(f"Probability: {binary_result['probability']:.4f}")

# Get failure type prediction
type_response = requests.post(url_type, json=data)
type_result = type_response.json()['data']
print(f"Failure Type: {type_result['prediction']}")
print(f"Confidence: {type_result['confidence']:.4f}")
```

### cURL
```bash
curl -X POST "http://localhost:8000/api/v1/failure/predict/binary" \
  -H "Content-Type: application/json" \
  -d '{
    "product_id": "L47257",
    "type": "L",
    "air_temperature": 298.8,
    "process_temperature": 308.9,
    "rotational_speed": 1455.0,
    "torque": 41.3,
    "tool_wear": 208.0
  }'
```

---

## License

MIT License

---