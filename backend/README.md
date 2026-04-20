# Gold Purity Estimator Backend

FastAPI service for image-based volume estimation and gold purity calculation.

Each uploaded photo must include the dark `1 cm x 1 cm` reference square used by the notebooks. The marker is how the backend converts pixels into centimeters.

## Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn gold_estimator.api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/health` to confirm the API is running.

For testing on a physical phone, keep the phone and computer on the same Wi-Fi network. Start the Expo app with `EXPO_PUBLIC_API_URL` set to your computer's LAN address, for example `http://192.168.1.20:8000`.

## API

- `POST /estimate/quick`: requires `weight_g`, `top_image`, and `side_image`.
- `POST /estimate/multiview`: requires `weight_g`, `top_image`, and repeated `side_images`. Optional `angles_deg` is a JSON array such as `[0,30,60]`.

The response includes estimated volume, density, polynomial karat estimate, density-range match, diagnostics, mask/graph visuals, and a prototype warning.

