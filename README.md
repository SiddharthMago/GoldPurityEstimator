# Gold Purity Estimator

This project estimates gold purity from object weight and image-based volume.

The original research notebooks are still available:

- `prototype.ipynb`: quick two-view volume estimation.
- `multiview.ipynb`: multiview visual hull volume estimation.

The app implementation adds:

- `backend/`: FastAPI service wrapping the notebook logic.
- `mobile_app/`: Expo React Native app for iOS and Android.

## Run Locally

Start the backend:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn gold_estimator.api:app --host 0.0.0.0 --port 8000
```

Start the app for laptop web testing:

```bash
cd mobile_app
npm install
npm run web
```

The web app uses `http://localhost:8000` by default.

Start the app for iPhone or Android testing with Expo Go:

```bash
cd mobile_app
npm install
EXPO_PUBLIC_API_URL=http://YOUR_COMPUTER_WIFI_IP:8000 npm run start -- --host lan --clear
```

Use Expo Go on iOS or Android to scan the QR code. This is free for local testing.

## Important Setup Notes

- Each photo must include a dark `1 cm x 1 cm` reference square.
- Use a plain background with strong contrast.
- Keep the object and marker in focus.
- For multiview, take side images in rotation order. Twelve images at 30 degree steps gives the best match to the notebook.
- On a real phone, `EXPO_PUBLIC_API_URL` must point to your computer's Wi-Fi address, such as `http://192.168.1.20:8000`. `localhost` only works for laptop web testing.

This is a prototype estimator, not an assay. Real purity depends on alloy composition, plating, internal voids, segmentation quality, and measurement setup.
