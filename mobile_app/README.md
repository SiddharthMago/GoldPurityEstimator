# Gold Purity Estimator Mobile App

Expo React Native app for laptop web testing and iOS/Android testing through Expo Go.

The project targets Expo SDK 54.

## Setup

```bash
cd mobile_app
npm install
```

For laptop web testing:

```bash
npm run web
```

The web app uses `http://localhost:8000` by default.

For physical phone testing, keep the backend running and start Expo with your computer's Wi-Fi IP:

```bash
EXPO_PUBLIC_API_URL=http://YOUR_COMPUTER_WIFI_IP:8000 npm run start -- --host lan --clear
```

Install Expo Go on your phone and scan the QR code. The phone must be able to open `http://YOUR_COMPUTER_WIFI_IP:8000/health` in Safari/Chrome.

## Photo Flow

Quick mode:

- Enter object weight in grams.
- Add one top photo.
- Add one side photo.
- Estimate purity.

Multiview mode:

- Enter object weight in grams.
- Add one top photo.
- Add at least 3 side photos in rotation order. Twelve photos at 30 degree steps is preferred.
- Estimate purity.

Every photo needs the dark `1 cm x 1 cm` square reference marker visible.

