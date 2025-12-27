# Focus Tracker Backend

Focus Tracker is a Python backend that uses OpenCV and MediaPipe to compute a `looking_proxy` signal based on camera input.  
It is part of a larger desktop productivity monitoring system.

This backend is **responsible only for the vision logic and signals**.  

---

## 🧠 What This Backend Does

- Opens the camera using OpenCV
- Detects a face and landmarks using MediaPipe
- Computes a simple proxy for “user looking at the screen”
- Exposes structured signals used by the rest of the application



## 📦 Setup (Backend Only)

1. Create a virtual environment:

```bash
python3 -m venv venv
````

2. Activate it:

```bash
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

> This installs OpenCV, MediaPipe, NumPy and other required packages.

---

## ▶️ Running the Vision Module

Run the vision sandbox:

```bash
python -m backend.app
```


---

## 🧪 How It Works (High Level)

1. **OpenCV** opens the camera.
2. **MediaPipe FaceMesh** extracts facial landmarks.
3. A simple rule computes `looking_proxy`:

   * Head roughly centered → `True`
   * Not centered → `False`

This output can be used by a controller or UI module.

---

## 📁 Backend Structure

```
backend/
├── __init__.py
├── app.py
├── trackers/
│   ├── __init__.py
│   ├── camera.py
│   └── vision.py
└── utils/
    └── ...
```


## Logging

The application writes logs to `app.log`.

### Log rotation (Linux)

For long-running usage, log rotation is recommended.
A sample `logrotate` configuration is provided in:

To enable it on Fedora/Linux:

sudo cp config/log_rotate/ai-focus.conf /etc/log_rotate.d/ai-focus





