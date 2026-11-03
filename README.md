**Tech stack:** Python, Ultralytics YOLO11, OpenCV, ByteTrack, Streamlit, WebRTC, Pandas

# FrameSight

A Python video intelligence dashboard with YOLO11 detection, OpenCV video decoding, ByteTrack tracking, browser webcam streaming, and Pandas statistics.

## Run locally

Use Python 3.10–3.12 (3.11 recommended). From the project folder, start everything with:

```sh
./start.sh
```

The launcher creates `.venv` when needed, installs dependencies on the first run or when requirements change, and starts the dashboard. Open the local URL printed in the terminal. Press Ctrl+C to stop. Internet is needed for initial dependency installation and model download. Additional Streamlit options can be passed directly, for example `./start.sh --server.port 8503`.

For manual setup:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. The first analysis downloads official pretrained YOLO weights; internet is needed for this initial download. Inference runs on your server/computer. Browser webcam frames are sent to the Python server, so use a server you trust.

Upload an MP4, MOV, AVI, MKV, or WebM (up to 200 MB), choose confidence, resolution and categories, then click **Analyze video**. Empty category selection enables all 80 COCO classes. Each run starts a fresh tracker. The default frame limit is 1,500; raise it for longer clips. The preview updates every five processed frames, while every frame contributes to tracking and counts. Summary and detection CSVs remain available after processing.

For webcam use, select **Webcam**, press **START**, and allow camera access. The dashboard refreshes every second. Stop before changing settings. **Reset webcam session** clears counts and model state. Live inference speed depends on your hardware; try the nano model at 320 pixels if slow. Localhost works without TLS; remote browser camera access requires HTTPS. STUN is configured by default. Restrictive networks need a TURN service: add its URL and credentials to `rtc_configuration` in `app.py`, using server secrets for credentials.

## What the metrics mean

- **Objects in latest frame:** current detections, including boxes not yet assigned an ID.
- **Unique track IDs:** IDs first observed per category during a run. The first category seen owns the ID if predictions later fluctuate.
- **Detections:** per-frame observations; one object seen in 100 frames produces 100 observations.
- **Inference FPS:** processed frames divided by total detector/annotation time, excluding decoding, network and dashboard rendering.

ByteTrack is a motion-based tracker, not identity recognition. Occlusion, cuts and re-entry may assign a new ID to the same object. Counts are estimates rather than guaranteed distinct real-world objects. Pretrained categories are COCO classes; custom categories require custom weights and a corresponding UI update. History is bounded to 1,800 frames and detection logs to 50,000 rows. Unique ID and category totals span the full run. Video files are temporary and removed after processing; uploads are not permanently stored by the app.

## Structure and validation

`app.py` owns the Streamlit/WebRTC interface. `framesight/engine.py` owns one YOLO model and ByteTrack state per source/session. `framesight/stats.py` maintains thread-safe statistics. Models are deliberately not shared between users.

```sh
python -m unittest discover -s tests -v
python -m compileall -q app.py framesight tests
```

Manual smoke test: analyze a short clip containing people or cars, confirm labeled boxes and persistent IDs, download both CSVs, then start webcam and verify live metrics. Analyze a second clip to confirm IDs/counts reset. Try an invalid video to verify the readable error. Camera hardware and initial model downloads require an actual runtime environment.

For deployment, install requirements on a Python 3.11 host and run `streamlit run app.py --server.address 0.0.0.0`. Provide HTTPS and TURN for remote webcams, and budget CPU/GPU capacity per simultaneous session. Review Ultralytics' AGPL/enterprise licensing before distributing a proprietary deployment: https://www.ultralytics.com/license.

## Resume example

Built an interactive computer vision dashboard using Python, OpenCV, YOLO and ByteTrack to detect and track objects in uploaded videos and live webcam streams, with category counts, confidence annotations and exportable detection analytics.

Add measured throughput and validation results from your own hardware rather than claiming untested accuracy.

`requirements.lock.txt` records the exact versions tested on macOS ARM with Python 3.11. Use `requirements.txt` for portable installation; use the lock file when reproducing this environment.
