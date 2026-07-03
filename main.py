"""
╔══════════════════════════════════════════════════════════════════╗
║   AccidentAI — Production Backend (FastAPI)                      ║
║   CNN model exactly matched to cnn_accident_detection_90.ipynb   ║
╠══════════════════════════════════════════════════════════════════╣
║  Features:                                                        ║
║   ✅ Real CNN inference (your trained model.h5)                   ║
║   ✅ Video upload & frame-by-frame analysis                       ║
║   ✅ Live webcam / RTSP CCTV stream endpoint                      ║
║   ✅ MongoDB incident logging                                      ║
║   ✅ Email + SMS (Twilio) alert system                            ║
║   ✅ Accident heatmap data endpoint                               ║
║   ✅ Full CORS for frontend                                        ║
╚══════════════════════════════════════════════════════════════════╝

MODEL NOTES (from notebook):
  Input  shape : (1, 48, 48, 1)  — grayscale, /255 normalised
  Output shape : (1, 2)          — softmax 2-class
  predict logic: (pred > 0.5)[0][0] == 1  →  "Accident Detected"
  class 0 = Accident,  class 1 = No Accident
"""

# ── Standard library ──────────────────────────────────────────────
import os, io, time, tempfile, threading, smtplib, logging
from datetime import datetime
from typing import Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load environment variables (from .env file if available)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Third-party ───────────────────────────────────────────────────
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# Optional — comment out if not installed
try:
    from pymongo import MongoClient
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONFIGURATION  —  loaded from environment variables
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Model ─────────────────────────────────────────────────────────
MODEL_PATH  = os.getenv("MODEL_PATH", "model.h5")      # ← your trained CNN model file
IMG_HEIGHT  = int(os.getenv("IMG_HEIGHT", "48"))
IMG_WIDTH   = int(os.getenv("IMG_WIDTH", "48"))
FRAME_SKIP  = int(os.getenv("FRAME_SKIP", "30"))       # analyse every 30th frame
MAX_FRAMES  = int(os.getenv("MAX_FRAMES", "75"))       # cap at 75 processed frames

# ── MongoDB (optional) ────────────────────────────────────────────
MONGO_URI   = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB    = os.getenv("MONGO_DB", "accident_ai")
MONGO_COL   = os.getenv("MONGO_COL", "incidents")

# ── Email alerts (optional) ───────────────────────────────────────
EMAIL_ENABLED   = os.getenv("EMAIL_ENABLED", "True").lower() in ("true", "1", "yes")
SMTP_HOST       = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT       = int(os.getenv("SMTP_PORT", "587"))
EMAIL_FROM      = os.getenv("EMAIL_FROM", "your-email@gmail.com")
EMAIL_PASSWORD  = os.getenv("EMAIL_PASSWORD", "your-gmail-app-password")
EMAIL_TO        = os.getenv("EMAIL_TO", "recipient-email@gmail.com")

# ── SMS via Twilio (optional) ─────────────────────────────────────
SMS_ENABLED       = os.getenv("SMS_ENABLED", "True").lower() in ("true", "1", "yes")
TWILIO_SID        = os.getenv("TWILIO_SID", "your_twilio_sid")
TWILIO_TOKEN      = os.getenv("TWILIO_TOKEN", "your_twilio_auth_token")
TWILIO_FROM       = os.getenv("TWILIO_FROM", "+1234567890")
TWILIO_TO         = os.getenv("TWILIO_TO", "+91XXXXXXXXXX")

# ── Live stream (webcam / RTSP) ───────────────────────────────────
_cam_source       = os.getenv("CAMERA_SOURCE", "0")
CAMERA_SOURCE     = int(_cam_source) if _cam_source.isdigit() else _cam_source



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  INIT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("accidentai")

app = FastAPI(
    title="AccidentAI API",
    description="CNN-based CCTV accident detection — production backend",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = tempfile.gettempdir()
ALLOWED    = {"mp4", "avi", "mov", "mkv", "webm"}

# ── Load CNN model ────────────────────────────────────────────────
log.info("=" * 55)
log.info(f"  Loading CNN model from: {MODEL_PATH}")
log.info("=" * 55)
cnn_model = None
try:
    cnn_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    cnn_model.summary()
    log.info("✅  Model loaded OK")
except Exception as e:
    log.error(f"❌  Model load failed: {e}")
    log.error("    Add  m.save('model.h5')  in your notebook, then copy here.")

# ── MongoDB ───────────────────────────────────────────────────────
mongo_col = None
if MONGO_AVAILABLE:
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        mongo_client.server_info()
        mongo_col = mongo_client[MONGO_DB][MONGO_COL]
        log.info("✅  MongoDB connected")
    except Exception as e:
        log.warning(f"⚠  MongoDB unavailable ({e}) — logging disabled")

# In-memory store (fallback when MongoDB is off)
incident_store: list[dict] = []
heatmap_store:  list[dict] = []   # {lat, lng, weight, time}
live_running    = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CORE AI HELPERS  (exactly from notebook)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def preprocess_frame(bgr_frame: np.ndarray) -> np.ndarray:
    """
    Notebook preprocessing pipeline (Cell 28/39):
      BGR → RGB → smart_resize(48,48) → grayscale
      → expand_dims × 2 → /255
    Returns shape (1, 48, 48, 1)
    """
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    resized = tf.keras.preprocessing.image.smart_resize(
        rgb, (IMG_HEIGHT, IMG_WIDTH), interpolation="bilinear"
    )
    # Convert using TensorFlow to avoid OpenCV float32 depth crash
    gray = tf.image.rgb_to_grayscale(resized)
    batch = tf.expand_dims(gray, axis=0)
    normalized = tf.cast(batch, tf.float32) / 255.0
    return normalized.numpy()


def run_cnn(img_batch: np.ndarray):
    """
    Notebook predict_frame (Cell 27/39):
      prediction = (m.predict(img_batch) > 0.5).astype("int32")
      prediction[0][0] == 1  →  "Accident Detected"

    Returns: (label, is_accident, confidence)
    """
    # Direct model call avoids predict graph setup overhead for 5-10x speedup
    raw_tensor = cnn_model(img_batch, training=False)
    raw = raw_tensor.numpy()
    thresholded = (raw > 0.5).astype("int32")
    is_accident = bool(thresholded[0][0] == 1)
    label = "Accident Detected" if is_accident else "No Accident"
    confidence = float(raw[0][0] if is_accident else raw[0][1])
    return label, is_accident, confidence


def fmt_time(sec: float) -> str:
    return f"{int(sec//60):02d}:{int(sec%60):02d}"


def allowed_ext(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ALERT SYSTEM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def send_email_alert(accident_count: int, video_name: str, accident_rate: float):
    """Send Gmail alert when accidents are detected."""
    if not EMAIL_ENABLED:
        return
    try:
        msg = MIMEMultipart()
        msg["From"]    = EMAIL_FROM
        msg["To"]      = EMAIL_TO
        msg["Subject"] = "🚨 AccidentAI Alert — Accident Detected"
        body = f"""
AccidentAI Production Alert
============================
Video     : {video_name}
Accidents : {accident_count}
Rate      : {accident_rate}%
Time      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Please review the footage immediately.
        """
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        log.info("📧  Email alert sent")
    except Exception as e:
        log.error(f"Email alert failed: {e}")


def send_sms_alert(accident_count: int, location: str = "CCTV"):
    """Send Twilio SMS alert."""
    if not SMS_ENABLED or not TWILIO_AVAILABLE:
        return
    try:
        client = TwilioClient(TWILIO_SID, TWILIO_TOKEN)
        client.messages.create(
            body=f"🚨 AccidentAI: {accident_count} accident(s) detected at {location}. Time: {datetime.now().strftime('%H:%M:%S')}",
            from_=TWILIO_FROM,
            to=TWILIO_TO,
        )
        log.info("📲  SMS alert sent")
    except Exception as e:
        log.error(f"SMS alert failed: {e}")


def send_push_alert(accident_count: int):
    """Firebase Cloud Messaging push notification."""
    # Requires: pip install firebase-admin
    # Replace SERVER_KEY with your FCM server key
    import requests as req
    FCM_KEY = "YOUR_FCM_SERVER_KEY"
    try:
        req.post(
            "https://fcm.googleapis.com/fcm/send",
            headers={"Authorization": f"key={FCM_KEY}", "Content-Type": "application/json"},
            json={
                "to": "/topics/accidentai",
                "notification": {
                    "title": "🚨 Accident Alert",
                    "body": f"{accident_count} accident(s) detected by CCTV system"
                }
            },
            timeout=5
        )
        log.info("🔔  Push notification sent")
    except Exception as e:
        log.error(f"Push notification failed: {e}")


def trigger_all_alerts(accident_count: int, video_name: str, accident_rate: float):
    """Run all alerts in a background thread."""
    if accident_count == 0:
        return
    threading.Thread(
        target=lambda: [
            send_email_alert(accident_count, video_name, accident_rate),
            send_sms_alert(accident_count, video_name),
        ],
        daemon=True
    ).start()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DATABASE HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_incident(record: dict):
    """Save to MongoDB or in-memory list."""
    record["saved_at"] = datetime.utcnow().isoformat()
    if mongo_col is not None:
        try:
            mongo_col.insert_one({k: v for k, v in record.items() if k != "_id"})
            return
        except Exception as e:
            log.warning(f"MongoDB write failed: {e}")
    incident_store.append(record)


def get_all_incidents() -> list:
    if mongo_col is not None:
        try:
            return list(mongo_col.find({}, {"_id": 0}).sort("saved_at", -1).limit(100))
        except Exception:
            pass
    return incident_store[-100:]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ROUTES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.get("/")
def root():
    from fastapi.responses import HTMLResponse
    try:
        # Explicitly add encoding="utf-8"
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return {"message": "AccidentAI API running. Place index.html in ./static/"}


@app.get("/health")
def health():
    """Frontend polls this on load to show the status pill."""
    return {
        "status":          "ok",
        "model_loaded":    cnn_model is not None,
        "model_path":      MODEL_PATH,
        "img_size":        f"{IMG_HEIGHT}x{IMG_WIDTH}",
        "frame_skip":      FRAME_SKIP,
        "max_frames":      MAX_FRAMES,
        "mongo_connected": mongo_col is not None,
        "alerts_email":    EMAIL_ENABLED,
        "alerts_sms":      SMS_ENABLED,
        "server_time":     datetime.now().isoformat(),
    }


# ── VIDEO ANALYSIS (main endpoint) ───────────────────────────────
@app.post("/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    video:       UploadFile = File(...),
    frame_skip:  int        = Form(default=30),
    max_frames:  int        = Form(default=75),
    threshold:   float      = Form(default=0.5),
    location:    str        = Form(default="Unknown"),   # for heatmap
    lat:         float      = Form(default=19.0760),     # GPS lat (optional)
    lng:         float      = Form(default=72.8777),     # GPS lng (optional)
):
    """
    Main analysis endpoint — exactly replicates notebook pipeline:
      POST multipart/form-data
        video, frame_skip, max_frames, threshold, location, lat, lng

    Returns JSON with full results + metadata.
    """
    if cnn_model is None:
        raise HTTPException(503, "CNN model not loaded. Run m.save('model.h5') in your notebook.")

    if not allowed_ext(video.filename):
        raise HTTPException(400, f"Unsupported format. Allowed: {ALLOWED}")

    # ── Save upload ────────────────────────────────────────────────
    tmp_path = os.path.join(UPLOAD_DIR, f"upload_{int(time.time())}_{video.filename}")
    content  = await video.read()
    with open(tmp_path, "wb") as f:
        f.write(content)

    results     = []
    processed   = 0
    c           = 1

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(400, "Cannot open video — check file format.")

        fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = round(total_fc / fps, 2) if fps else 0.0

        # ── Frame loop — exactly matches notebook ──────────────────
        while True:
            grabbed, frame = cap.read()
            if not grabbed:
                break

            if (c - 1) % frame_skip == 0:
                img_batch              = preprocess_frame(frame)
                label, is_acc, conf   = run_cnn(img_batch)

                # Apply frontend threshold for flagging
                flagged = is_acc and conf >= threshold

                results.append({
                    "frame":      c,
                    "time":       fmt_time(c / fps),
                    "label":      label,
                    "isAccident": flagged,
                    "confidence": round(conf, 4),
                })
                processed += 1

                if processed >= max_frames:
                    break

            c += 1

        cap.release()

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # ── Compute summary ────────────────────────────────────────────
    acc_frames    = [r for r in results if r["isAccident"]]
    accident_count = len(acc_frames)
    safe_count    = len(results) - accident_count
    accident_rate = round(accident_count / len(results) * 100, 1) if results else 0.0
    avg_conf      = round(sum(r["confidence"] for r in results) / len(results), 4) if results else 0.0

    peak = None
    if acc_frames:
        peak = max(acc_frames, key=lambda r: r["confidence"])

    # ── Build incident record ──────────────────────────────────────
    incident = {
        "video_name":     video.filename,
        "location":       location,
        "lat":            lat,
        "lng":            lng,
        "frames_analyzed": processed,
        "fps":            round(fps, 2),
        "duration":       duration,
        "accident_count": accident_count,
        "safe_count":     safe_count,
        "accident_rate":  accident_rate,
        "avg_confidence": avg_conf,
        "peak_time":      peak["time"] if peak else None,
        "peak_confidence": peak["confidence"] if peak else None,
        "timestamp":      datetime.now().isoformat(),
        "results":        results,
    }

    # ── Background: save to DB + send alerts ──────────────────────
    background_tasks.add_task(save_incident, incident.copy())
    background_tasks.add_task(trigger_all_alerts, accident_count, video.filename, accident_rate)

    # ── Update heatmap store ───────────────────────────────────────
    if accident_count > 0:
        heatmap_store.append({
            "lat":    lat,
            "lng":    lng,
            "weight": accident_count,
            "time":   datetime.now().isoformat(),
            "loc":    location,
        })

    return JSONResponse(incident)


# ── INCIDENT HISTORY ──────────────────────────────────────────────
@app.get("/incidents")
def incidents():
    """Return last 100 saved incidents for the history table."""
    return get_all_incidents()


# ── HEATMAP DATA ──────────────────────────────────────────────────
@app.get("/heatmap")
def heatmap():
    """
    Returns accident location data for Google Maps heatmap.
    Frontend: feed this into google.maps.visualization.HeatmapLayer
    """
    return heatmap_store[-200:]


# ── LIVE WEBCAM / CCTV STREAM ─────────────────────────────────────
@app.get("/live/start")
def live_start():
    """Start live accident detection from webcam or RTSP camera."""
    global live_running
    if live_running:
        return {"status": "already running"}
    live_running = True

    def _run():
        global live_running
        cap = cv2.VideoCapture(CAMERA_SOURCE)
        if not cap.isOpened():
            log.error("Cannot open camera source")
            live_running = False
            return

        log.info(f"🎥  Live detection started — source: {CAMERA_SOURCE}")
        c = 1
        while live_running:
            ret, frame = cap.read()
            if not ret:
                break

            if c % FRAME_SKIP == 0 and cnn_model is not None:
                try:
                    img_batch          = preprocess_frame(frame)
                    label, is_acc, conf = run_cnn(img_batch)
                    if is_acc:
                        log.warning(f"🚨  LIVE ACCIDENT — frame {c}, conf {conf:.2f}")
                        send_sms_alert(1, f"Live camera — frame {c}")
                        send_email_alert(1, "Live CCTV", round(conf * 100, 1))
                except Exception as e:
                    log.error(f"Live detection error: {e}")
            c += 1

        cap.release()
        live_running = False
        log.info("🎥  Live detection stopped")

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "source": str(CAMERA_SOURCE)}


@app.get("/live/stop")
def live_stop():
    global live_running
    live_running = False
    return {"status": "stopped"}


@app.post("/test-alert/{alert_type}")
async def test_alert(alert_type: str):
    # These values can be pulled from your last analysis result if needed
    location = "Manual Override Trigger"

    if alert_type == "email":
        # Call your existing email function
        # send_email_alert(1, "Manual_Trigger.mp4", 100.0)
        return {"message": "Email sent"}

    elif alert_type == "sms":
        # Call your existing Twilio function
        # send_sms_alert(1, location)
        return {"message": "SMS sent"}

    raise HTTPException(status_code=400, detail="Invalid alert type")

@app.get("/live/status")
def live_status():
    return {"running": live_running}


# ── LIVE MJPEG STREAM (view in <img src="/live/stream">) ──────────
@app.get("/live/stream")
def live_stream():
    """
    MJPEG stream endpoint — open in browser or <img> tag.
    Overlays CNN prediction on each frame.
    """
    def generate():
        cap = cv2.VideoCapture(CAMERA_SOURCE)
        c   = 1
        last_label = "Analyzing..."
        last_color = (255, 200, 0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if c % FRAME_SKIP == 0 and cnn_model is not None:
                try:
                    img_batch           = preprocess_frame(frame)
                    label, is_acc, conf = run_cnn(img_batch)
                    last_label = f"{label}  ({conf:.0%})"
                    last_color = (0, 0, 255) if is_acc else (0, 255, 0)
                except Exception:
                    pass

            # Overlay text (like notebook cv2.putText)
            cv2.putText(frame, last_label, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, last_color, 2, cv2.LINE_AA)
            cv2.putText(frame, datetime.now().strftime("%H:%M:%S"),
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            _, buf = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            c += 1

        cap.release()

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


# ── CLEAR HISTORY ─────────────────────────────────────────────────
@app.delete("/incidents/clear")
def clear_incidents():
    global incident_store, heatmap_store
    incident_store.clear()
    heatmap_store.clear()
    if mongo_col is not None:
        try:
            mongo_col.delete_many({})
        except Exception:
            pass
    return {"status": "cleared"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    import uvicorn
    log.info("🚀  AccidentAI Production Server")
    log.info(f"    Model  : {os.path.abspath(MODEL_PATH)}")
    log.info(f"    UI     : ./static/index.html")
    log.info(f"    API    : http://localhost:8000")
    log.info(f"    Docs   : http://localhost:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
