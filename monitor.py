import os
import time
import sys
from datetime import datetime
import cv2
import threading
# from ultralytics import YOLO

# ================= CONFIGURATION =================
RTSP_URL = os.environ.get("ContainerMonitorRTSP", "").strip()
if not RTSP_URL:
    print("Error: Environment variable 'ContainerMonitorRTSP' is not set or is empty.")
    print("Please set it before running")
    sys.exit(1)

RECORDING_DIR = r"G:\\My Drive\\Projects - Container Monitoring\\Recordings\\"

# Performance & Detection Tuning
CONFIDENCE_THRESHOLD = 0.2   # Minimum confidence to trigger detection since it is at top view
INFERENCE_WIDTH = 640         # Downscale width for AI inference (lowers CPU load)
TARGET_INFERENCE_FPS = 2.0    # Process only 2 frames per second
POST_BUFFER_SECONDS = 3.0     # Keep recording for N seconds after truck leaves view
MANUAL_CLIP_SECONDS = 10.0
TARGET_CLASSES = [5, 6, 7, 8]         # Anything related, just detect
STARTUP_WARMUP_SECONDS = 2.0

# Motion trigger if a huge object (>12% of screen area) passes under top-view camera, within the center 50% of the video.
MIN_MOTION_AREA_RATIO = 0.12
CENTER_ZONE_RATIO = 0.50
# =================================================

# Force RTSP over TCP for reliable streaming without artifacts
# Increase timeout to 10 seconds (in microseconds) and add reorder queue for internet jitter
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|"
    "stimeout;10000000|"
    "max_delay;500000"
)
os.makedirs(RECORDING_DIR, exist_ok=True)

class RTSPStream:
    """Continuously reads frames in a background thread so AI inference never blocks the network stream."""
    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        self.ret, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = True
                    self.frame = frame
            else:
                with self.lock:
                    self.ret = False
                # Reconnect silently in background
                self.cap.release()
                time.sleep(2)
                self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def isOpened(self):
        return self.cap.isOpened()

    def get(self, prop):
        return self.cap.get(prop)

    def release(self):
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()

def connect_stream(url):
    """Connects to the RTSP stream with minimal buffer size to reduce latency."""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    return cap

# Clickable button area (top-right / below status) & click state
BTN_RECT = (20, 60, 230, 100)  # (x1, y1, x2, y2) on the preview window
manual_trigger_requested = False


def on_mouse_click(event, x, y, flags, param):
    """Triggers a 10s manual recording when the user clicks the on-screen button."""
    global manual_trigger_requested
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1, x2, y2 = BTN_RECT
        if x1 <= x <= x2 and y1 <= y <= y2:
            manual_trigger_requested = True

def main():
    # print("Loading lightweight YOLOv8-Nano model...")
    # model = YOLO("yolov8n_openvino_model/", task="detect")  # or "yolov8n.onnx"
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=50, detectShadows=False)

    # cap = connect_stream(RTSP_URL)
    cap = RTSPStream(RTSP_URL)
    if not cap.isOpened():
        print("Error: Unable to connect to RTSP stream. Check network/credentials.")
        return

    # Retrieve original stream properties
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 60:
        fps = 20.0

    aspect_ratio = orig_h / orig_w
    inf_h = int(INFERENCE_WIDTH * aspect_ratio)
    inference_interval = 1.0 / TARGET_INFERENCE_FPS  # 0.5s between checks

    print(f"Connected! Stream: {orig_w}x{orig_h} @ {fps:.1f} FPS | AI Rate: {TARGET_INFERENCE_FPS} FPS")
    print("Press 'q' in the live view window to stop.")

    # State variables
    global manual_trigger_requested
    video_writer = None
    is_recording = False
    last_detection_time = 0
    last_inference_time = 0 
    manual_record_until = 0
    stream_start_time = time.time()
    last_detections = []

    cv2.namedWindow("Container CCTV Monitor")
    cv2.setMouseCallback("Container CCTV Monitor", on_mouse_click)
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Stream interrupted. Reconnecting in 3 seconds...")
            cap.release()
            time.sleep(3)
            cap = connect_stream(RTSP_URL)
            continue

        current_time = time.time()

        # 1. RUN LIGHTWEIGHT DETECTION AT 2 FPS (Every 0.5 seconds)
        if (current_time - last_inference_time) >= inference_interval:
            last_inference_time = current_time
            small_frame = cv2.resize(frame, (INFERENCE_WIDTH, inf_h))

            # # Leave out yolo implementation but keep it here in case future...
            # results = model(
            #     small_frame,
            #     classes=TARGET_CLASSES,
            #     conf=CONFIDENCE_THRESHOLD,
            #     verbose=False
            # )
            # for box in results[0].boxes:
            #     x1, y1, x2, y2 = box.xyxy[0].tolist()
            #     x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            #     y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
            #     conf = float(box.conf[0])
            #     last_detections.append((x1, y1, x2, y2, conf, "Container/Truck"))

            last_detections = []
            scale_x = orig_w / INFERENCE_WIDTH
            scale_y = orig_h / inf_h

            fg_mask = bg_subtractor.apply(small_frame)
            if (current_time - stream_start_time) >= STARTUP_WARMUP_SECONDS:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # The middle 50% zone boundaries in inference coordinates
                margin_x = (1.0 - CENTER_ZONE_RATIO) / 2.0  # 0.25 (25% margin on left/right)
                zone_x1 = INFERENCE_WIDTH * margin_x
                zone_x2 = INFERENCE_WIDTH * (1.0 - margin_x)

                min_pixels = (INFERENCE_WIDTH * inf_h) * MIN_MOTION_AREA_RATIO
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area >= min_pixels:
                        mx, my, mw, mh = cv2.boundingRect(cnt)

                        # Calculate center of the detected object
                        cx = mx + (mw / 2.0)
                        cy = my + (mh / 2.0)

                        # Only trigger if the object's center falls inside the 50% center zone
                        if zone_x1 <= cx <= zone_x2:
                            x1, y1 = int(mx * scale_x), int(my * scale_y)
                            x2, y2 = int((mx + mw) * scale_x), int((my + mh) * scale_y)
                            coverage_pct = (area / (INFERENCE_WIDTH * inf_h)) * 100
                            last_detections.append((x1, y1, x2, y2, coverage_pct, "Container (Center)"))

            if len(last_detections) > 0:
                last_detection_time = current_time

        # 2. HANDLE ORIGINAL-RESOLUTION RECORDING
        if manual_trigger_requested:
            manual_record_until = current_time + MANUAL_CLIP_SECONDS
            manual_trigger_requested = False
            print(f"[MANUAL REC] Recording {int(MANUAL_CLIP_SECONDS)}s clip...")

        auto_record = (current_time - last_detection_time) <= POST_BUFFER_SECONDS
        manual_record = current_time <= manual_record_until
        should_record = auto_record or manual_record  

        if should_record and not is_recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = "manual" if manual_record else "container"
            filename = os.path.join(RECORDING_DIR, f"{prefix}_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"avc1") 
            video_writer = cv2.VideoWriter(filename, fourcc, fps, (orig_w, orig_h))
            is_recording = True
            print(f"[REC START] {filename}")

        elif not should_record and is_recording:
            is_recording = False
            video_writer.release()
            video_writer = None
            print("[REC SAVED] Clip finalized.")

        if is_recording and video_writer is not None:
            video_writer.write(frame)

        # 3. RENDER LIVE VIEW WITH OVERLAYS
        display_frame = frame.copy()

        for (x1, y1, x2, y2, conf, label) in last_detections:
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} ({conf:.0f}% view)"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(display_frame, (x1, y1 - th - 10), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(display_frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        status_text = "STATUS: RECORDING" if is_recording else "STATUS: MONITORING"
        status_color = (0, 0, 255) if is_recording else (0, 255, 0)
        cv2.putText(display_frame, status_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

        if orig_w > 1280:
            preview = cv2.resize(display_frame, (1280, int(1280 * aspect_ratio)))
        else:
            preview = display_frame

        bx1, by1, bx2, by2 = BTN_RECT
        if manual_record:
            remaining = max(0.0, manual_record_until - current_time)
            btn_color = (0, 0, 200)
            btn_label = f"REC: {remaining:.1f}s left"
        else:
            btn_color = (200, 100, 0)
            btn_label = "REC 10s CLIP [R]"

        cv2.rectangle(preview, (bx1, by1), (bx2, by2), btn_color, -1)
        cv2.rectangle(preview, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
        cv2.putText(preview, btn_label, (bx1 + 12, by1 + 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw subtle guide rectangle for the active 50% center detection zone
        prev_h, prev_w = preview.shape[:2]
        zx1 = int(prev_w * ((1.0 - CENTER_ZONE_RATIO) / 2.0))
        zx2 = int(prev_w * (1.0 - (1.0 - CENTER_ZONE_RATIO) / 2.0))
        cv2.line(preview, (zx1, 0), (zx1, prev_h), (0, 255, 255), 1)
        cv2.line(preview, (zx2, 0), (zx2, prev_h), (0, 255, 255), 1)
        cv2.putText(preview, "ACTIVE ZONE (50% WIDTH)", (zx1 + 8, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("Container CCTV Monitor", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):                   
            manual_trigger_requested = True

    if video_writer is not None:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print("Monitoring stopped cleanly.")


if __name__ == "__main__":
    main()