import os


# ====== Paths and urls =============
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
RECORDING_DIR = r"G:\\My Drive\\Projects - Container Monitoring\\Recordings\\"
RTSP_URL = os.environ.get("ContainerMonitorRTSP", "").strip()


# ====== Common image settings =======
MIN_MOTION_AREA_RATIO = 0.2         # Motion trigger if a huge object (>X% of screen area) passes under top-view camera, within the center 50% of the video.
ZONE_START_RATIO = 0.30             # Left boundary at 30% of width
ZONE_END_RATIO = 0.70               # Right boundary at 70% of width


HIGH_RES  = True