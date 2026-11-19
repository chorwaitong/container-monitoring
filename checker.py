import os
from config import SCRIPT_DIR, RECORDING_DIR, RTSP_URL

def check():
    ffmpeg_path = os.path.join(SCRIPT_DIR, "ffmpeg.exe")
    if not os.path.isfile(ffmpeg_path):
        print("[!!!!!!!!!!!!!!]  File ffmpeg.exe does not exist exists in the script's directory, please get it...")
        return False

    os.makedirs(RECORDING_DIR, exist_ok=True)
    if not os.path.isdir(RECORDING_DIR):
        print(f"[!!!!!!!!!!!!!!] The recording folder {RECORDING_DIR} doesnt exist nor can be created, please work it out manually...")
        return False

    if not RTSP_URL:
        print("[!!!!!!!!!!!!!!] Environment variable 'ContainerMonitorRTSP' is not set or is empty.")
        print("Please set the environment variable before running")
        return False

    return True