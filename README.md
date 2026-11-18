#Installations

```bash
# 1. Create a minimal Python 3.10 environment
conda create -n container_monitor python=3.10 -y

# 2. Activate the environment
conda activate container_monitor

# 3. Install dependencies
pip install opencv-python
pip install ultralytics openvino #If using yolo
yolo export model=yolov8n.pt format=openvino
```

Also, need the followings:
1. ffmpeg.exe to be placed in the root directory.
2. Environment variable of RTSP_URL to be set.