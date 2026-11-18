from monitor import RECORDING_DIR, ZONE_START_RATIO, ZONE_END_RATIO, MIN_MOTION_AREA_RATIO
import os
import cv2

# Extraction Optimization Tuning
SKIP_FRAMES = 3          # 0 = save every frame, n = save every n-th frame (skip 1), 2 = save every 3rd frame, etc.
MAX_USEFUL_RATIO = 0.6   # Stop extraction after 60% of the video duration (skipping the empty tail-end)
MOTION_COOLDOWN_SECONDS = 1 # Stop early if no motion is detected for 1s

def extract_frames_from_mkv(mkv_path, output_base_dir):
    """Extracts optimized frames from a single .mkv file with skipping and smart early-stopping."""
    base_name = os.path.splitext(os.path.basename(mkv_path))[0]
    output_dir = os.path.join(output_base_dir, base_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[*] Opening: {os.path.basename(mkv_path)}")
    cap = cv2.VideoCapture(mkv_path)
    
    if not cap.isOpened():
        print(f"[X] Could not open video file: {mkv_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)   
    motion_cooldown_frames = int(fps*MOTION_COOLDOWN_SECONDS/SKIP_FRAMES) #e.g., (FPS 20 x 1s cool down)/Skip frame 3 = 7 frames
    frame_count = 0
    saved_count = 0
    consecutive_no_motion = 0
    inference_width = 320
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of file
            
        frame_count += 1
        # --- FRAME SKIPPING MECHANISM ---
        if frame_count % (SKIP_FRAMES + 1) != 0:
            continue

        # --- SMART EARLY STOPPING CHECK ---
        # Check if there's a continuous 1s of still image within the focus zone.
        
        inference_height = int(inference_width * (frame.shape[0] / frame.shape[1]))
        small_frame = cv2.resize(frame, (inference_width, inference_height))    
        zx1 = int(inference_width * ZONE_START_RATIO)
        zx2 = int(inference_width * ZONE_END_RATIO)

        zone_crop = small_frame[:, zx1:zx2]
        gray_zone = cv2.cvtColor(zone_crop, cv2.COLOR_BGR2GRAY)

        # Initialize previous frame tracker for comparison on the first iteration
        if 'prev_gray_zone' not in locals():
            prev_gray_zone = gray_zone

        # Compute frame-to-frame absolute difference within the zone
        diff = cv2.absdiff(gray_zone, prev_gray_zone)
        mean_diff = cv2.mean(diff)[0]

        # If change is near zero (accounting for minor camera sensor noise), it's a still image/empty road
        if mean_diff < 3:
            consecutive_no_motion += 1
        else:
            consecutive_no_motion = 0  # Reset counter if motion/truck is still present

        prev_gray_zone = gray_zone.copy()

        print(f"[*] Frame {saved_count} - mean_diff {mean_diff}")
        
        if consecutive_no_motion >= motion_cooldown_frames:
            print(f"\n[*] Truck has cleared the view at frame {frame_count} for {motion_cooldown_frames}. Ending extraction early.")
            break

        saved_count += 1
        frame_filename = f"{base_name}_frame_{saved_count:04d}.png"
        frame_path = os.path.join(output_dir, frame_filename)
        
        cv2.imwrite(frame_path, frame)
        print(f"Extracted useful frame {saved_count} (Source frame: {frame_count})...", end="\r")

    cap.release()
    print(f"\n[/] Saved {saved_count} frames (skipped redundant ones) to: {output_dir}")
    
def main():
    if not os.path.exists(RECORDING_DIR):
        print(f"[X] Recording directory does not exist: {RECORDING_DIR}")
        return

    # Scan directory for all .mkv files
    mkv_files = [os.path.join(RECORDING_DIR, f) for f in os.listdir(RECORDING_DIR) if f.lower().endswith('.mkv')]
    
    if not mkv_files:
        print(f"[!] No .mkv files found in {RECORDING_DIR}")
        return

    print(f"Found {len(mkv_files)} MKV recording(s). Starting extraction...")

    for mkv_path in mkv_files:
        extract_frames_from_mkv(mkv_path, RECORDING_DIR)
        break

    print("\n[/] All extractions completed!")

if __name__ == "__main__":
    main()