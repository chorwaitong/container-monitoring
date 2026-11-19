from config import ZONE_START_RATIO, ZONE_END_RATIO
import numpy as np
import os
import cv2
import traceback

# Extraction Optimization Tuning
MOTION_COOLDOWN_SECONDS = 1 # Stop early if no motion is detected for 1s

# Masking params
MASK_ZONE_START_RATIO = 0.2
MASK_ZONE_END_RATIO = 0.7

def crop_container(frame):
    """
    Instantly crops out the areas beyond the start and end zones,
    preserving the original color of the container inside the zone.
    """
    h, w = frame.shape[:2]
    
    # Calculate pixel boundaries based on your zone ratios
    x1 = int(w * MASK_ZONE_START_RATIO)
    x2 = int(w * MASK_ZONE_END_RATIO)
    
    # Crop horizontally to keep ONLY the active zone (original colors intact)
    roi = frame[:, x1:x2]
    
    return roi


def detect_vertical_motion_blur(
    frame,
    lap_threshold=0.002,
    direction_threshold=0.7,
    min_texture=0.0001
):
    """
    Detect predominantly vertical motion blur.

    Returns
    -------
    is_blurred : bool
    metrics : dict
    """

    if frame is None or frame.size == 0:
        return False, {}

    # Convert to normalized grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0

    # Exclude heavily saturated and very dark pixels
    valid = (gray > 0.04) & (gray < 0.96)

    if np.count_nonzero(valid) < 100:
        return False, {"reason": "Insufficient valid pixels"}

    # Laplacian sharpness
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap_var = float(np.var(lap[valid]))

    # Directional gradient energy
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    ex = float(np.mean(gx[valid] ** 2))
    ey = float(np.mean(gy[valid] ** 2))

    texture = ex + ey

    vertical_ratio = ey / (ex + 1e-8)

    # Detect insufficient sharpness and directional attenuation
    low_sharpness = lap_var < lap_threshold
    vertical_smearing = vertical_ratio < direction_threshold

    is_blurred = (
        (low_sharpness
        or vertical_smearing)
        and texture > min_texture
    )

    metrics = {
        "laplacian_variance": lap_var,
        "horizontal_energy": ex,
        "vertical_energy": ey,
        "vertical_ratio": vertical_ratio,
        "texture_energy": texture,
        "low_sharpness": low_sharpness,
        "vertical_smearing": vertical_smearing,
    }

    return is_blurred, metrics



def deblur_frame(frame, blur_length=81, nsr=0.015, strength=0.85):
    """
    Vertical motion deblurring using 1D Wiener deconvolution.

    Parameters
    ----------
    frame : np.ndarray
        Input BGR image.

    blur_length : int
        Estimated vertical motion blur length in pixels.

    nsr : float
        Wiener regularisation parameter.
        Increase to suppress noise and ringing.
        Decrease for stronger deconvolution.

    strength : float
        Blending factor between original and restored luminance.
        0.0 = original image.
        1.0 = fully restored image.

    Returns
    -------
    np.ndarray
        Deblurred BGR image.
    """

    if frame is None or frame.size == 0:
        return frame

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected a BGR image.")


    is_blurred, metrics = detect_vertical_motion_blur(
        frame,
        lap_threshold=0.002,
        direction_threshold=0.85
    )


    # Stage 2: Skip if image is already sharp
    if not is_blurred:
        print (f"Clear - blur metrics: {metrics}")
        return frame
    else:
        print (f"Image is blur, blur metrics: {metrics}")
    

    h, w = frame.shape[:2]

    blur_length = max(3, int(blur_length))
    if blur_length % 2 == 0:
        blur_length += 1

    # Work on luminance only
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    y = ycrcb[:, :, 0].astype(np.float32) / 255.0

    # Reflective padding prevents circular FFT boundary artefacts
    pad = max(blur_length * 2, 64)

    y_pad = cv2.copyMakeBorder(
        y,
        pad, pad,
        0, 0,
        cv2.BORDER_REFLECT
    )

    H, W = y_pad.shape

    # --------------------------------------------------
    # 1. Construct vertical motion PSF
    # --------------------------------------------------

    psf = np.zeros(H, dtype=np.float32)

    center = H // 2
    half = blur_length // 2

    psf[center - half:center + half + 1] = 1.0

    psf /= psf.sum()

    # Move PSF origin to FFT origin
    psf = np.roll(psf, -center)

    # --------------------------------------------------
    # 2. Compute vertical frequency-domain transforms
    # --------------------------------------------------

    Y = np.fft.rfft(y_pad, axis=0)

    PSF = np.fft.rfft(psf)

    # --------------------------------------------------
    # 3. Wiener deconvolution
    # --------------------------------------------------

    denominator = (
        np.abs(PSF) ** 2 + nsr
    )

    wiener_filter = (
        np.conj(PSF) / denominator
    )

    restored_fft = (
        Y * wiener_filter[:, None]
    )

    restored = np.fft.irfft(
        restored_fft,
        n=H,
        axis=0
    )

    # --------------------------------------------------
    # 4. Remove padding
    # --------------------------------------------------

    restored = restored[pad:pad + h, :]

    restored = np.clip(restored, 0.0, 1.0)

    # --------------------------------------------------
    # 5. Blend with original image
    # --------------------------------------------------

    output_y = (
        (1.0 - strength) * y +
        strength * restored
    )

    ycrcb[:, :, 0] = np.clip(
        output_y * 255.0,
        0,
        255
    ).astype(np.uint8)

    return cv2.cvtColor(
        ycrcb,
        cv2.COLOR_YCrCb2BGR
    )


def extract_frames_from_mkv(mkv_path, output_dir, inference_width = 320, skip_frame = 1):
    """Extracts optimized frames from a single .mkv file with skipping and smart early-stopping."""
    hasProcessedSomething = False
    try:
        os.makedirs(output_dir, exist_ok=True)

        if any(f.lower().endswith('.png') for f in os.listdir(output_dir)):
            print(f"[!] Extraction skipped {output_dir} as it already contains png files.")
            return hasProcessedSomething

        print(f"\n[*] Extraction starts: {os.path.basename(mkv_path)}")
        cap = cv2.VideoCapture(mkv_path)
        
        if not cap.isOpened():
            print(f"[X] Could not open video file: {mkv_path}")
            return hasProcessedSomething

        fps = cap.get(cv2.CAP_PROP_FPS)   
        motion_cooldown_frames = int(fps*MOTION_COOLDOWN_SECONDS/skip_frame) #e.g., (FPS 20 x 1s cool down)/Skip frame 3 = 7 frames
        frame_count = 0
        saved_count = 0
        consecutive_no_motion = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of file
                
            frame_count += 1
            # --- FRAME SKIPPING MECHANISM ---
            if frame_count % (skip_frame + 1) != 0:
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
            #  Usually with container passing by, the mean_diff value is 10-30
            if mean_diff < 3: 
                consecutive_no_motion += 1
            else:
                consecutive_no_motion = 0  # Reset counter if motion/truck is still present

            prev_gray_zone = gray_zone.copy()

            # print(f"[*] Frame {saved_count} - mean_diff {mean_diff}")
            
            if consecutive_no_motion >= motion_cooldown_frames:
                print(f"\n[*] Truck has cleared the view at frame {frame_count} for {motion_cooldown_frames}. Ending extraction early.")
                break

            cropped_frame = crop_container(frame)
            # processed_frame = deblur_frame(cropped_frame, 81, 0.010, 0.95)
            processed_frame = cropped_frame

            saved_count += 1
            frame_filename = f"frame_{saved_count:04d}.png"
            frame_path = os.path.join(output_dir, frame_filename)
            
            
            cv2.imwrite(frame_path, processed_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            print(f"Extracted useful frame {saved_count} (Source frame: {frame_count})...")
            hasProcessedSomething = True

        cap.release()
        print(f"\n[/] Saved {saved_count} frames (skipped redundant ones) to: {output_dir}")
    except Exception as e:
        print(f"Exception encountered while extracting frames for {mkv_path}:{e} ")
        traceback.print_exc()
    finally:
        return hasProcessedSomething

