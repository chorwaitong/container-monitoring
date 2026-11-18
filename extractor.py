from monitor import RECORDING_DIR, ZONE_START_RATIO, ZONE_END_RATIO, MIN_MOTION_AREA_RATIO
import numpy as np
import os
import cv2
import traceback

# Extraction Optimization Tuning
SKIP_FRAMES = 3          # 0 = save every frame, n = save every n-th frame (skip 1), 2 = save every 3rd frame, etc.
MAX_USEFUL_RATIO = 0.6   # Stop extraction after 60% of the video duration (skipping the empty tail-end)
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

def deblur_frame(frame):
    """
    Applies a lightweight Unsharp Mask to counteract motion blur 
    optimized for low-CPU overhead.
    """
    # Create a Gaussian blur of the image
    gaussian = cv2.GaussianBlur(frame, (0, 0), sigmaX=2.0)
    
    # Unsharp Mask formula: Original + (Original - Blurred) * weight
    # This sharpens edges blurred by truck movement
    sharpened = cv2.addWeighted(frame, 1.5, gaussian, -0.5, 0)
    
    return sharpened

def extract_frames_from_mkv(mkv_path, output_base_dir):
    """Extracts optimized frames from a single .mkv file with skipping and smart early-stopping."""
    hasProcessedSomething = False
    try:
        base_name = os.path.splitext(os.path.basename(mkv_path))[0]
        output_dir = os.path.join(output_base_dir, base_name)
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
            processed_frame = deblur_frame(cropped_frame)

            saved_count += 1
            frame_filename = f"frame_{saved_count:04d}.png"
            frame_path = os.path.join(output_dir, frame_filename)
            
            
            cv2.imwrite(frame_path, processed_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            print(f"Extracted useful frame {saved_count} (Source frame: {frame_count})...", end="\r")
            hasProcessedSomething = True

        cap.release()
        print(f"\n[/] Saved {saved_count} frames (skipped redundant ones) to: {output_dir}")
    except Exception as e:
        print(f"Exception encountered while extracting frames for {mkv_path}:{e} ")
        traceback.print_exc()
    finally:
        return hasProcessedSomething

def stitch_frames(mkv_path, output_base_dir):
    """
    Stitches extracted frames into a single composite image using a static background 
    reference derived from the last 5 frames, saving step-by-step debug outputs per frame.
    """
    hasProcessedSomething = False
    base_name = os.path.splitext(os.path.basename(mkv_path))[0]
    output_dir = os.path.join(output_base_dir, base_name)
    output_path = os.path.join(output_dir, "stitched.png")
    try:        
        if os.path.isfile(output_path):
            print(f"[!] Stitching skipped for {output_dir} as the stitched.png is found")
            return hasProcessedSomething

        # Filter out any pre-existing debug or stitched files so we only read source frames
        frame_files = sorted([
            os.path.join(output_dir, f) for f in os.listdir(output_dir) 
            if f.endswith('.png') and 'frame_' in f and not '_diff' in f and not '_cleanmask' in f and not '_mask' in f and not '_blended' in f and not 'debug_background_plate' in f
        ])

        if len(frame_files) < 10:
            print(f"[X] Not enough frames to perform stitching (found {len(frame_files)}).")
            return None

        print(f"\n[*] Stitching {len(frame_files)} frames with frame-by-frame debug export...")
        
        # Read sample frame to get dimensions
        first_frame = cv2.imread(frame_files[0])
        h, w = first_frame.shape[:2]

        # Start canvas with the first frame
        canvas = first_frame.copy()
        current_y_offset = 0

        # 1. Build pristine background plate from the last 5 frames
        last_few_paths = frame_files[-5:]
        last_few_frames = [cv2.imread(f) for f in last_few_paths]
        background_plate = np.median(np.array(last_few_frames), axis=0).astype(np.uint8)
        bg_float = background_plate.astype(np.float32) + 1e-6
        bg_sum = np.sum(bg_float, axis=2, keepdims=True)
        bg_norm = bg_float / bg_sum

        # # Save background plate reference for inspection
        cv2.imwrite(os.path.join(output_dir, "debug_background_plate.png"), background_plate)

        # Exclude the last 5 background reference frames from being stitched as moving truck frames
        truck_frame_files = frame_files[:-5]
        
        # # 2. Automatically calculate required canvas height based on active truck frame count
        # total_frames = len(truck_frame_files)
        # travel_distance = int(h * 1.2) # Total vertical pixels the truck travels
        # step_size = travel_distance / max(1, total_frames - 1)
        
        # start_y = int(h * 0.5)
        # canvas_h = int(start_y + travel_distance + h + (h * 0.5)) 

        # # Define structural kernels for morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))

        # # Initialize dynamic canvas
        # canvas = np.tile(background_plate[0:1, :, :], (canvas_h, 1, 1))

        # Start canvas with the first frame
        canvas = first_frame.copy()
        # canvas_h = h * len(truck_frame_files) # Safe max height allocation
        # We can use a dynamically growing canvas or pre-allocate a tall canvas and crop later
        # canvas = np.tile(background_plate[0:1, :, :], (canvas_h, 1, 1))
        # Track previous matched position to maintain smooth downward progression
        last_y = int(h * 0.2)

        # Initialize ORB detector (Fast and lightweight for i3 CPU)
        orb = cv2.ORB_create(nfeatures=500)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        for i in range(len(truck_frame_files) - 1):
            curr_path = truck_frame_files[i]
            next_path = truck_frame_files[i + 1]
            
            curr_img = cv2.imread(curr_path)
            next_img = cv2.imread(next_path)
            
            if curr_img is None or next_img is None:
                continue            
            
            # Detect ORB keypoints and descriptors
            kp1, des1 = orb.detectAndCompute(curr_img, None)
            kp2, des2 = orb.detectAndCompute(next_img, None)
            
            if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
                # Fallback safety shift if texture is momentarily poor
                print(f"[!] Frame {i} Cant detect ORB keypoints")
                shift = int(h * 0.05)
            else:
                # Match descriptors using KNN matcher
                matches = bf.knnMatch(des1, des2, k=2)
                
                # Apply Lowe's ratio test to filter out bad matches
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
                
                if len(good_matches) > 10:
                    pts_curr = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    pts_next = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    
                    # Use RANSAC to filter good matches
                    _, mask_ransac = cv2.findHomography(pts_curr, pts_next, cv2.RANSAC, 5.0)
                    
                    if mask_ransac is not None:
                        inlier_curr = pts_curr[mask_ransac.ravel() == 1].reshape(-1, 2)
                        inlier_next = pts_next[mask_ransac.ravel() == 1].reshape(-1, 2)
                        
                        if len(inlier_curr) > 3:
                            dy_diffs = inlier_next[:, 1] - inlier_curr[:, 1]
                            
                            # Widen valid range to catch any natural truck speed (e.g. 1 to h*0.25)
                            valid_dys = dy_diffs[(dy_diffs > 0.5) & (dy_diffs < int(h * 0.25))]
                            
                            if len(valid_dys) > 1:
                                shift = int(np.median(valid_dys))
                                print(f"[*] Frame {i} RANSAC shift calculated: {shift}")
                            else:
                                shift = int(h * 0.02) # Lower fallback
                                print(f"[!] Frame {i} low valid dys, using fallback: {shift}")
                        else:
                            shift = int(h * 0.02)
                            print(f"[!] Frame {i} low inliers, using fallback: {shift}")
                    else:
                        shift = int(h * 0.02)
                        print(f"[!] Frame {i} RANSAC failed, using fallback: {shift}")
                else:
                    shift = int(h * 0.02)
                    print(f"[!] Frame {i} low good matches, using fallback: {shift}")
                
                # Final safety boundaries
                shift = max(2, min(shift, int(h * 0.25)))
                

        
            # Dynamically expand canvas downward by the measured feature shift
            canvas = np.vstack([canvas, next_img[-shift:, :, :]])
            
            # Save step output for debugging
            frame_id = f"{i+2:04d}"
            cv2.imwrite(os.path.join(output_dir, f"frame_{frame_id}_blended.png"), canvas)

        # for i, f_path in enumerate(truck_frame_files):            
        #     frame = cv2.imread(f_path)
        #     if frame is None:
        #         continue
                
        #     frame_id = f"{i+1:04d}"
            
        #     # Step A: Shadow-Invariant Chrominance Difference
        #     f_float = frame.astype(np.float32) + 1e-6            
        #     f_sum = np.sum(f_float, axis=2, keepdims=True)
        #     f_norm = f_float / f_sum            
            
        #     chroma_diff = np.sqrt(np.sum((f_norm - bg_norm) ** 2, axis=2))
        #     gray_diff = cv2.normalize(chroma_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        #     cv2.imwrite(os.path.join(output_dir, f"frame_{frame_id}_diff.png"), gray_diff)

        #     # Step B: Thresholding Mask
        #     _, mask = cv2.threshold(gray_diff, 35, 255, cv2.THRESH_BINARY)
        #     cv2.imwrite(os.path.join(output_dir, f"frame_{frame_id}_mask.png"), mask)
            
        #     # Step C: Heavy Morphological Closing to fill holes *locally* (No Convex Hull)
        #     kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
        #     mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
            
        #     # Step D: Contour Filtering (Keep only the truck body within the zone)
        #     contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #     clean_mask = np.zeros_like(gray_diff)
            
        #     total_frame_area = h * w
        #     min_area_ratio = 0.05  # The contour must cover at least 1.5% of the total frame
            
        #     if contours:
        #         # Find the single largest contour by area
        #         largest_cnt = max(contours, key=cv2.contourArea)
        #         contour_area = cv2.contourArea(largest_cnt)
        #         area_ratio = contour_area / float(total_frame_area)
                
        #         # Check if the largest contour meets our minimum size requirement
        #         if area_ratio > min_area_ratio:
        #             # Fit a clean geometric rectangle box to the largest contour
        #             rect = cv2.minAreaRect(largest_cnt)
        #             box = cv2.boxPoints(rect)
        #             box = box.astype(np.int32)
                    
        #             # Draw the solid geometric rectangle onto the clean mask
        #             cv2.drawContours(clean_mask, [box], 0, 255, thickness=cv2.FILLED)
                    
        #     clean_mask = cv2.dilate(clean_mask, kernel, iterations=2)
        #     cv2.imwrite(os.path.join(output_dir, f"frame_{frame_id}_cleanmask.png"), clean_mask)

        #     mask_3c = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR) / 255.0

        #     # 2. ROBUST POSITIONING: Use vertical projection (1D row sum) of the masked frame
        #     # Sum pixel intensities horizontally across the road zone to create a unique vertical signature
        #     row_profile = np.sum(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) * (clean_mask > 0), axis=1)
            
        #     if i == 0:
        #         current_y = last_y
        #     else:
        #         # Cross-correlate current frame's vertical profile with the canvas region below last_y
        #         # This finds the exact physical shift even if the truck stopped or slowed down
        #         search_window_size = int(h * 0.8)
        #         search_start = max(0, last_y - int(h * 0.1))
        #         search_end = min(canvas_h - h, last_y + search_window_size)
                
        #         # Fallback to smooth incremental step if profile is too weak
        #         current_y = last_y + int(h * 0.15) # Default estimated flow step
                
        #         # If we have a valid truck signature, refine position via 1D template matching on row sums
        #         if np.max(row_profile) > 0:
        #             # Match against canvas history profile
        #             canvas_gray = cv2.cvtColor(canvas[search_start:search_end + h, :], cv2.COLOR_BGR2GRAY)
        #             # Simple sliding dot product to find best alignment
        #             best_offset = 0
        #             max_corr = -1
                    
        #             # Normalize profile for matching
        #             profile_subset = row_profile[int(h*0.2):int(h*0.8)] # Use middle stable part of container
        #             if len(profile_subset) > 10:
        #                 for offset in range(0, max(1, canvas_gray.shape[0] - len(profile_subset))):
        #                     canvas_strip = canvas_gray[offset:offset+len(profile_subset), :]
        #                     corr = np.sum(profile_subset[:, np.newaxis] * canvas_strip) # basic correlation
        #                     if corr > max_corr:
        #                         max_corr = corr
        #                         best_offset = offset
                        
        #                 # If correlation is plausible, update current_y relative to search start
        #                 if max_corr > 0:
        #                     # Keep motion strictly forward (monotonic downward progression)
        #                     candidate_y = search_start + best_offset - int(h * 0.2)
        #                     if candidate_y > last_y:
        #                         current_y = min(candidate_y, last_y + int(h * 0.5)) # Cap max jump

        #     last_y = current_y

        #     # Stamp onto Master Canvas
        #     if 0 <= current_y and current_y + h <= canvas_h:
        #         roi_canvas = canvas[current_y:current_y+h, :, :].copy()
        #         blended_slice = np.where(mask_3c > 0, frame, roi_canvas)
                
        #         cv2.imwrite(os.path.join(output_dir, f"frame_{frame_id}_blended.png"), blended_slice)
        #         canvas[current_y:current_y+h, :, :] = blended_slice

            # # Compute smooth progressive position for this frame
            # current_y = int(start_y + (i * step_size))
            
            # # Step F: Stamp and save individual blended slice for inspection
            # if 0 <= current_y and current_y + h <= canvas_h:
            #     roi_canvas = canvas[current_y:current_y+h, :, :].copy()
            #     blended_slice = np.where(mask_3c > 0, frame, roi_canvas)
                
            #     # Save isolated blended output for this frame step
            #     cv2.imwrite(os.path.join(output_dir, f"frame_{frame_id}_blended.png"), blended_slice)
                
            #     # Apply to master canvas
            #     canvas[current_y:current_y+h, :, :] = blended_slice

        # # 3. Correctly tile the background plate to match canvas size for auto-cropping
        # full_bg_plate = np.tile(background_plate, (int(canvas_h / h) + 1, 1, 1))[:canvas_h, :, :]

        # # Crop out empty top and bottom padding tightly
        # diff_canvas = np.mean(cv2.absdiff(canvas, full_bg_plate), axis=2)
        # non_empty_rows = np.where(np.max(diff_canvas, axis=1) > 10)[0]
        
        # if len(non_empty_rows) > 0:
        #     ymin, ymax = max(0, non_empty_rows[0] - 20), min(canvas_h, non_empty_rows[-1] + 20)
        #     final_composite = canvas[ymin:ymax, :, :]
        # else:
        #     final_composite = canvas

        # cv2.imwrite(output_path, final_composite, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        # print(f"[/] Composite saved to: {output_path}")
        hasProcessedSomething = True

    except Exception as e:
        print(f"Exception encountered while stitching frames for {output_dir}: {e} ")
        traceback.print_exc()
        raise
    finally:
        return hasProcessedSomething

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
        hasProcessedFrames = extract_frames_from_mkv(mkv_path, RECORDING_DIR)
        hasStichedFrames = stitch_frames(mkv_path, RECORDING_DIR)
        break

    print("\n[/] All extractions completed!")

if __name__ == "__main__":
    main()