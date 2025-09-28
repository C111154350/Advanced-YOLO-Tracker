import cv2
import numpy as np
from ultralytics import YOLO
import time
import sys
from collections import deque

# --- Configuration (directly in code) ---
config = {
    "model_path": "",
    "video_path": "",  # <-- Set your video path here
    "target_class_id": 2,
    "confidence_threshold": 0.4,
    "tracker_settings": {
        "id_merge_distance_thresh_px": 75,
        "id_merge_time_thresh_frames": 60,
        "id_stale_frames": 120
    }
}

model = YOLO(config['model_path'])
model.to('cuda')

cap = cv2.VideoCapture(config['video_path'])
if not cap.isOpened():
    print(f"Cannot open video: {config['video_path']}")
    sys.exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0:
    fps = 30
    print("FPS not detected, defaulting to 30.")

output_path = ""
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

next_real_id = 0
active_tracks = {}
lost_tracks = {}
raw_id_to_real_id = {}
frame_count = 0

print(f"System started, processing video '{config['video_path']}'...")
prev_time = time.time()

FONT = cv2.FONT_HERSHEY_SIMPLEX

def draw_id_info(frame, box, track_id):
    x1, y1, _, _ = map(int, box)
    info_text = f"Real ID: {track_id}"
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1 - 30), (x1 + 120, y1), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, info_text, (x1 + 5, y1 - 10), FONT, 0.6, (255,255,255), 1)

def draw_dashboard(frame, fps, current_count, unique_count):
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40), FONT, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Current Count: {current_count}", (20, 80), FONT, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"Total Unique: {unique_count}", (20, 120), FONT, 0.8, (0, 255, 0), 2)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video processing finished.")
            break

        frame_count += 1
        annotated_frame = frame.copy()
        
        results = model.track(frame, persist=True, conf=config['confidence_threshold'], classes=[config['target_class_id']], verbose=False, tracker="botsort.yaml")
        
        current_raw_ids = set()
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            raw_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, raw_id in zip(boxes, raw_ids):
                current_raw_ids.add(raw_id)
                center_point = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))

                # --- ID management core logic ---
                if raw_id in raw_id_to_real_id:
                    real_id = raw_id_to_real_id[raw_id]
                    if real_id in lost_tracks:
                        active_tracks[real_id] = lost_tracks.pop(real_id)
                    
                    active_tracks[real_id]['box'] = box
                    active_tracks[real_id]['last_seen_frame'] = frame_count
                    active_tracks[real_id]['trajectory'].append(center_point)
                else:
                    match_found = False
                    tracker_settings = config['tracker_settings']
                    for lost_id, lost_info in list(lost_tracks.items()):
                        time_diff = frame_count - lost_info['lost_frame']
                        dist_center = np.linalg.norm(np.array(center_point) - np.array((int((lost_info['box'][0] + lost_info['box'][2]) / 2), int((lost_info['box'][1] + lost_info['box'][3]) / 2))))
                        
                        if time_diff < tracker_settings['id_merge_time_thresh_frames'] and dist_center < tracker_settings['id_merge_distance_thresh_px']:
                            real_id = lost_id
                            active_tracks[real_id] = lost_info
                            active_tracks[real_id].update({'box': box, 'last_seen_frame': frame_count, 'raw_id': raw_id})
                            active_tracks[real_id]['trajectory'].append(center_point)
                            raw_id_to_real_id[raw_id] = real_id
                            del lost_tracks[real_id]
                            match_found = True
                            break
                    
                    if not match_found:
                        real_id = next_real_id
                        next_real_id += 1
                        active_tracks[real_id] = {
                            'box': box, 'last_seen_frame': frame_count, 'raw_id': raw_id,
                            'trajectory': deque(maxlen=32)
                        }
                        active_tracks[real_id]['trajectory'].append(center_point)
                        raw_id_to_real_id[raw_id] = real_id

        # --- Update and cleanup ---
        newly_lost_ids = []
        for real_id, track_info in list(active_tracks.items()):
            if track_info['last_seen_frame'] != frame_count:
                newly_lost_ids.append(real_id)
        for real_id in newly_lost_ids:
            lost_tracks[real_id] = active_tracks.pop(real_id)
            lost_tracks[real_id]['lost_frame'] = frame_count
        
        tracker_settings = config['tracker_settings']
        for real_id, lost_info in list(lost_tracks.items()):
            if frame_count - lost_info['lost_frame'] > tracker_settings['id_stale_frames']:
                raw_id_to_del = lost_info.get('raw_id')
                if raw_id_to_del and raw_id_to_del in raw_id_to_real_id:
                    del raw_id_to_real_id[raw_id_to_del]
                del lost_tracks[real_id]

        # --- Drawing ---
        for real_id, track in active_tracks.items():
            x1, y1, x2, y2 = track['box']
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 182, 193), 2)
            draw_id_info(annotated_frame, track['box'], real_id)

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        
        draw_dashboard(annotated_frame, fps, len(active_tracks), next_real_id)
        
        out.write(annotated_frame)
        
        cv2.imshow("Video Processing...", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("System shutting down...")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nFinal statistics: {next_real_id} unique objects detected.")