# Advanced YOLO Object Tracking and Counting System

This is an advanced object tracking and counting system based on Python, `ultralytics` YOLO, and **BoT-SORT**. The core of this project is a custom **ID Re-linking algorithm** built on top of the powerful BoT-SORT tracker, designed to solve the problem where standard trackers assign new IDs to objects after temporary occlusion or leaving the frame, resulting in duplicate counting.

## 🚀 Features

- **High-Performance Tracking**: Utilizes the advanced **BoT-SORT** algorithm, which combines appearance feature matching for stable short-term tracking.
- **ID Re-linking Algorithm**: When an object disappears and reappears nearby, the system uses both **temporal** and **spatial** similarity to re-associate it with its original ID, enabling more persistent long-term tracking.
- **Accurate Counting**: Displays both the "current object count" in the frame and the "total unique object count" after ID merging correction.
- **Video Processing**: Supports reading video files and writes annotated results with persistent IDs frame-by-frame into a new video file.

## ⚙️ Requirements

Please ensure all required Python packages are installed in your environment. You can install them with:

```bash
pip install -r requirements.txt
```

## 🛠️ How to Use

1.  **Download the Project**:
    ```bash
    git clone https://github.com/YourUsername/Advanced-YOLO-Tracker.git
    cd Advanced-YOLO-Tracker
    ```
    (Replace `YourUsername` with your GitHub username)

2.  **Modify Settings**:
    - Open the main script (e.g., `tracker.py`).
    - At the top of the file, edit the `config` dictionary with your own settings:
      ```python
      config = {
          "model_path": "your_yolo_model.pt",
          "video_path": "your_video.mp4",
          "target_class_id": 2, # The object class ID to track (e.g., 2 for 'car')
          "confidence_threshold": 0.4,
          "tracker_settings": {
              "id_merge_distance_thresh_px": 75,  # Distance threshold for ID merging (pixels)
              "id_merge_time_thresh_frames": 60,  # Time threshold for ID merging (frames)
              "id_stale_frames": 120              # Frames to wait before completely clearing a track
          }
      }
      # ...
      output_path = "output_video.mp4" # Set the output video path and filename
      ```

3.  **Run the Script**:
    ```bash
    python tracker.py
    ```

4.  **View Results**:
    - The program will display the processed frames in real time.
    - After processing, the complete result video will be saved to your specified `output_path`.

## 📄 License

This project is licensed under the MIT License.