import cv2
from ultralytics import YOLO
import os

# --- 1. DEFINE YOUR PATHS ---
# Make sure your video is in the 'data' folder
VIDEO_NAME = "test_video.mp4" # <--- IMPORTANT: CHANGE THIS to your video file's name
VIDEO_IN_PATH = os.path.join('../data', VIDEO_NAME)
VIDEO_OUT_PATH = os.path.join('../outputs', f'yolo_output_{VIDEO_NAME}')

# --- 2. LOAD THE YOLO MODEL ---
# 'yolov8n.pt' is the smallest and fastest "nano" model
model = YOLO('yolov8n.pt')

# --- 3. LOAD THE VIDEO FILE ---
# Open the video file
cap = cv2.VideoCapture(VIDEO_IN_PATH)

# Check if the video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file {VIDEO_IN_PATH}")
    exit()

# Get video properties (width, height, frames-per-second)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# --- 4. CREATE THE VIDEO WRITER ---
# Define the codec and create VideoWriter object
# 'mp4v' is a common codec for .mp4 files
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(VIDEO_OUT_PATH, fourcc, fps, (frame_width, frame_height))

print("Processing video... This may take a while.")

# --- 5. PROCESS THE VIDEO FRAME BY FRAME ---
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Send the frame to the YOLO model for detection
        # 'track' will detect and also track objects between frames
        # 'conf=0.3' means only show detections with > 30% confidence
        results = model.track(frame, persist=True, conf=0.3)

        # Get the frame with the boxes and labels plotted on it
        annotated_frame = results[0].plot()

        # Write the annotated frame to the output video
        out.write(annotated_frame)
    else:
        # Break the loop if we've reached the end of the video
        break

# --- 6. CLEAN UP ---
# Release the video capture and video writer objects
cap.release()
out.release()
cv2.destroyAllWindows() # Close any open CV windows

print(f"Video processing complete! Output saved to: {VIDEO_OUT_PATH}")