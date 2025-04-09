import os
import cv2
import av
import time
import torch
import numpy as np
import re
from ultralytics import YOLO
from ocr_model import CRNN
import json
from datetime import datetime

# 🛠 CONFIGURABLE PARAMETERS
username = "admin"
password = "password"
ip = "192.168.1.98"
port = "554"
rtsp_url = "rtsp://" + username + ":" + password + "@" + ip + ":" + port + "/onvif1"

THRESHOLD_VALUE = 600  # 🎯 Define your threshold value
STOP_DELAY = 60  # ⏳ Stop recording `N` seconds after value normalizes
LOG_INTERVAL = 5  # ⏰ Only print detected value every 5 seconds

# 🎥 OUTPUT FOLDERS
output_folder = "recordings"
os.makedirs(output_folder, exist_ok=True)

# ✅ Load YOLO Model
model = YOLO("runs/detect/train10/weights/best.pt")
model.overrides["verbose"] = False

# ✅ Load OCR Model
MODEL_PATH = "ocr_crnn.pth"
CHAR_MAP_FILE = "char_to_idx.json"

# Load character mapping
with open(CHAR_MAP_FILE, "r") as f:
    char_to_idx = json.load(f)
idx_to_char = {v: k for k, v in char_to_idx.items()}

# Initialize OCR Model
num_classes = len(char_to_idx)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ocr_model = CRNN(num_classes).to(device)
ocr_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
ocr_model.eval()

# Open RTSP Stream
container = av.open(RTSP_URL, options={
    "rtsp_transport": "udp",
    "flags": "low_delay",
    "max_delay": "100000",
    "buffer_size": "8192000",
    "reorder_queue_size": "32",
    "threads": "auto",
    "hwaccel": "cuda"
})

# Get Video and Audio Streams
video_stream = container.streams.video[0]
audio_stream = next((s for s in container.streams if s.type == 'audio'), None)

# 🎥 Recording State Variables
recording = False
recording_end_time = 0
current_recording = None
video_writer = None
audio_writer = None
last_log_time = 0  # ⏰ Tracks last log print time

# ⏳ Video Frame Properties
frame_width = video_stream.width
frame_height = video_stream.height
frame_rate = int(video_stream.average_rate)

# Ensure Display Window Exists
cv2.namedWindow("YOLO + OCR Live Detection", cv2.WINDOW_NORMAL)
time.sleep(2)
cv2.setWindowProperty("YOLO + OCR Live Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def preprocess_image(img, target_size=(120, 100)):
    """Ensures input images match the expected size for the OCR model."""
    resized_img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    resized_img = resized_img.astype(np.float32) / 255.0
    return torch.tensor(resized_img).unsqueeze(0).unsqueeze(0).to(device)  # Shape: [1, 1, H, W]

def decode_text(output):
    """Convert model output to readable text while handling tensor format correctly."""
    _, preds = torch.max(output, 2)
    preds = preds.squeeze().tolist()
    decoded_text = ""
    last_char = None
    for idx in preds:
        if isinstance(idx, list):
            idx = idx[0]
        if idx in idx_to_char:
            char = idx_to_char[idx]
            if char != "<BLANK>" and char != last_char:
                decoded_text += char
            last_char = char
    return decoded_text.strip()

def is_valid_meter_reading(text):
    """Check if OCR output is exactly 3 digits (ignores non-numeric characters)."""
    digits_only = re.sub(r"\D", "", text)
    return len(digits_only) == 3

def start_recording():
    """Starts a new recording session."""
    global recording, current_recording, video_writer, audio_writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_recording = os.path.join(output_folder, f"record_{timestamp}.mp4")
    
    # Open video writer
    video_writer = av.open(current_recording, mode="w")
    video_stream_out = video_writer.add_stream("h264", rate=frame_rate)
    video_stream_out.width = frame_width
    video_stream_out.height = frame_height
    video_stream_out.pix_fmt = "yuv420p"
    
    # Open audio writer if audio exists
    if audio_stream:
        audio_writer = video_writer.add_stream("aac", rate=audio_stream.rate)
        audio_writer.channels = audio_stream.channels
        audio_writer.format = "fltp"
    
    print(f"🎥 Recording started: {current_recording}")
    recording = True

def stop_recording():
    """Stops the current recording session."""
    global recording, video_writer, audio_writer
    if video_writer:
        video_writer.close()
    recording = False
    print(f"🛑 Recording stopped: {current_recording}")

# Process Frames from RTSP Stream
for frame in container.decode(video=0):
    if frame.pts is None:
        continue
    
    frame_time = frame.pts * video_stream.time_base
    img = frame.to_ndarray(format="bgr24")

    # Run YOLO Detection
    results = model(img, conf=0.5)

    detected = False
    detected_value = None

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_screen = img[y1:y2, x1:x2]
            gray_screen = cv2.cvtColor(cropped_screen, cv2.COLOR_BGR2GRAY)
            processed_screen = preprocess_image(gray_screen)

            # Run OCR Model
            with torch.no_grad():
                ocr_output = ocr_model(processed_screen).permute(1, 0, 2)
            predicted_text = decode_text(ocr_output)

            # Extract Only Digits
            digits = re.sub(r"\D", "", predicted_text)
            if len(digits) == 3:
                detected = True
                detected_value = int(digits)
                
                # Log detected value only every 5 seconds
                if time.time() - last_log_time >= LOG_INTERVAL:
                    print(f"🔢 Detected Meter Value: {detected_value}")
                    last_log_time = time.time()

    # Handle Recording Logic
    if detected and detected_value is not None:
        if detected_value >= THRESHOLD_VALUE:
            if not recording:
                start_recording()
            recording_end_time = time.time() + STOP_DELAY
        else:
            if recording and time.time() >= recording_end_time:
                stop_recording()

    # Save Frames to Video
    if recording and video_writer:
        video_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        video_writer.mux(video_stream_out.encode(video_frame))

    # Show Detection
    cv2.imshow("YOLO + OCR Live Detection", img)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
container.close()
if recording:
    stop_recording()
