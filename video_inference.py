import os
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor
from model import AgeGenderViTModel

# --- Configuration ---
VIDEO_PATH = "video/video.mp4"
OUTPUT_PATH = "output_new_video/video.mp4"
VIT_MODEL_PATH = "./age_gender_fine_tuned/checkpoint-11860"
# Using standard YOLOv8 Nano
face_model = YOLO("yolov8n-face.pt")
# 1. Load Models
device = "cuda" if torch.cuda.is_available() else "cpu"
vit_model = AgeGenderViTModel.from_pretrained(VIT_MODEL_PATH).to(device)
vit_processor = AutoImageProcessor.from_pretrained(VIT_MODEL_PATH)
vit_model.eval()

# 2. Setup Video
cap = cv2.VideoCapture(VIDEO_PATH)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print(f"Running Inference on {device}...")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # YOLO Inference - Standard model
    results = face_model(frame, conf=0.5, verbose=False)

    for r in results:
        for box in r.boxes:
            # No more "top 25%" guessing! 
            # These coordinates are already mapped to the face.
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
            # Ensure coordinates are within frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            face_crop = frame[y1:y2, x1:x2]
            
            # --- THE FIX: CROP HEAD ONLY ---
            # Standard YOLO gives the whole body. We estimate the head is at the top.
            if face_crop.size == 0: continue

            # 3. Age/Gender Inference
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            pil_face = Image.fromarray(face_rgb)
            inputs = vit_processor(pil_face, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = vit_model(**inputs)
                logits = outputs.logits[0]
            
            pred_age = float(logits[0].item())
            gender_prob = float(logits[1].item())
            pred_gender = "Female" if gender_prob >= 0.5 else "Male"

            # 4. Visualization
            color = (0, 255, 0)
            # Draw box only around the "Head" area we cropped
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{pred_gender}, {int(pred_age)}y"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out.write(frame)

cap.release()
out.release()
print(f"Finished! Saved to {OUTPUT_PATH}")