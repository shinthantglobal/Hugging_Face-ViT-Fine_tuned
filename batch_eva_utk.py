import os
import cv2
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import AutoImageProcessor
from model import AgeGenderViTModel

# --- Configuration ---
DATASET_PATH = "UTKFaceRandom1K"
# RESULTS_CSV = "DeepFace/results/UTKFaceRandom1K.csv"
VISUALIZATION_DIR = "vis_UTKFaceRandom1K/"
valid_extensions = ('.jpg', '.jpeg', '.png')
MODEL_PATH = './my_age_gender_model/checkpoint-5930'

# Create the output directory if it doesn't exist
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Get all valid JPG images
image_files = [f for f in os.listdir(DATASET_PATH) if f.endswith(valid_extensions)]

# Optional: Limit the batch size for testing so it doesn't process all 23,000+ images at once
#image_files = image_files[:100] 

results_data = []

# Loading fine-tuned ViT model
print(f"Loading custom ViT model from {MODEL_PATH}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AgeGenderViTModel.from_pretrained(MODEL_PATH).to(device)
processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
model.eval()

print(f"Starting evaluation and visualization on {len(image_files)} images...")

for filename in tqdm(image_files, desc="Processing Faces"):
    
    # 1. Parse Ground Truth
    parts = filename.split('_')
    try:
        true_age = int(parts[0])
        true_gender = "male" if int(parts[1]) == 0 else "female"
    except ValueError:
        continue

    img_path = os.path.join(DATASET_PATH, filename)

    try:
        # 2. Run DeepFace Analysis
        pil_img = Image.open(img_path).convert("RGB")
        inputs = processor(pil_img, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]

        pred_age = round(logits[0].item())
        gender_prob = float(logits[1].item())

        pred_gender = 'female' if gender_prob >= 0.5 else 'male'
        conf = gender_prob if pred_gender == 'female' else (1.0 - gender_prob)

        gender_correct = (true_gender == pred_gender)
        age_error = abs(pred_age - true_age)
        age_within_5_years = age_error <= 5
        
        # 3. Store Results
        results_data.append({
            "filename": filename,
            "true_age": true_age,
            "pred_age": pred_age,
            "age_error": age_error,
            "true_gender": true_gender,
            "pred_gender": pred_gender,
            "gender_correct": gender_correct
        })

        # --- 4. Image Visualization ---
        img = cv2.imread(img_path)
        if img is not None:
            # Extract bounding box region from DeepFace
            h, w, _ = img.shape
            color = (0, 255, 0) if (gender_correct and age_within_5_years) else (0, 0, 255)

            # Draw Bounding Box
            cv2.rectangle(img, (5, 5), (w-5, h-5), color, 2)

            # Prepare Text Overlays
            true_text = f"True: {true_age}, {true_gender}"
            pred_text = f"Pred: {pred_age}, {pred_gender}"

            # Draw Text (Positioned above the bounding box)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            # Prevent text from drawing completely off-screen if face is near the top

            cv2.putText(img, true_text, (10, 25), font, font_scale, (255, 255, 255), thickness+1) # White outline
            cv2.putText(img, true_text, (10, 25), font, font_scale, (0, 0, 0), thickness)         # Black text

            cv2.putText(img, pred_text, (10, 50), font, font_scale, color, thickness+1)
            cv2.putText(img, pred_text, (10, 50), font, font_scale, color, thickness)

            # Save the annotated image
            output_path = os.path.join(VISUALIZATION_DIR, filename)
            cv2.imwrite(output_path, img)

    except Exception as e:
        pass

    # if len(results_data)>200: break

# --- 5. Process and Save Data ---
df = pd.DataFrame(results_data)

if not df.empty:
    gender_accuracy = df['gender_correct'].mean() * 100
    age_mae = df['age_error'].mean()
    df['age_within_5_years'] = df["age_error"] <= 5
    age_acceptable_accuracy = df['age_within_5_years'].mean() * 100

    print("\n" + "="*40)
    print(" BATCH EVALUATION RESULTS")
    print("="*40)
    print(f"Total Processed   : {len(df)}")
    print(f"Gender Accuracy   : {gender_accuracy:.2f}%")
    print(f"Age MAE           : {age_mae:.2f} years")
    print(f"Age Accuracy (±5y): {age_acceptable_accuracy:.2f}%")
    print("="*40)

    # df.to_csv(RESULTS_CSV, index=False)
    # print(f"\nResults saved to {RESULTS_CSV}")
    print(f"Visualized images saved to {os.path.abspath(VISUALIZATION_DIR)}")
else:
    print("No faces were successfully processed.")