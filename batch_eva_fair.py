import os
import cv2
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import AutoImageProcessor
from model import AgeGenderViTModel  # Your custom class

# --- Configuration ---
CSV_PATH = "./validation/metadata.csv" 
DATASET_PATH = "./validation/"
VISUALIZATION_DIR = "./validation_visualized/"
MODEL_PATH = "./my_age_gender_model/checkpoint-5930"  # Path to your trained weights

# Create the output directory if it doesn't exist
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# HuggingFace/FairFace Class Mappings
HF_AGE_CLASSES = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
HF_GENDER_CLASSES = ["male", "female"]

# 1. Load your Fine-Tuned ViT Model
print(f"Loading custom ViT model from {MODEL_PATH}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AgeGenderViTModel.from_pretrained(MODEL_PATH).to(device)
processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
model.eval()

# Helper function for FairFace Age Ranges
def parse_age_range(age_str):
    age_str = str(age_str).strip().lower()
    if age_str == 'more than 70' or age_str == '8': # Handle index or string
        return 75.0, 70, 100 
    elif '-' in age_str:
        parts = age_str.split('-')
        lower, upper = int(parts[0]), int(parts[1])
        return (lower + upper) / 2.0, lower, upper
    else:
        # If it's an index from HF_AGE_CLASSES
        try:
            idx = int(float(age_str))
            return parse_age_range(HF_AGE_CLASSES[idx])
        except:
            val = float(age_str)
            return val, val, val

# Load dataset metadata
df = pd.read_csv(CSV_PATH)[:200]
results_data = []

print(f"Starting ViT evaluation on FairFace dataset ({len(df)} images)...")

for index, row in tqdm(df.iterrows(), total=len(df)):
    filename = row['filename']
    base_name = os.path.basename(filename)
    img_path = os.path.join(DATASET_PATH, base_name)
    
    if not os.path.exists(img_path): continue

    # Parse Ground Truth
    try:
        gender_val = row['gender']
        true_gender = HF_GENDER_CLASSES[int(float(gender_val))] if str(gender_val).replace('.','').isdigit() else str(gender_val).lower()
        mid_age, lower_age, upper_age = parse_age_range(row['age'])
    except Exception: continue

    try:
        # 2. Run Custom ViT Inference
        pil_img = Image.open(img_path).convert("RGB")
        inputs = processor(pil_img, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]
        
        # Unpack predictions
        pred_age = float(logits[0].item())
        gender_prob = float(logits[1].item())
        
        pred_gender = "female" if gender_prob >= 0.5 else "male"
        conf = gender_prob if pred_gender == "female" else (1.0 - gender_prob)

        # 3. Calculate Metrics
        gender_correct = (true_gender == pred_gender)
        age_error = abs(pred_age - mid_age)
        age_within_5_years = (lower_age - 5) <= pred_age <= (upper_age + 5)
        
        results_data.append({
            "gender_correct": gender_correct,
            "age_error": age_error,
            "age_within_5_years": age_within_5_years
        })

        # --- 4. Image Visualization ---
        img = cv2.imread(img_path)
        if img is not None:
            # ViT usually looks at the whole crop, so we draw a full-image border or a simple box
            h, w, _ = img.shape
            color = (0, 255, 0) if (gender_correct and age_within_5_years) else (0, 0, 255)
            
            cv2.rectangle(img, (5, 5), (w-5, h-5), color, 2)
            true_text = f"True: {int(lower_age)}-{int(upper_age)}s, {true_gender}"
            pred_text = f"ViT: {int(pred_age)}, {pred_gender} ({conf:.0%})"

            cv2.putText(img, true_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(img, true_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(img, pred_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.imwrite(os.path.join(VISUALIZATION_DIR, base_name), img)

    except Exception as e:
        continue

# --- 5. Final Metrics ---
df_results = pd.DataFrame(results_data)
if not df_results.empty:
    print("\n" + "="*40)
    print(" FINE-TUNED ViT EVALUATION RESULTS")
    print("="*40)
    print(f"Total Processed   : {len(df_results)}")
    print(f"Gender Accuracy   : {df_results['gender_correct'].mean()*100:.2f}%")
    print(f"Age MAE           : {df_results['age_error'].mean():.2f} years")
    print(f"Age Accuracy (±5y): {df_results['age_within_5_years'].mean()*100:.2f}%")
    print("="*40)