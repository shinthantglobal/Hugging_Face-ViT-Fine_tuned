import os
import cv2
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import AutoImageProcessor
from model import AgeGenderViTModel 

# --- Configuration ---
CSV_PATH = "./data/validation/metadata.csv" 
DATASET_PATH = "./data/validation/"
VISUALIZATION_DIR = "./new_model_metrics/"
MODEL_PATH = "./age_gender_fine_tuned/checkpoint-11860"

os.makedirs(VISUALIZATION_DIR, exist_ok=True)

HF_AGE_CLASSES = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
HF_GENDER_CLASSES = ["male", "female"]

# 1. Load Model
print(f"Loading custom ViT model from {MODEL_PATH}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AgeGenderViTModel.from_pretrained(MODEL_PATH).to(device)
processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
model.eval()

def parse_age_range(age_str):
    age_str = str(age_str).strip().lower()
    # Handle "more than 70" or class index 8
    if age_str in ['more than 70', '8']:
        return 75.0, 70.0, 100.0 
    elif '-' in age_str:
        parts = age_str.split('-')
        lower, upper = float(parts[0]), float(parts[1])
        return (lower + upper) / 2.0, lower, upper
    else:
        try:
            # Handle cases where age is passed as the class index string
            idx = int(float(age_str))
            return parse_age_range(HF_AGE_CLASSES[idx])
        except:
            val = float(age_str)
            return val, val, val

def get_range_error(pred, lower, upper):
    if pred < lower:
        return lower - pred
    elif pred > upper:
        return pred - upper
    else:
        return 0.0

# Load metadata
df = pd.read_csv(CSV_PATH)
results_data = []

print(f"Starting ViT evaluation on dataset ({len(df)} images)...")

for index, row in tqdm(df.iterrows(), total=len(df)):
    filename = row['filename']
    base_name = os.path.basename(filename)
    img_path = os.path.join(DATASET_PATH, base_name)
    
    if not os.path.exists(img_path): continue

    try:
        # Robust Gender Parsing
        gender_val = str(row['gender']).strip().lower()
        if gender_val in ['0', '0.0', 'male']:
            true_gender = "male"
        elif gender_val in ['1', '1.0', 'female']:
            true_gender = "female"
        else:
            true_gender = gender_val # Fallback
            
        mid_age, lower_age, upper_age = parse_age_range(row['age'])
    except Exception: continue

    try:
        pil_img = Image.open(img_path).convert("RGB")
        inputs = processor(pil_img, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]
        
        pred_age = float(logits[0].item())
        gender_prob = float(logits[1].item()) # Assuming Sigmoid output for gender
        
        pred_gender = "female" if gender_prob >= 0.5 else "male"
        conf = gender_prob if pred_gender == "female" else (1.0 - gender_prob)

        # --- Metrics Calculation ---
        gender_correct = (true_gender.lower() == pred_gender.lower())
        
        # Strict Error (vs Midpoint)
        strict_error = abs(pred_age - mid_age)
        
        # Range-Aware Error (0 if inside bucket)
        range_error = get_range_error(pred_age, lower_age, upper_age)
        
        # Exact Bucket Accuracy (Added small epsilon for float precision)
        is_in_bucket = (lower_age - 0.1 <= pred_age <= upper_age + 0.1)
        
        age_within_5_years = (lower_age - 5) <= pred_age <= (upper_age + 5)
        
        results_data.append({
            "gender_correct": gender_correct,
            "strict_error": strict_error,
            "range_error": range_error,
            "is_in_bucket": is_in_bucket,
            "age_within_5_years": age_within_5_years
        })

        # --- Enhanced Visualization ---
        img = cv2.imread(img_path)
        if img is not None:
            h, w, _ = img.shape
            
            # Logic for box color:
            # Green: Both Correct | Orange: One Correct | Red: Both Wrong
            if gender_correct and is_in_bucket:
                color = (0, 255, 0) # Green
            elif gender_correct or is_in_bucket:
                color = (0, 165, 255) # Orange
            else:
                color = (0, 0, 255) # Red
            
            cv2.rectangle(img, (2, 2), (w-2, h-2), color, 2)
            
            true_text = f"GT: {int(lower_age)}-{int(upper_age)}, {true_gender}"
            # Showing 1 decimal place for pred_age helps explain why a box is Red/Orange
            pred_text = f"ViT: {pred_age:.1f}, {pred_gender} ({conf:.0%})"

            # Draw background contrast for text
            cv2.putText(img, true_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
            cv2.putText(img, true_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.putText(img, pred_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
            cv2.putText(img, pred_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            cv2.imwrite(os.path.join(VISUALIZATION_DIR, base_name), img)

    except Exception as e: 
        print(f"Error processing {filename}: {e}")
        continue

# --- Final Metrics Summary ---
df_results = pd.DataFrame(results_data)
if not df_results.empty:
    print("\n" + "="*45)
    print("      FINE-TUNED ViT EVALUATION RESULTS")
    print("="*45)
    print(f"Total Processed       : {len(df_results)}")
    print(f"Gender Accuracy       : {df_results['gender_correct'].mean()*100:.2f}%")
    print(f"Strict Age MAE (mid)  : {df_results['strict_error'].mean():.2f} years")
    print(f"Range-Aware Age MAE   : {df_results['range_error'].mean():.2f} years")
    print(f"Exact Bucket Accuracy : {df_results['is_in_bucket'].mean()*100:.2f}%")
    print(f"Age Accuracy (±5y)    : {df_results['age_within_5_years'].mean()*100:.2f}%")
    print("="*45)