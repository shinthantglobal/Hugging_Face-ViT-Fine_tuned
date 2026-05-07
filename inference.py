import torch
from transformers import AutoImageProcessor
from model import AgeGenderViTModel # Import your custom class
from PIL import Image

# 1. Load manually (avoiding the pipeline)
model_path = "./final_production_model"
model = AgeGenderViTModel.from_pretrained(model_path)
processor = AutoImageProcessor.from_pretrained(model_path)

# 2. Process Image
img = Image.open("images/img_20.jpg").convert("RGB")
inputs = processor(img, return_tensors="pt")

# 3. Predict
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    # The author's model.py returns an object with 'logits'
    logits = outputs.logits[0] 

# 4. THE REAL MATH
# Logit 0 is Age
predicted_age = logits[0].item() 
# Logit 1 is Gender (0 to 1)
gender_prob = logits[1].item()

if gender_prob >= 0.5:
    gender = "Female"
    conf = gender_prob
else:
    gender = "male"
    conf = 1 - gender_prob

print(f"--- FINAL PREDICTION ---")
print(f"Age: {int(predicted_age)} years old")
print(f"Gender: {gender} ({conf:.1%} confidence)")