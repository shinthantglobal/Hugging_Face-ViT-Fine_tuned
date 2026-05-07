import os
import torch
import pandas as pd
from PIL import Image
from torch import nn
from transformers import TrainingArguments, Trainer, ViTImageProcessor
from model import AgeGenderViTModel

# --- CONFIGURATION ---
BASE_DIR = "./data/" 
TRAIN_CSV = os.path.join(BASE_DIR, "train/metadata.csv")
VAL_CSV = os.path.join(BASE_DIR, "validation/metadata.csv")
PREVIOUS_CHECKPOINT = "./my_age_gender_model/checkpoint-5930"

AGE_MAP = {
    "0-2": 1.5, "3-9": 6.5, "10-19": 14.5, "20-29": 24.5, 
    "30-39": 34.5, "40-49": 44.5, "50-59": 54.5, "60-69": 64.5, "more than 70": 75.0
}

# 1. Load Model & Processor
print("Loading model from previous UTKFace checkpoint...")
model = AgeGenderViTModel.from_pretrained(PREVIOUS_CHECKPOINT)
processor = ViTImageProcessor.from_pretrained(PREVIOUS_CHECKPOINT)

# 2. Custom Dataset Class
class FairFaceLocalDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, img_dir, processor):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a dummy or handle skip
            return None

        inputs = self.processor(image, return_tensors="pt")
        
        # Clean Age Label
        age_key = str(row['age']).strip()
        age_num = AGE_MAP.get(age_key, 25.0) # Default to 25 if key missing
        
        # Gender Label (Assuming 0 or 1 from your print debug)
        gender_num = float(row['gender'])
        
        # IMPORTANT: These keys ("pixel_values", "age_labels", "gender_labels") 
        # are what the Trainer will see in the 'inputs' dictionary.
        return {
            "pixel_values": inputs["pixel_values"].squeeze(),
            "age_labels": torch.tensor(age_num).float(),
            "gender_labels": torch.tensor(gender_num).float()
        }

# 3. Multi-Task Trainer Logic
class Stage2Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 1. Extract labels using the EXACT keys from the Dataset return
        age_labels = inputs.pop("age_labels")
        gender_labels = inputs.pop("gender_labels")
        
        # 2. Forward pass (inputs now only contains 'pixel_values')
        outputs = model(**inputs)
        logits = outputs.logits # Shape: [batch_size, 2]
        
        # 3. Calculate Losses
        # Index 0 = Age, Index 1 = Gender
        loss_age = nn.MSELoss()(logits[:, 0], age_labels)
        
        # Using BCEWithLogitsLoss is safer for raw model outputs
        loss_gender = nn.BCEWithLogitsLoss()(logits[:, 1], gender_labels)
        
        # 4. Combine (Weighted to balance Age vs Gender scales)
        total_loss = (loss_age * 0.05) + loss_gender
        
        return (total_loss, outputs) if return_outputs else total_loss

# 4. Initialize Data and Training
train_dir = os.path.join(BASE_DIR, 'train')
val_dir = os.path.join(BASE_DIR, 'validation')

train_ds = FairFaceLocalDataset(TRAIN_CSV, train_dir, processor)
val_ds = FairFaceLocalDataset(VAL_CSV, val_dir, processor)

training_args = TrainingArguments(
    output_dir="./fairface_finetuned",
    per_device_train_batch_size=32,
    num_train_epochs=2,
    learning_rate=1e-5,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    fp16=True if torch.cuda.is_available() else False,
    remove_unused_columns=False # Crucial when using custom keys in Dataset
)

trainer = Stage2Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

print("Starting Stage 2: FairFace Calibration...")
trainer.train()
trainer.save_model("./final_production_model")