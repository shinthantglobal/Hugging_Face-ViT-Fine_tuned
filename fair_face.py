import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# Load FairFace validation split
print("Loading FairFace dataset...")
dataset = load_dataset("HuggingFaceM4/FairFace", "0.25")['train']
SAVE_DIR = "data/train"
os.makedirs(SAVE_DIR, exist_ok=True)

# List to store metadata for the CSV
metadata_list = []

print("Saving images and generating metadata...")
for i, row in tqdm(enumerate(dataset), total=len(dataset)):
    # Define a clean filename
    filename = f"img_{i}.jpg"
    
    # Save the image
    image_path = os.path.join(SAVE_DIR, filename)
    row['image'].save(image_path)
    
    # Store the labels in a list
    metadata_list.append({
        "filename": filename,
        "age": row['age'],
        "gender": row['gender'],
        "race": row['race']
    })

# Save metadata to CSV
df = pd.DataFrame(metadata_list)
df.to_csv(os.path.join(SAVE_DIR, "metadata.csv"), index=False)

print(f"Done! Check '{SAVE_DIR}' for images and 'metadata.csv' with your labels.")