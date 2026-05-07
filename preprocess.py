import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Setup paths
image_folder = "./UTKFace" 
data = []

# 2. Parse the UTKFace filenames
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg"):
        parts = filename.split('_')
        # UTKFace format: age_gender_race_date.jpg
        if len(parts) >= 3: # Ensure it has age and gender
            try:
                age = int(parts[0])
                gender = int(parts[1])
                data.append({"file_name": filename, "age": age, "gender": gender})
            except ValueError:
                continue

df = pd.DataFrame(data)

# 3. Create the Split (80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 4. Save them
# Note: imagefolder loader can read a single metadata.csv, 
# but it's often easier to just save the full one and split in the training script.
df.to_csv(os.path.join(image_folder, "metadata.csv"), index=False)

print(f"Total images: {len(df)}")
print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")
print("Metadata.csv created successfully!")