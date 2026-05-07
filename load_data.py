from datasets import load_dataset

# Load the training and validation splits
# 'HuggingFaceM4/FairFace' is the most common version
fairface = load_dataset("HuggingFaceM4/FairFace", "1.25")