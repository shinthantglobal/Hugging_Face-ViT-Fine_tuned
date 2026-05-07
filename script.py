import os
import torch
from torch import nn
from transformers import TrainingArguments, Trainer, ViTImageProcessor
from datasets import load_dataset
from model import AgeGenderViTModel 

# 1. Load the model and processor
model_name = "abhilash88/age-gender-prediction"
model = AgeGenderViTModel.from_pretrained(model_name)
processor = ViTImageProcessor.from_pretrained(model_name)

# 2. Data processing (Fixes crop_size error)
def transform(example_batch):
    inputs = processor(
        [x for x in example_batch['image']], 
        return_tensors='pt',
        size={"height": 224, "width": 224},
        crop_size={"height": 224, "width": 224}
    )
    inputs['age_labels'] = torch.tensor(example_batch['age']).float()
    # BCELoss requires float labels for the gender probability
    inputs['gender_labels'] = torch.tensor(example_batch['gender']).float()
    return inputs

# 3. Load and Split Dataset
# Ensure your UTKFace folder contains the metadata.csv we generated
raw_dataset = load_dataset("imagefolder", data_dir="./UTKFace")
split_ds = raw_dataset["train"].train_test_split(test_size=0.2, seed=42)
prepared_ds = split_ds.with_transform(transform)

# 4. Data Collator (Ensures batches are shaped correctly)
def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'age_labels': torch.stack([x['age_labels'] for x in batch]),
        'gender_labels': torch.stack([x['gender_labels'] for x in batch])
    }

# 5. CUSTOM TRAINER (Fixes 'eval_loss' KeyError)
class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Unpack labels
        age_labels = inputs.pop("age_labels")
        gender_labels = inputs.pop("gender_labels")
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits # Logits is [batch_size, 2]
        
        # Split logits: index 0 is Age, index 1 is Gender
        age_logits = logits[:, 0]
        gender_logits = logits[:, 1]
        
        # Age Loss (MSE)
        loss_fct_age = nn.MSELoss()
        age_loss = loss_fct_age(age_logits, age_labels)
        
        # Gender Loss (BCE because of Sigmoid in model.py)
        loss_fct_gender = nn.BCELoss()
        gender_loss = loss_fct_gender(gender_logits, gender_labels)
        
        # Total Weighted Loss (Scaling age down by 0.01)
        loss = (age_loss * 0.1) + gender_loss
        
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Fixes the KeyError: 'eval_loss'. 
        This is called during the evaluation phase at the end of every epoch.
        """
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            # Simply reuse our compute_loss logic
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        
        loss = loss.detach()
        logits = outputs.logits.detach()
        
        # Returns (loss, logits, labels)
        return (loss, logits, None)

# 6. Training Arguments
training_args = TrainingArguments(
    output_dir="./age_gender_fine_tuned",
    per_device_train_batch_size=16,
    num_train_epochs=10,
    learning_rate=2e-5,
    logging_steps=10,
    eval_strategy="epoch", # Modern naming for evaluation_strategy
    save_strategy="epoch",
    load_best_model_at_end=True,
    remove_unused_columns=False,
    fp16=True if torch.cuda.is_available() else False,
)

# 7. Initialize Trainer
trainer = MultiTaskTrainer(
    model=model,
    args=training_args,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["test"],
    data_collator=collate_fn,
)

# 8. Start Training
# This will try to resume from step 1186 if the checkpoint exists
# This looks into your output_dir and finds the latest checkpoint folder
trainer.train(resume_from_checkpoint=True)