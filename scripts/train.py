# train.py
import sys
import os
# Aggiungi la root al path per importare src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import Seq2SeqTrainingArguments
from src.data_utils import load_datasets, get_tokenize_function
from src.model_utils import load_base_model, apply_lora_config
from src.trainer_utils import ExpectedValueTrainer, RobustDataCollator

def main():
    TRAIN_FILE = "data/raw/train.json"
    DEV_FILE = "data/raw/dev.json"
    OUTPUT_DIR = "outputs/models/flan_t5_lora_v1"
   
    train_ds, dev_ds = load_datasets(TRAIN_FILE, DEV_FILE)
    
    model, tokenizer = load_base_model()
    model = apply_lora_config(model)
    
    print("Tokenizing datasets...")
    tokenize_fn = get_tokenize_function(tokenizer)
    train_encoded = train_ds.map(tokenize_fn, batched=True)
    dev_encoded = dev_ds.map(tokenize_fn, batched=True)
    
    # Setup Trainer
    # Token IDs per i numeri "1", "2", "3", "4", "5"
    target_token_ids = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1, 6)]
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        num_train_epochs=5,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        label_names=["labels", "target_scores", "stdev"]
    )
    
    trainer = ExpectedValueTrainer(
        model=model,
        args=training_args,
        train_dataset=train_encoded,
        eval_dataset=dev_encoded,
        data_collator=RobustDataCollator(tokenizer, model=model),
        target_token_ids=target_token_ids
    )
    
    # 6. Start Training
    print("Starting training...")
    trainer.train()
    
    # 7. Save
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()