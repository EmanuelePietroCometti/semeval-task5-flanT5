# train.py
import sys
import os
import torch
# Aggiungi la root al path per importare src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import Seq2SeqTrainingArguments, EarlyStoppingCallback
from src.data_utils import load_datasets, get_tokenize_function
from src.model_utils import load_base_model, apply_lora_config
from src.trainer_utils import ExpectedValueTrainer, RobustDataCollator, compute_metrics

def main():
    TRAIN_FILE = "data/raw/train.json"
    DEV_FILE = "data/raw/dev.json"
    OUTPUT_DIR = "outputs/models/flan_t5_lora_v1"
   
    train_ds, dev_ds = load_datasets(TRAIN_FILE, DEV_FILE)
    
    model, tokenizer = load_base_model()
    model = apply_lora_config(model)
    
    print("Tokenizing datasets...")
    tokenize_fn = get_tokenize_function(tokenizer)
    print("Tokenizing datasets...")
    tokenize_fn = get_tokenize_function(tokenizer)
    
    column_names = train_ds.column_names
    
    train_encoded = train_ds.map(
        tokenize_fn, 
        batched=True, 
        remove_columns=column_names
    )
    
    dev_encoded = dev_ds.map(
        tokenize_fn, 
        batched=True, 
        remove_columns=column_names
    )
    
    # Setup Trainer
    # Token IDs per i numeri "1", "2", "3", "4", "5"
    target_token_ids = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1, 6)]

    # 2. Configurazione Training
    batch_size = 4
    num_epochs = 10
    learning_rate = 2e-4
    weight_decay = 0.01

    training_args = Seq2SeqTrainingArguments(
        output_dir="./outputs/models/flan_t5_lora", # Usa un path coerente
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=100,
        save_steps=100,
        lr_scheduler_type="cosine",
        warmup_ratio=0.01,
        optim="adafactor", # Ottimo per T5
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        dataloader_num_workers=4, # OK su Colab. Se su Windows dà errore, metti 0
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        logging_dir="./logs",
        logging_steps=20,
        max_grad_norm=1.0,
        predict_with_generate=False,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy_within_std",
        greater_is_better=True,
        fp16=False,
        bf16=torch.cuda.is_bf16_supported(), # Rilevamento automatico
        
        # --- PARAMETRI CRITICI MANCANTI NEL TUO SNIPPET ---
        label_names=["labels", "target_scores", "stdev"], 
        remove_unused_columns=False  # IMPORTANTE: altrimenti cancella target_scores!
    )

    # 3. Inizializzazione Trainer
    trainer = ExpectedValueTrainer(
        model=model,
        args=training_args,
        train_dataset=train_encoded,
        eval_dataset=dev_encoded,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        data_collator=RobustDataCollator(tokenizer, model=model),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=6)],
        target_token_ids=target_token_ids 
    )

    print("Avvio Training...")
    trainer.train()
    
    # 7. Save
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()