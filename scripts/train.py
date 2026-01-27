import sys
import os
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
import logging
warnings.filterwarnings("ignore")
import torch
from functools import partial
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)

from transformers import EarlyStoppingCallback
from transformers.trainer_callback import PrinterCallback, ProgressCallback
from src.data_utils import load_datasets, get_tokenize_function
from src.model_utils import load_base_model, apply_lora_config
from src.trainer_utils import ExpectedValueTrainer, RobustDataCollator, compute_metrics, CustomSeq2SeqTrainingArguments, EvaluationLogCallback

def main():
    TRAIN_FILE = "data/raw/train.json"
    DEV_FILE = "data/raw/dev.json"
    OUTPUT_DIR = "outputs/models/flan_t5_lora_v1"
   
    train_ds, dev_ds = load_datasets(TRAIN_FILE, DEV_FILE)
    
    model, tokenizer = load_base_model()
    model = apply_lora_config(model)
    
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

    # Configurazione Training
    batch_size = 4
    num_epochs = 10
    learning_rate = 2e-4
    weight_decay = 0.01

    training_args = CustomSeq2SeqTrainingArguments(
        output_dir="./outputs/models/flan_t5_lora",
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=100,
        save_steps=100,
        logging_strategy="steps",
        logging_steps=20,
        disable_tqdm=False,
        report_to="wandb",
        lr_scheduler_type="cosine",
        warmup_ratio=0.01,
        optim="adafactor",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        max_grad_norm=1.0,
        predict_with_generate=False,
        load_best_model_at_end=True,
        acc_weight=0.7,
        spearman_weight=0.3,
        metric_for_best_model="combined_score",
        greater_is_better=True,
        fp16=False,
        bf16=torch.cuda.is_bf16_supported(),
        label_names=["labels", "target_scores", "stdev"], 
        remove_unused_columns=False
    )

    custom_metrics_fn = partial(
        compute_metrics, 
        acc_weight=training_args.acc_weight, 
        spearman_weight=training_args.spearman_weight
    )

    # Trainer
    trainer = ExpectedValueTrainer(
        model=model,
        args=training_args,
        train_dataset=train_encoded,
        eval_dataset=dev_encoded,
        compute_metrics=custom_metrics_fn,
        processing_class=tokenizer,
        data_collator=RobustDataCollator(tokenizer, model=model),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4), EvaluationLogCallback()],
        target_token_ids=target_token_ids 
    )
    

    for callback in trainer.callback_handler.callbacks:
        print(type(callback).__name__)

    is_printer_present = any(isinstance(c, PrinterCallback) for c in trainer.callback_handler.callbacks)
    print(f"PrinterCallback presente: {is_printer_present}")

    trainer.remove_callback(PrinterCallback)
    trainer.remove_callback(ProgressCallback)
    print("-- Avvio Training --")
    trainer.train()
    
    print("-- Avvio Evaluation --")
    eval = trainer.eval()
    print("Accuracy within std:", eval.get("eval_accuracy_within_std", "N/A"))
    print("Spearman:", eval.get("eval_spearman", "N/A"))

    
    # Salvataggio modello
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()