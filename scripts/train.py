import sys
import os
import yaml
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
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
from transformers.trainer_callback import DefaultFlowCallback
from src.data_utils import load_datasets, get_tokenize_function
from src.model_utils import load_base_model, apply_lora_config
from src.trainer_utils import ExpectedValueTrainer, RobustDataCollator, compute_metrics, CustomSeq2SeqTrainingArguments, EvaluationLogCallback, MasterProgressCallback

def main():
    # Parse command-line arguments.
    argparser = argparse.ArgumentParser(description="Training script for Flan-T5 with LoRA")
    argparser.add_argument("--train_file", type=str, default="data/raw/train.json", help="Path to the training dataset file.")
    argparser.add_argument("--dev_file", type=str, default="data/raw/dev.json", help="Path to the development dataset file.")
    argparser.add_argument("--batch_size", type=int, default=4, help="Batch size for the trainer.")
    argparser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    argparser.add_argument("--learning_rate", type=float, default=2e-4, help="Optimizer learning rate.")
    argparser.add_argument("--weight_decay", type=float, default=0.01, help="Optimizer weight decay.")
    argparser.add_argument("--lr_scheduler", type=str, default="cosine", help="Scheduler type for learning rate.")
    argparser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximmum gradient for clipping")
    argparser.add_argument("--acc_weight", type=float, default=0.7, help="Weight for accuracy in the combined score calculation.")
    argparser.add_argument("--spearman_weight", type=float, default=0.3, help="Weight for Spearman correlation in the combined score calculation.")
    argparser.add_argument("--early_stopping_patience", type=int, default=5, help="Number of epochs without improvement before early stopping.")
    argparser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for tokenization.")
    argparser.add_argument("--ce_weight", type=float, default=1.0, help="Weight for Cross-Entropy loss in the combined loss calculation.")
    argparser.add_argument("--mse_weight", type=float, default=1.0, help="Weight for MSE loss in the combined loss calculation.")
    argparser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients before performing an update pass.")

    

    # Configuration loading
    with open("config/config.yaml", 'r') as config_file:
        config = yaml.safe_load(config_file)

    args = argparser.parse_args()
    train_file = args.train_file
    dev_file = args.dev_file
    output_dir = config["paths"]["output_dir"]
   
    train_ds, dev_ds = load_datasets(train_file, dev_file)
    
    model, tokenizer = load_base_model(config['model']['base_model'])
    model = apply_lora_config(
        model, 
        r=config['model']['lora_r'], 
        alpha=config['model']['lora_alpha'], 
        dropout=config['model']['lora_dropout'], 
        target_modules=config['model']['target_modules']
    )
    
    print("Running tokenization on datasets...")
    tokenize_fn = get_tokenize_function(tokenizer, args.max_length)

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
    
    ## Identify target token IDs (1, 2, 3, 4, 5)
    target_token_ids = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1, 6)]

    training_args = CustomSeq2SeqTrainingArguments(
        # Output and logging
        output_dir=output_dir,
        report_to="wandb",
        logging_strategy=config['training']['logging_strategy'],
        logging_steps=config['training']['logging_steps'],
        disable_tqdm=True,

        # Evaluation and save strategy
        eval_strategy=config['training']['eval_strategy'],
        save_strategy=config['training']['save_strategy'],
        eval_steps=config['training']['eval_steps'],
        save_steps=config['training']['save_steps'],
        
        # "Best Model" Logic
        load_best_model_at_end=True,
        metric_for_best_model="combined_score", # Accuracy + Spearman
        greater_is_better=True,
        predict_with_generate=False,

        # Training hyperparameters
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        # Optimizer and scheduler
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        optim=config['training']['optimizer'],
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=config['training']['warmup_ratio'],

        # Custom loss and metrics configuration
        # Weights to balance the Loss function
        ce_weight=args.ce_weight,           # CrossEntropy Weight (Syntax focus)
        mse_weight=args.mse_weight,         # MSE Weight (Semantics/Regression focus)
        
        # Weights for the final combined score calculation
        acc_weight=args.acc_weight,
        spearman_weight=args.spearman_weight,

        # Data and system management
        remove_unused_columns=False, 
        label_names=["labels", "target_scores", "stdev"], # Explicitly pass these columns to compute_loss
        
        # Hardware Optimizations
        fp16=False,
        bf16=False,                         # Set to True if A100
    )

    custom_metrics_fn = partial(
        compute_metrics, 
        acc_weight=training_args.acc_weight, 
        spearman_weight=training_args.spearman_weight
    )

    # Trainer initialization
    trainer = ExpectedValueTrainer(
        # Model and data configuration
        model=model,
        args=training_args,
        train_dataset=train_encoded,
        eval_dataset=dev_encoded,
        
        # Processing components
        processing_class=tokenizer,
        data_collator=RobustDataCollator(tokenizer, model=model),
        
        # Metrics and evaluation
        compute_metrics=custom_metrics_fn,
        
        # Callbacks
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience), 
            EvaluationLogCallback(), 
            MasterProgressCallback()
        ],
        
        # Custom Logic
        target_token_ids=target_token_ids 
    )

    keep_callbacks = ["EarlyStoppingCallback", "EvaluationLogCallback", "MasterProgressCallback", "WandbCallback", "DefaultFlowCallback"]
    for cb in trainer.callback_handler.callbacks.copy():
        if type(cb).__name__ not in keep_callbacks:
            trainer.remove_callback(cb)

    print("\n" + "="*40)
    print("--- STARTING TRAINING ---")
    print("="*40 + "\n")
    
    trainer.train()

    print("\n" + "="*40)
    print("--- STARTING EVALUATION ---")
    print("="*40)

    metrics = trainer.evaluate()
    
    acc = metrics.get("eval_accuracy_within_std", 0.0)
    spearman = metrics.get("eval_spearman", 0.0)

    print(f"\n{'-'*45}")
    print(f"| {'METRIC':<28} | {'VALUE':<10} |")
    print(f"{'-'*45}")
    print(f"| {'Accuracy (within std)':<28} | {acc:<10.4f} |")
    print(f"| {'Spearman Correlation':<28} | {spearman:<10.4f} |")
    print(f"{'-'*45}\n")

    
    # Saving model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model successfully saved to: {output_dir}")
    
if __name__ == "__main__":
    main()