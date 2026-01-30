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
    argparser = argparse.ArgumentParser(description="Training script for Flan-T5 with LoRA")
    argparser.add_argument("--train_file", type=str, default="data/raw/train.json", help="Percorso al file di training")
    argparser.add_argument("--dev_file", type=str, default="data/raw/dev.json", help="Percorso al file di sviluppo")
    argparser.add_argument("--batch_size", type=int, default=4, help="Batch size per dispositivo")
    argparser.add_argument("--num_epochs", type=int, default=10, help="Numero di epoche di training")
    argparser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate per l'ottimizzatore")
    argparser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay per l'ottimizzatore")
    argparser.add_argument("--lr_scheduler", type=str, default="cosine", help="Tipo di scheduler del learning rate")
    argparser.add_argument("--max_grad_norm", type=float, default=1.0, help="Massimo gradiente per il clipping")
    argparser.add_argument("--acc_weight", type=float, default=0.7, help="Peso per l'accuracy nella metrica combinata")
    argparser.add_argument("--spearman_weight", type=float, default=0.3, help="Peso per Spearman nella metrica combinata")
    argparser.add_argument("--early_stopping_patience", type=int, default=5, help="Numero di epoche senza miglioramento prima di fermarsi")
    argparser.add_argument("--max_length", type=int, default=1024, help="Lunghezza massima per il tokenizing")
    argparser.add_argument("--ce_weight", type=float, default=0.4, help="Peso per la CrossEntropy nella loss combinata")
    argparser.add_argument("--mse_weight", type=float, default=0.6, help="Peso per la MSE nella loss combinata")
    argparser.add_argument("--patience_lronplateau", type=int, default=2, help="Patience per ReduceLROnPlateau scheduler")
    argparser.add_argument("--threshold_lronplateau", type=float, default=0.005, help="Threshold per ReduceLROnPlateau scheduler")
    argparser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Numero di step per l'accumulazione del gradiente")

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
    
    print("Tokenizing datasets...")
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
    
    # Setup Trainer
    # Token IDs per i numeri "1", "2", "3", "4", "5"
    target_token_ids = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1, 6)]

    training_args = CustomSeq2SeqTrainingArguments(
        # Parametri di percorso
        output_dir=output_dir,
        eval_strategy=config['training']['eval_strategy'],
        save_strategy=config['training']['save_strategy'],
        eval_steps=config['training']['eval_steps'],
        save_steps=config['training']['save_steps'],
        logging_strategy=config['training']['logging_strategy'],
        logging_steps=config['training']['logging_steps'],

        # Parametri di training
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler,
        learning_rate=args.learning_rate,
        warmup_ratio=config['training']['warmup_ratio'],
        max_grad_norm=args.max_grad_norm,
        acc_weight=args.acc_weight,
        spearman_weight=args.spearman_weight,
        optim=config['training']['optimizer'],

        # Altri parametri
        metric_for_best_model="combined_score",
        report_to="wandb",
        predict_with_generate=False,
        load_best_model_at_end=True,
        greater_is_better=True,
        fp16=False,
        bf16=False,
        label_names=["labels", "target_scores", "stdev"], 
        remove_unused_columns=False,
        ce_weight=args.ce_weight,
        mse_weight=args.mse_weight,
        patience_lronplateau=args.patience_lronplateau,
        disable_tqdm=True,
        gradient_checkpointing=True,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience), EvaluationLogCallback(), MasterProgressCallback()],
        target_token_ids=target_token_ids 
    )

    keep_callbacks = ["EarlyStoppingCallback", "EvaluationLogCallback", "MasterProgressCallback", "WandbCallback", "DefaultFlowCallback"]
    
    for cb in trainer.callback_handler.callbacks.copy():
        if type(cb).__name__ not in keep_callbacks:
            trainer.remove_callback(cb)

    print("-- Avvio Training Pulito (v5) --")
    trainer.train()

    print("-- Avvio Evaluation --")
    eval = trainer.evaluate()
    print("Accuracy within std:", eval.get("eval_accuracy_within_std", "N/A"))
    print("Spearman:", eval.get("eval_spearman", "N/A"))

    
    # Salvataggio modello
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()