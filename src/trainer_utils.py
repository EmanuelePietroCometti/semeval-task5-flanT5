import torch
import numpy as np
from dataclasses import dataclass, field
from transformers import Seq2SeqTrainer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, TrainerCallback
from scipy.stats import spearmanr
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

@dataclass
class CustomSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    acc_weight: float = field(default=0.7, metadata={"help": "Peso per l'accuracy"})
    spearman_weight: float = field(default=0.3, metadata={"help": "Peso per Spearman"})
    ce_weight: float = field(default=0.2)
    kl_weight: float = field(default=0.2)
    mse_weight: float = field(default=0.6)
    patience_lronplateau: int = field(default=2)
    threshold_lronplateau: float = field(default=0.005)

# Collator Custom
class RobustDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features):
        target_scores = [f.get("target_scores", 3.0) for f in features]
        stdevs = [f.get("stdev", 1.0) for f in features]
        
        batch = super().__call__(features)
        
        batch["target_scores"] = torch.tensor(target_scores, dtype=torch.float32)
        batch["stdev"] = torch.tensor(stdevs, dtype=torch.float32)
        return batch

# Trainer Custom
class ExpectedValueTrainer(Seq2SeqTrainer):
    def __init__(self, *args, target_token_ids=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_token_ids = target_token_ids
        
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if self.lr_scheduler is None:
            self.lr_scheduler = ReduceLROnPlateau(
                optimizer if optimizer is not None else self.optimizer,
                mode='max',
                factor=0.5,
                patience=self.args.patience_lronplateau,
                threshold=0.005,
            )
        return self.lr_scheduler


    # All'interno di ExpectedValueTrainer
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        target_scores = inputs.pop("target_scores", None)
        stdevs = inputs.pop("stdev", None)

        outputs = model(**inputs)
        
        # Estrazione Logit e Probabilità del Modello
        logits = outputs.logits[:, 0, :].to(torch.float32)
        relevant_logits = logits[:, self.target_token_ids]
        
        # Per la KL, il modello deve fornire log_softmax
        log_probs = F.log_softmax(relevant_logits, dim=-1)
        # Per la MSE, usiamo le probabilità classiche
        probs = F.softmax(relevant_logits, dim=-1)
        
        weights_val = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=logits.device, dtype=torch.float32)
        preds_cont = torch.sum(probs * weights_val, dim=-1)

        if target_scores is not None and stdevs is not None:
            target_scores = target_scores.to(model.device)
            stdevs = stdevs.to(model.device)

            # --- CALCOLO DISTRIBUZIONE TARGET (Soft Labels) ---
            # Creiamo un range [1, 5] per ogni esempio nel batch
            voti = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=model.device)
            mu = target_scores.unsqueeze(1)    # Shape: [batch, 1]
            sigma = stdevs.unsqueeze(1) + 0.1  # Aggiungiamo epsilon per stabilità
            
            # Formula Gaussiana: exp(-0.5 * ((x - mu) / sigma)^2)
            target_dist = torch.exp(-0.5 * ((voti - mu) / sigma) ** 2)
            # Normalizzazione per sommare a 1
            target_dist = target_dist / target_dist.sum(dim=-1, keepdim=True)

            # --- CALCOLO LOSS ---
            # KL Divergence (confronta le distribuzioni)
            # Nota: F.kl_div richiede log_probs in input e target_dist come target
            kl_loss = F.kl_div(log_probs, target_dist, reduction='batchmean')

            # Weighted MSE (confronta le medie)
            loss_weights = 1.0 / (stdevs + 0.1)
            mse_raw = (preds_cont - target_scores) ** 2
            weighted_mse_loss = (mse_raw * loss_weights).mean()

            ce = outputs.loss
            # --- COMBINAZIONE ---
            # Bilanciamento: CE + KL (forma) + MSE (valore puntuale)
            total_loss = (self.args.ce_weight*ce) + (self.args.kl_weight * kl_loss) + (self.args.mse_weight * weighted_mse_loss)
            
        else:
            total_loss = outputs.loss # Fallback alla CrossEntropy standard

        return (total_loss, outputs) if return_outputs else total_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            target_scores = inputs.pop("target_scores", None)
            stdevs = inputs.pop("stdev", None)

            outputs = model(**inputs)

            logits = outputs.logits[:, 0, :].to(torch.float32)
            relevant_logits = logits[:, self.target_token_ids]
            probs = torch.nn.functional.softmax(relevant_logits, dim=-1)
            weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=logits.device, dtype=torch.float32)
            preds_cont = torch.sum(probs * weights, dim=-1)

            if target_scores is None:
                target_scores = torch.zeros(preds_cont.shape, device=model.device)
            if stdevs is None:
                stdevs = torch.zeros(preds_cont.shape, device=model.device)

            labels_with_meta = torch.stack([target_scores, stdevs], dim=1)

            return (outputs.loss, preds_cont, labels_with_meta)

# Caclolo metriche
def compute_metrics(eval_pred, acc_weight=0.7, spearman_weight=0.3):
    preds, labels_with_meta = eval_pred

    y_true = labels_with_meta[:, 0]
    stdevs = labels_with_meta[:, 1]

    thresholds = np.maximum(1.0, stdevs)
    diff = np.abs(preds - y_true)
    accuracy = np.mean(diff <= thresholds)

    rho, _ = spearmanr(y_true, preds)
    if np.isnan(rho): rho = 0.0
    
    combined_score = (acc_weight * accuracy) + (spearman_weight * rho)

    return {
        "accuracy_within_std": float(accuracy),
        "spearman": float(rho),
        "combined_score": float(combined_score)
    }

class EvaluationLogCallback(TrainerCallback):
    """Mostra i risultati dell'evaluation in una tabella pulita con combined score."""
    def __init__(self):
        self.header_printed = False

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            train_loss = "N/A"
            for log in reversed(state.log_history):
                if "loss" in log:
                    train_loss = f"{log['loss']:.4f}"
                    break

            step = state.global_step
            eval_loss = metrics.get("eval_loss", 0.0)
            acc = metrics.get("eval_accuracy_within_std", 0.0)
            spearman = metrics.get("eval_spearman", 0.0)
            combined = metrics.get("eval_combined_score", 0.0)

            header = f"{'Step':<8} | {'Train Loss':<12} | {'Eval Loss':<12} | {'Acc':<10} | {'Spearman':<10} | {'Combined':<10}"
            separator = "-" * len(header)
            row = f"{step:<8} | {train_loss:<12} | {eval_loss:<12.4f} | {acc:<10.4f} | {spearman:<10.4f} | {combined:<10.4f}"

            if not self.header_printed:
                print(f"\n{header}")
                print(separator)
                self.header_printed = True

            print(row, flush=True)