import torch
import numpy as np
from dataclasses import dataclass, field
from transformers import Seq2SeqTrainer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, TrainerCallback, ProgressCallback
from scipy.stats import spearmanr
import torch.nn.functional as F
from tqdm.auto import tqdm


class MasterProgressCallback(ProgressCallback):
    """Gestisce un'unica barra per il training evitando ogni altro log testuale."""
    def __init__(self):
        super().__init__()
        self.training_bar = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.training_bar = tqdm(total=state.max_steps, dynamic_ncols=True)
        self.training_bar.set_description("Training")

    def on_step_end(self, args, state, control, **kwargs):
        if self.training_bar:
            self.training_bar.update(1)
            if len(state.log_history) > 0:
                last_log = state.log_history[-1]
                if "loss" in last_log:
                    self.training_bar.set_postfix(loss=f"{last_log['loss']:.4f}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        pass

    def on_prediction_step(self, args, state, control, **kwargs):
        pass

    def on_train_end(self, args, state, control, **kwargs):
        if self.training_bar:
            self.training_bar.close()

@dataclass
class CustomSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    acc_weight: float = field(default=0.7, metadata={"help": "Peso per l'accuracy"})
    spearman_weight: float = field(default=0.3, metadata={"help": "Peso per Spearman"})
    ce_weight: float = field(default=0.1)
    mse_weight: float = field(default=0.9)

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

class ExpectedValueTrainer(Seq2SeqTrainer):
    def __init__(self, *args, target_token_ids=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_token_ids = target_token_ids
        
        if self.target_token_ids:
            print(f"DEBUG: Target Token IDs per 1-5: {self.target_token_ids}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        target_scores = inputs.pop("target_scores", None)
        stdevs = inputs.pop("stdev", None)

        outputs = model(**inputs)
        
        # CrossEntropy Standard
        # Questa forza il modello a generare effettivamente i token numerici corretti
        ce_loss = outputs.loss 
        
        logits = outputs.logits[:, 0, :].to(torch.float32) 
        relevant_logits = logits[:, self.target_token_ids] 
        
        probs = F.softmax(relevant_logits, dim=-1)
        
        weights_val = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=logits.device, dtype=torch.float32)
        preds_cont = torch.sum(probs * weights_val, dim=-1)

        total_loss = ce_loss

        if target_scores is not None:
            target_scores = target_scores.to(model.device)
            stdevs = stdevs.to(model.device)

            # MSE pesato dall'incertezza degli annotatori
            loss_weights = 1.0 / (stdevs + 0.5)
            mse_raw = (preds_cont - target_scores) ** 2
            weighted_mse_loss = (mse_raw * loss_weights).mean()

            # La CE deve guidare l'apprendimento sintattico. La MSE raffina la semantica.
            total_loss = (self.args.ce_weight * ce_loss) + (self.args.mse_weight * weighted_mse_loss)

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

# Calcolo metriche
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
                tqdm.write(f"\n{header}")
                tqdm.write(separator)
                self.header_printed = True

            tqdm.write(row)