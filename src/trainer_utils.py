import torch
import numpy as np
from transformers import Seq2SeqTrainer, DataCollatorForSeq2Seq
from scipy.stats import spearmanr
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
                patience=4,
                threshold=0.0001,
            )
        return self.lr_scheduler

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        target_scores = inputs.pop("target_scores", None)
        inputs.pop("stdev", None) 

        outputs = model(**inputs)
        ce_loss = outputs.loss

        logits = outputs.logits[:, 0, :].to(torch.float32)
        relevant_logits = logits[:, self.target_token_ids]
        relevant_logits = torch.clamp(relevant_logits, min=-100, max=100)

        probs = torch.nn.functional.softmax(relevant_logits, dim=-1)
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=logits.device, dtype=torch.float32)
        preds_cont = torch.sum(probs * weights, dim=-1)

        if target_scores is not None:
            target_scores = target_scores.to(model.device)
            mse_loss = torch.nn.functional.mse_loss(preds_cont, target_scores)
            total_loss = (0.4 * ce_loss) + (0.6 * mse_loss)
        else:
            total_loss = ce_loss

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
def compute_metrics(eval_pred):
    preds, labels_with_meta = eval_pred

    y_true = labels_with_meta[:, 0]
    stdevs = labels_with_meta[:, 1]

    thresholds = np.maximum(1.0, stdevs)
    diff = np.abs(preds - y_true)
    accuracy = np.mean(diff <= thresholds)

    rho, _ = spearmanr(y_true, preds)
    if np.isnan(rho): rho = 0.0

    return {
        "accuracy_within_std": float(accuracy),
        "spearman": float(rho)
    }