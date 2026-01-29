import json
from datasets import Dataset

def build_prompt_text(precontext, sentence, ending, homonym, judged_meaning):
    story = f"Context: {precontext}\nAmbiguous Sentence: {sentence}\nEnding: {ending}"
    query = (
        f"Task: I want you to act as an expert linguistic annotator. Rate the plausibility of the word sense for the homonym in the story.\n"
        f"Scale: 1 (not plausible) to 5 (very plausible).\n\n"
        f"Story: {story}\n"
        f"Homonym: {homonym}\n"
        f"Sense to evaluate: {judged_meaning}\n\n"
        f"Constraint: Respond only with a single integer between 1 and 5.\n"
        f"Answer: "
    )
    # Prompt Repetition
    full_prompt = f"{query}"
    return full_prompt

# Funzione per caricare i JSON in Dataset HuggingFace
def load_datasets(train_path, dev_path):
    with open(train_path, "r") as f:
        train_data = list(json.load(f).values())
    with open(dev_path, "r") as f:
        dev_data = list(json.load(f).values())
        
    train_ds = Dataset.from_list(train_data)
    dev_ds = Dataset.from_list(dev_data)
    return train_ds, dev_ds

# Funzione di tokenizzazione per il Trainer
def get_tokenize_function(tokenizer, max_length=512):
    def tokenize_function(batch):
        prompts = [
            build_prompt_text(p, s, e, h, js)
            for p, s, e, h, js in zip(
                batch["precontext"], batch["sentence"], batch["ending"], 
                batch["homonym"], batch["judged_meaning"]
            )
        ]
        
        model_inputs = tokenizer(prompts, padding="max_length", truncation=True, max_length=max_length)
        
        labels = tokenizer(
            [str(int(round(v))) for v in batch["average"]],
            padding="max_length", truncation=True, max_length=2
        )
        model_inputs["labels"] = labels["input_ids"]
        
        model_inputs["target_scores"] = [float(v) for v in batch["average"]]
        model_inputs["stdev"] = batch["stdev"]
        
        return model_inputs
    return tokenize_function