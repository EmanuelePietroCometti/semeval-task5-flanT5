import json
from datasets import Dataset

def build_prompt_text(precontext, sentence, ending, homonym, judged_meaning, example_sentence):
    story = f"Context: {precontext}\nAmbiguous Sentence: {sentence}\nEnding: {ending}"
    query = (
        f"Task: You are an expert linguistic annotator. Your task is to rate the plausibility of a specific word sense within a story on a continuous scale from 1 to 5\n\n"
        f"Story:\n{story}\n\n"
        f"Target word: {homonym}\n"
        f"Proposed Meaning: {judged_meaning}\n"
        f"Sense Example: {example_sentence}\n\n"
        f"Rating criteria:\n"
        f"1 = The sense is completely impossible or contradicts the story.\n"
        f"2 = The sense is unlikely or does not fit well.\n"
        f"3 = The sense is ambiguous or partially fits.\n"  
        f"4 = The sense is plausible and fits the story.\n"
        f"5 =  The sense is perfectly natural and implied by the story..\n\n"
        f"Answer: "
    )
    # Prompt Repetition
    full_prompt = f"{query}\n\nLet me repeat that:\n\n{query}"
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
            build_prompt_text(p, s, e, h, j, e)
            for p, s, e, h, j, e in zip(
                batch["precontext"], batch["sentence"], batch["ending"], 
                batch["homonym"], batch["judged_meaning"], batch["example_sentence"]
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