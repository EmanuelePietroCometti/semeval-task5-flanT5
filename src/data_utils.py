import json
from datasets import Dataset

def build_prompt_text(precontext, sentence, ending, homonym, judged_meaning):
    story = f"Context: {precontext}\nAmbiguous Sentence: {sentence}\nEnding: {ending}"
    few_shot_block = (
        f"Example 1:\n"
        f"Story: Context: The chef was preparing the signature dish for the gala. He reached for the herb garden.\n"
        f"Ambiguous Sentence: He picked some sage.\n"
        f"Ending: He carefully selected the freshest leaves to season the roasted poultry, ensuring a perfect aromatic balance.\n"
        f"Homonym: sage\n"
        f"Sense to evaluate: a person of great wisdom\n"
        f"Answer: 1"
        f"\n\nExample 2:\n"
        f"Story: Context: The hiker was lost in the forest for two days. He was cold and exhausted. He finally saw a light in the distance.\n"
        f"Ambiguous Sentence: He reached the camp.\n"
        f"Ending: He stumbled into the clearing where a group of scouts had set up their tents and were huddling around a warm fire.\n"
        f"Homonym: camp\n"
        f"Sense to evaluate: a place where people live temporarily in tents or cabins\n"
        f"Answer: 5\n\n"
        f"\n\nExample 3:\n"
        f"Story: Context: The carpenter was finishing the custom table. He needed to smooth the edges.\n"
        f"Ambiguous Sentence: He used a plane.\n"
        f"Ending: He ran the tool across the surface of the oak wood, producing long, thin shavings until the texture was perfectly even.\n"
        f"Homonym: plane\n"
        f"Sense to evaluate: a flat surface on which a straight line joining any two points on it would wholly lie\n"
        f"Answer: 2"
        f"\n\nExample 4:\n"
        f"Story: Context: The athlete was preparing for the final jump. The crowd was silent.\n"
        f"Ambiguous Sentence: He took a long spring.\n"
        f"Ending: He sprinted down the track and launched himself into the air with immense power, aiming to break the standing record.\n"
        f"Homonym: spring\n"
        f"Sense to evaluate: the season after winter and before summer\n"
        f"Answer: 1\n\n"
    )
    query = (
        f"Task: Rate the plausibility of the word sense for the homonym in the story.\n"
        f"Scale: 1 (not plausible) to 5 (very plausible).\n\n"
        f"{few_shot_block}\n\n"
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
def get_tokenize_function(tokenizer, max_length):
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