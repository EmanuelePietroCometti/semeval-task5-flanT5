import sys
import os
import json
import zipfile
from tqdm import tqdm
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference_utils import load_inference_model, get_prediction

def main():
    BASE_MODEL_ID = "google/flan-t5-large"
    LORA_MODEL_PATH = "outputs/models/flan_t5_lora_v1" 
    
    TEST_FILE = "data/raw/test.json"
    OUTPUT_FILE = "outputs/predictions/predictions.jsonl"
    ZIP_FILE = "outputs/predictions/predictions.zip"
    
    if not os.path.exists(LORA_MODEL_PATH):
        raise FileNotFoundError(f"Non trovo il modello in {LORA_MODEL_PATH}. Hai lanciato train.py?")

    model, tokenizer = load_inference_model(BASE_MODEL_ID, LORA_MODEL_PATH)
    
    # Identificazione ID dei token target (1, 2, 3, 4, 5)
    target_token_ids = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1, 6)]
    target_token_ids_tensor = torch.tensor(target_token_ids, device=model.device)

    print(f"Leggo i dati da {TEST_FILE}...")
    with open(TEST_FILE, 'r') as f:
        raw_data = json.load(f)

    print(f"Avvio inferenza su {len(raw_data)} esempi...")
    predictions = []

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, 'w') as f_out:
        for sample_id, example in raw_data.items():
            try:
                score = get_prediction(
                    model=model,
                    tokenizer=tokenizer,
                    precontext=example.get("precontext", ""),
                    sentence=example["sentence"],
                    ending=example["ending"],
                    homonym=example["homonym"],
                    judged_meaning=example["judged_meaning"],
                    target_token_ids=target_token_ids_tensor
                )
                
                result = {
                    "instance_id": sample_id,
                    "prediction": score
                }
    
                f_out.write(json.dumps(result) + "\n")
                
            except Exception as e:
                print(f"Errore sull'ID {example.get('instance_id', 'unknown')}: {e}")

    print(f"Predizioni salvate in: {OUTPUT_FILE}")
    print(f"Creo archivio ZIP...")
    with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(OUTPUT_FILE, arcname="predictions.jsonl")
    
    print(f"Tutto pronto! File da sottomettere: {ZIP_FILE}")

if __name__ == "__main__":
    main()