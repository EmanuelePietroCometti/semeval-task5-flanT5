import sys
import os
import json
import zipfile
from tqdm import tqdm
import torch
import yaml
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference_utils import load_inference_model, get_prediction

def main():
    argparser = argparse.ArgumentParser(description="Script di inferenza per Flan-T5")
    argparser.add_argument("--output_file", type=str, default="outputs/predictions/predictions.jsonl", help="File di output per le predizioni")
    argparser.add_argument("--test_file", type=str, default="data/raw/test.json", help="Percorso al file di test")
    argparser.add_argument("--zip_file", type=str, default="outputs/predictions/predictions.zip", help="File ZIP di output")
    argparser.add_argument("--max_length", type=int, default=1024, help="Lunghezza massima per il tokenizing")
    
    args = argparser.parse_args()

    test_file = args.test_file
    output_file = args.output_file
    zip_file= args.zip_file

    with open("config/config.yaml" , "r") as config_file:
        config = yaml.safe_load(config_file)

    if not os.path.exists(config['paths']['output_dir']):
        raise FileNotFoundError(f"Non trovo il modello in {config['paths']['output_dir']}. Hai lanciato train.py?")

    model, tokenizer = load_inference_model(config['model']['base_model'], config['paths']['output_dir'])
    
    # Identificazione ID dei token target (1, 2, 3, 4, 5)
    target_token_ids = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1, 6)]
    target_token_ids_tensor = torch.tensor(target_token_ids, device=model.device)

    print(f"Leggo i dati da {test_file}...")
    with open(test_file, 'r') as f:
        raw_data = json.load(f)

    print(f"Avvio inferenza su {len(raw_data)} esempi...")
    predictions = []

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f_out:
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
                    target_token_ids=target_token_ids_tensor,
                    max_length=args.max_length
                )
                
                result = {
                    "id": sample_id,
                    "prediction": round(score, 3)
                }
    
                f_out.write(json.dumps(result) + "\n")
                
            except Exception as e:
                print(f"Errore sull'ID {example.get('instance_id', 'unknown')}: {e}")

    print(f"Predizioni salvate in: {output_file}")
    print(f"Creo archivio ZIP...")
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_file, arcname="predictions.jsonl")
    
    print(f"Tutto pronto! File da sottomettere: {zip_file}")

if __name__ == "__main__":
    main()