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
    # Parse command-line arguments
    argparser = argparse.ArgumentParser(description="Flan-T5 Inference Pipeline")
    argparser.add_argument("--output_file", type=str, default="outputs/predictions/predictions.jsonl", help="Path to the output file for generated predictions.")
    argparser.add_argument("--test_file", type=str, default="data/raw/test.json", help="Path to the input test file.")
    argparser.add_argument("--zip_file", type=str, default="outputs/predictions/predictions.zip", help="Path to the output ZIP archive.")
    argparser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for tokenization.")
    
    args = argparser.parse_args()

    test_file = args.test_file
    output_file = args.output_file
    zip_file= args.zip_file

    # Configuration loading
    with open("config/config.yaml" , "r") as config_file:
        config = yaml.safe_load(config_file)

    if not os.path.exists(config['paths']['output_dir']):
        raise FileNotFoundError(f"Model not found at {config['paths']['output_dir']}. Have you executed 'train.py'?")

    model, tokenizer = load_inference_model(config['model']['base_model'], config['paths']['output_dir'])
    
    # Identify target token IDs (1, 2, 3, 4, 5)
    target_token_ids = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1, 6)]
    target_token_ids_tensor = torch.tensor(target_token_ids, device=model.device)

    print(f"Leggo i dati da {test_file}...")
    with open(test_file, 'r') as f:
        raw_data = json.load(f)

    print(f"Running inference on {len(raw_data)} examples...")
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
               print(f"Error processing sample ID {sample_id}: {e}")

    print(f"Predictions saved to: {output_file}")
    print(f"Creating ZIP archive...")
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_file, arcname="predictions.jsonl")
    
    print(f"Submission archive created successfully: {zip_file}")

if __name__ == "__main__":
    main()