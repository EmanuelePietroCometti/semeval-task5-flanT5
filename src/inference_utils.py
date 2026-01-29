import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .data_utils import build_prompt_text

def load_inference_model(base_model_name, lora_model_path):
    """
    Carica il modello base e applica i pesi LoRA addestrati.
    """
    print(f"Loading base model: {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Carica modello base
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map="auto"
    )

    print(f"Loading LoRA adapters from: {lora_model_path}...")
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    model.eval()
    
    return model, tokenizer

def get_prediction(model, tokenizer, precontext, sentence, ending, homonym, judged_meaning, example_sentence, target_token_ids):
    """
    Esegue l'inferenza su un singolo esempio e restituisce il valore continuo (float).
    """
    prompt = build_prompt_text(precontext, sentence, ending, homonym, judged_meaning, example_sentence)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    decoder_input_ids = torch.tensor([[model.config.pad_token_id]], device=model.device)

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=decoder_input_ids
        )

        # Prendiamo i logit del primo token generato
        logits = outputs.logits[:, 0, :].to(torch.float32)
        
        # Isoliamo solo i logit dei numeri 1, 2, 3, 4, 5
        relevant_logits = logits[:, target_token_ids]

        # Softmax e Media Ponderata
        probs = torch.nn.functional.softmax(relevant_logits, dim=-1)
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=model.device, dtype=torch.float32)
        
        pred_cont = torch.sum(probs * weights, dim=-1)

        return pred_cont.item()