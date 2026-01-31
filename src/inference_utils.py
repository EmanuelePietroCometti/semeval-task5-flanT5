import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .data_utils import build_prompt_text

def load_inference_model(base_model_name, lora_model_path):
    """
    Function to load the base model and apply LoRA adapters for inference.
    :param base_model_name: name of the base model
    :param lora_model_path: path to the LoRA adapters
    :return: model with LoRA adapters and tokenizer
    """
    print(f"Loading base model: {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name,
        device_map="auto",
        use_cache=False
    )

    print(f"Loading LoRA adapters from: {lora_model_path}...")
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    model.eval()
    
    return model, tokenizer

def get_prediction(model, tokenizer, precontext, sentence, ending, homonym, judged_meaning, target_token_ids, max_length=512):
    """
    Function to execute inference on a single example and return the continuous value (float).
    :param model: the loaded model with LoRA adapters
    :param tokenizer: the tokenizer
    :param precontext: the precontext text
    :param sentence: the ambiguous sentence
    :param ending: the sentence ending
    :param homonym: the homonym word
    :param judged_meaning: the judged meaning of the homonym
    :param target_token_ids: list of token IDs corresponding to the target outputs (1-5)
    :param max_length: maximum length for tokenization
    :return: continuous prediction as float
    """
    prompt = build_prompt_text(precontext, sentence, ending, homonym, judged_meaning)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)

    decoder_input_ids = torch.tensor([[model.config.pad_token_id]], device=model.device)

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=decoder_input_ids
        )

        # Extract logits for the first token generated
        logits = outputs.logits[:, 0, :].to(torch.float32)
        
        # Select logits corresponding to target token IDs
        relevant_logits = logits[:, target_token_ids]

        # Compute probabilities and continuous prediction
        # Softmax over relevant logits
        probs = torch.nn.functional.softmax(relevant_logits, dim=-1)
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=model.device, dtype=torch.float32)
        
        # Aggregate to get continuous prediction
        pred_cont = torch.sum(probs * weights, dim=-1)

        return pred_cont.item()