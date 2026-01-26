import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType

def load_base_model(model_name="google/flan-t5-large"):
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch.float32
    )
    return model, tokenizer

def apply_lora_config(model):
    print("Applying LoRA config...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=16,           # Rank
        lora_alpha=32,  # Alpha
        lora_dropout=0.1,
        target_modules=["q", "v", "wi_0", "wi_1", "wo", "lm_head"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model