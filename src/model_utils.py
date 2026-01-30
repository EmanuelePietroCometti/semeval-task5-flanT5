import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType
import yaml

def load_base_model(model_name="google/flan-t5-xl"):
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
    with open("config/config.yaml" , "r") as config_file:
        config = yaml.safe_load(config_file)
    print("Applying LoRA config...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model