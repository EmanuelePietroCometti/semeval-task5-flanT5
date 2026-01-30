import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType

def load_base_model(model_name="google/flan-t5-xl"):
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map=device_map,
        use_cache=False
    )
    return model, tokenizer

def apply_lora_config(model, r, alpha, dropout, target_modules):
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=target_modules
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model