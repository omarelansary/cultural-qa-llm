import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_llama_model(model_name):
    print(f"🔄 Loading model: {model_name}...")
    
    # 1. Quantization Config (For 4-bit loading)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token # Llama 3 needs a pad token

    # 3. Load Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=bnb_config, 
        device_map="auto",
        use_cache=False # distinct for training
    )
    
    print("✅ Model loaded successfully!")
    return model, tokenizer