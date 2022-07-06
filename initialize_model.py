import torch
from arguments import InitializationArguments
from transformers import GPT2Config, AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

# Configuration
parser = HfArgumentParser(InitializationArguments)
args = parser.parse_args()

# Load codeparrot tokenizer trained for Python code tokenization
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

# Config: "scale_attn_by_layer_idx" and "reorder_and_upcast_attn" are Mistral stability tweaks
config_kwargs = {
    "vocab_size": len(tokenizer),
    "n_positions": 1024,
    "n_embd": 1024,
    "n_layer": 20,
    "n_head": 16,
    "scale_attn_by_inverse_layer_idx": True,
    "reorder_and_upcast_attn": True,
    "bos_token_id": 2,
    "eos_token_id":2
}

# Load model config (GPT-2 large in this case)
config = GPT2Config(**config_kwargs)

# Initialize new model with config
model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)

# Save model to the hub
model.save_pretrained(args.model_name, push_to_hub=args.push_to_hub)
