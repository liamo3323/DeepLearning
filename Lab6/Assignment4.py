from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

base_model = "facebook/opt-2.7b"
adapter_model = "dfurman/Mistral-7B-Instruct-v0.2"

model = AutoModelForCausalLM.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, adapter_model)
tokenizer = AutoTokenizer.from_pretrained(base_model)

model = model.to("cuda")
model.eval()