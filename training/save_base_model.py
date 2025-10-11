from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = 'Qwen/Qwen3-8B'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

local_model_path = './saved-Qwen3-model-8B'
tokenizer.save_pretrained(local_model_path)
model.save_pretrained(local_model_path)

print(f"模型已保存到 {local_model_path}")