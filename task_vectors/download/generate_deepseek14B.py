import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# モデルとトークナイザーをロード
model_name = "cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# プロンプトを設定
prompt = "pythonの勉強の方法を教えてください。"

# トークナイズ
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

if 'token_type_ids' in inputs:
    del inputs['token_type_ids']
inputs = inputs.to("cuda")

# テキスト生成
output = model.generate(**inputs, max_length=300)

# 結果を表示
print("Generated text:")
print(tokenizer.decode(output[0], skip_special_tokens=True))
