import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# モデルとトークナイザーをロード
model_name = "rinna/llama-3-youko-8b"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# プロンプトを設定
prompt = "運転手の安全な車で移動する->I travel in the driver's safe car.世の中には善と悪が存在する。->"
# トークナイズ
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# テキスト生成
output = model.generate(**inputs, max_length=100)

# 結果を表示
print("Generated text:")
print(tokenizer.decode(output[0], skip_special_tokens=True))
