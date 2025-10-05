import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# モデルとトークナイザーをロード
model_name = "rinna/nekomata-7b"
# trust_remote_code=Trueを追加して、モデルのカスタムコードを実行できるようにします
tokenizer = AutoTokenizer.from_pretrained(model_name, token=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

# プロンプトを設定
prompt = "明日は休みなので、"
# トークナイズ
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# テキスト生成
output = model.generate(**inputs, max_length=100)

# 結果を表示
print("Generated text:")
print(tokenizer.decode(output[0], skip_special_tokens=True))