import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# モデルとトークナイザーをロード
model_name = "haoranxu/X-ALMA-13B-Group6"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=True,
    use_fast=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # メモリ効率を上げるためにfloat16を使用
    device_map="auto",
    low_cpu_mem_usage=True
)

# 改善されたプロンプト
prompt = """
### 質問:
数学の勉強方法を教えてください。

### 回答:
"""

# トークナイズ
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 改善された生成パラメータ
output = model.generate(
    **inputs,
    max_length=500,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id  # 適切なパディングを設定
)

# 結果を表示
print("Generated text:")
print(tokenizer.decode(output[0], skip_special_tokens=True))