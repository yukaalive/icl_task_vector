import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# モデルとトークナイザーをロード
model_name = "llm-jp/llm-jp-13b-v2.0"

# モデルとトークナイザーをロード
tokenizer = AutoTokenizer.from_pretrained(model_name)  # token=Trueは不要です
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# プロンプトを設定
prompt = "日本の首都はどこですか？"
# トークナイズ
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# token_type_idsを取り除く
if "token_type_ids" in inputs:
    inputs.pop("token_type_ids")

# テキスト生成
output = model.generate(
    **inputs, 
    max_length=200,  # 長さを増やす
    temperature=0.7,  # 創造性と一貫性のバランス
    top_p=0.9,       # 多様性を制御
    do_sample=True,  # サンプリングを有効に
    num_return_sequences=1  # 生成する文章の数
)
# 結果を表示
print("Generated text:")
print(tokenizer.decode(output[0], skip_special_tokens=True))