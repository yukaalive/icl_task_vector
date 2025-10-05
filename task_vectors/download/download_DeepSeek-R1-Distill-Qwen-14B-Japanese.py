import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# 環境変数からトークンを取得
model_name = "cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese"
token = os.getenv("HUGGINGFACE_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token)
