import os
import shutil
from huggingface_hub.constants import HF_HUB_CACHE
from transformers.utils import TRANSFORMERS_CACHE

# キャッシュディレクトリを表示
print(f"HuggingFace Hub キャッシュ: {HF_HUB_CACHE}")
print(f"Transformers キャッシュ: {TRANSFORMERS_CACHE}")

# 特定のモデルを削除する例
model_name = "llm-jp/llm-jp-13b-v2.0"
model_dir = model_name.replace("/", "--")
model_cache_path = os.path.join(TRANSFORMERS_CACHE, f"models--{model_dir}")

if os.path.exists(model_cache_path):
    print(f"削除中: {model_cache_path}")
    shutil.rmtree(model_cache_path)
    print("削除完了")
else:
    print(f"キャッシュにモデル {model_name} が見つかりません")