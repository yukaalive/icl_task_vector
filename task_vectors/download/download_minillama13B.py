from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "MiniLLM/MiniLLM-Llama-7B"

print(f"モデル '{model_name}' ロード開始")
try:
    # 1. トークナイザーのロード
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("トークナイザーのロード完了。")

    # 2. モデルのロード 
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
    )
    
    print(f"モデルは {model.device} にロードされました")

    # 3. テキスト生成
    prompt = "how to learn coading"
    print(f"\nプロンプト: '{prompt}'")

    # プロンプトをトークンIDに変換
    inputs_dict = tokenizer(prompt, return_tensors="pt")

    if 'token_type_ids' in inputs_dict:
        del inputs_dict['token_type_ids']

    # 辞書の各テンソルをモデルと同じデバイスに送る
    inputs_on_device = {k: v.to(model.device) for k, v in inputs_dict.items()}

    # モデルを使ってテキスト生成を実行
    outputs = model.generate(**inputs_on_device, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)

    # 生成されたトークンIDをるテキストに
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"answer:\n{generated_text}")

# try に対応する except ブロック (try と同じインデントレベル)
except ImportError:
    print(f"モデルのロードまたは生成中にエラーが発生しました: {e}")
