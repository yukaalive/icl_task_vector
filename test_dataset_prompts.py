"""
データセット固有のプロンプト機能のテストスクリプト
"""

from core.data.datasets.few_shot_dataset import FewShotDataset
from core.data.datasets.few_shot_format import FewShotFormat
from core.data.datasets.format_factory import create_few_shot_format_for_task, extract_dataset_name_from_task
from core.data.datasets.dataset_prompts import get_dataset_prompts

def test_dataset_name_extraction():
    """データセット名の抽出をテストする"""
    print("=== データセット名抽出のテスト ===")
    
    test_cases = [
        "translation_ja_en",
        "translation_en_ja", 
        "translation_es_en",
        "linguistic_present_simple_gerund",
        "knowledge_country_capital"
    ]
    
    for task_name in test_cases:
        dataset_name = extract_dataset_name_from_task(task_name)
        print(f"タスク名: {task_name} -> データセット名: {dataset_name}")
    print()

def test_dataset_specific_prompts():
    """データセット固有のプロンプトをテストする"""
    print("=== データセット固有プロンプトのテスト ===")
    
    # テスト用のデータセットを作成
    test_dataset = FewShotDataset(
        train_inputs=["こんにちは", "ありがとう"],
        train_outputs=["Hello", "Thank you"],
        test_input="さようなら",
        test_output="Goodbye"
    )
    
    # デフォルトのフォーマット
    print("--- デフォルトフォーマット ---")
    default_format = FewShotFormat()
    default_prompt = default_format.format_dataset(test_dataset)
    print(f"デフォルト:\n{default_prompt}\n")
    
    # ja_en固有のフォーマット
    print("--- ja_en固有フォーマット ---")
    ja_en_format = create_few_shot_format_for_task("translation_ja_en")
    ja_en_prompt = ja_en_format.format_dataset(test_dataset, dataset_name="ja_en")
    print(f"ja_en固有:\n{ja_en_prompt}\n")
    
    # 比較のため、データセット名なしでja_en_formatを使用
    print("--- ja_en_formatをデータセット名なしで使用 ---")
    ja_en_prompt_no_name = ja_en_format.format_dataset(test_dataset)
    print(f"データセット名なし:\n{ja_en_prompt_no_name}\n")

def test_prompt_configuration():
    """プロンプト設定をテストする"""
    print("=== プロンプト設定のテスト ===")
    
    dataset_prompts = get_dataset_prompts()
    print("利用可能なデータセット固有プロンプト:")
    for dataset_name, prompts in dataset_prompts.items():
        print(f"  {dataset_name}:")
        for prompt_type, prompt_text in prompts.items():
            print(f"    {prompt_type}: {prompt_text}")
    print()

def main():
    """メイン関数"""
    print("データセット固有プロンプト機能のテスト開始\n")
    
    test_dataset_name_extraction()
    test_prompt_configuration()
    test_dataset_specific_prompts()
    
    print("テスト完了")

if __name__ == "__main__":
    main()
