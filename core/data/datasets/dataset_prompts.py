"""
データセット固有のプロンプト設定
各データセットに対して異なるプロンプトフォーマットを定義する
"""

from typing import Dict

# データセット固有のプロンプト設定
DATASET_SPECIFIC_PROMPTS: Dict[str, Dict[str, str]] = {
    "ja_en": {
        "task_description": "Follow the example to translate from Japanese to English.",
        "example_format": "example:{input}->{output}",
        "test_example_format": "example:{input}->",
    },
    "en_ja": {
        "task_description": "Follow the example to translate from English to Japanese.",
        "example_format": "example:{input}->{output}",
        "test_example_format": "example:{input}->",
    },
    "es_en": {
        "task_description": "Follow the example to translate from Spanish to English.",
        "example_format": "example:{input}->{output}",
        "test_example_format": "example:{input}->",
    },
    "en_es":{
        "task_description": "Follow the example to translate from English to Spanish.",
        "example_format": "example:{input}->{output}",
        "test_example_format": "example:{input}->",
    },
    "fr_en": {
        "task_description": "Follow the example to translate from French to English.",
        "example_format": "example:{input}->{output}",
        "test_example_format": "example:{input}->",
    },
    "en_fr": {
        "task_description": "Follow the example to translate from English to French.",
        "example_format": "example:{input}->{output}",
        "test_example_format": "example:{input}->",
    },
}

def get_dataset_prompts() -> Dict[str, Dict[str, str]]:
    """
    データセット固有のプロンプト設定を取得する
    Returns:
        データセット固有のプロンプト設定辞書
    """
    return DATASET_SPECIFIC_PROMPTS

def get_prompt_for_dataset(dataset_name: str, prompt_type: str) -> str:
    """
    特定のデータセットとプロンプトタイプに対するプロンプトを取得する
    Args:
        dataset_name: データセット名（例: "ja_en"）
        prompt_type: プロンプトタイプ（"task_description", "example_format", "test_example_format"）
    Returns:
        指定されたプロンプト文字列、見つからない場合はNone
    """
    return DATASET_SPECIFIC_PROMPTS.get(dataset_name, {}).get(prompt_type)
