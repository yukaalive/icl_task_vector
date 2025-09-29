"""
FewShotFormatのファクトリークラス
データセット固有のプロンプトを使用してFewShotFormatインスタンスを作成する
"""

from typing import Optional
from core.data.datasets.few_shot_format import FewShotFormat
from core.data.datasets.dataset_prompts import get_dataset_prompts

def extract_dataset_name_from_task(task_name: str) -> Optional[str]:
    """
    タスク名からデータセット名を抽出する
    Args:
        task_name: タスク名（例: "translation_ja_en"）
    Returns:
        データセット名（例: "ja_en"）、抽出できない場合はNone
    """
    # 翻訳タスクの場合
    if task_name.startswith("translation_"):
        # "translation_ja_en" -> "ja_en"
        return task_name.replace("translation_", "")
    
    # 他のタスクタイプも必要に応じて追加
    # 例: "linguistic_present_simple_gerund" -> "present_simple_gerund"
    # if task_name.startswith("linguistic_"):
    #     return task_name.replace("linguistic_", "")
    
    return None

def create_few_shot_format_for_task(task_name: str, base_format: Optional[FewShotFormat] = None) -> FewShotFormat:
    """
    タスク名に基づいてFewShotFormatインスタンスを作成する
    Args:
        task_name: タスク名（例: "translation_ja_en"）
        base_format: ベースとなるFewShotFormat（Noneの場合はデフォルト設定を使用）
    Returns:
        データセット固有のプロンプトが設定されたFewShotFormatインスタンス
    """
    # ベースフォーマットの設定を取得
    if base_format is None:
        base_format = FewShotFormat()
    
    # データセット名を抽出
    dataset_name = extract_dataset_name_from_task(task_name)
    
    # データセット固有のプロンプト設定を取得
    dataset_prompts = get_dataset_prompts()
    
    # FewShotFormatインスタンスを作成
    few_shot_format = FewShotFormat(
        example_format=base_format.example_format,
        example_separator=base_format.example_separator,
        task_description=base_format.task_description,
        test_example_format=base_format.test_example_format,
        dataset_specific_prompts=dataset_prompts,
    )
    
    return few_shot_format

def create_few_shot_format_with_prompts(dataset_specific_prompts: dict) -> FewShotFormat:
    """
    指定されたデータセット固有のプロンプトでFewShotFormatインスタンスを作成する
    Args:
        dataset_specific_prompts: データセット固有のプロンプト設定辞書
    Returns:
        FewShotFormatインスタンス
    """
    return FewShotFormat(dataset_specific_prompts=dataset_specific_prompts)
