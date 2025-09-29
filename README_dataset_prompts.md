# データセット固有プロンプト機能

このドキュメントでは、データセット別に異なるプロンプトを使用する機能について説明します。

## 概要

従来は1つの`example_format`を使用していましたが、この機能により同じ実験を2つの異なるプロンプトで比較できるようになりました。データセット別に異なるプロンプトを設定することが可能です。

## 主な変更点

### 1. FewShotFormatクラスの拡張
- `dataset_specific_prompts`パラメータを追加
- データセット名に基づいて適切なプロンプトを選択する機能を追加

### 2. 新しいファイル
- `core/data/datasets/dataset_prompts.py`: データセット固有のプロンプト設定
- `core/data/datasets/format_factory.py`: FewShotFormatインスタンス作成のヘルパー関数
- `test_dataset_prompts.py`: テストスクリプト

## 使用方法

### 実験の実行方法

修正されたメイン実験スクリプトを使用して、各モデルで2回の実験（ノーマル→拡張）を自動実行できます：

```bash
# 全モデルで実験を実行
cd 2025workspace/task_vectors/20_icl_task_vectors
python scripts/experiments/main.py

# 特定のモデルで実験を実行（例：モデル番号0）
python scripts/experiments/main.py 0

# 特定のモデルで実験を実行（例：pythia 2.8B）
python scripts/experiments/main.py pythia 2.8B
```

### 実験の流れ

1. **1回目**: 全タスクをノーマルプロンプト（デフォルト）で実行
   - ICL実験とTask Vector実験の両方を実行
   - 結果は `{model_type}_{model_variant}.pkl` に保存

2. **2回目**: 全タスクを拡張プロンプト（データセット固有）で実行
   - ICL実験とTask Vector実験の両方を実行
   - 結果は `{model_type}_{model_variant}_dataset_specific.pkl` に保存

### プログラムでの使用方法

```python
from core.data.datasets.format_factory import create_few_shot_format_for_task
from core.data.datasets.few_shot_format import FewShotFormat

# タスク名からデータセット固有のフォーマットを作成
few_shot_format = create_few_shot_format_for_task("translation_ja_en")

# データセット名を指定してフォーマット
prompt = few_shot_format.format_dataset(dataset, dataset_name="ja_en")
```

### 設定されているデータセット固有プロンプト

現在、以下の翻訳タスクに対してデータセット固有のプロンプトが設定されています：

- **ja_en**: "Follow the example to translate from Japanese to English."
- **en_ja**: "Follow the example to translate from English to Japanese."
- **es_en**: "Follow the example to translate from Spanish to English."
- **en_es**: "Follow the example to translate from English to Spanish."
- **fr_en**: "Follow the example to translate from French to English."
- **en_fr**: "Follow the example to translate from English to French."

### 新しいデータセット固有プロンプトの追加

`core/data/datasets/dataset_prompts.py`の`DATASET_SPECIFIC_PROMPTS`辞書に新しいエントリを追加します：

```python
DATASET_SPECIFIC_PROMPTS = {
    "新しいデータセット名": {
        "task_description": "タスクの説明",
        "example_format": "example:{input}->{output}",
        "test_example_format": "example:{input}->",
    },
    # 既存の設定...
}
```

## 実装の詳細

### データセット名の抽出
タスク名からデータセット名を自動的に抽出します：
- `translation_ja_en` → `ja_en`
- `translation_es_en` → `es_en`

### プロンプトの優先順位
1. データセット固有のプロンプト（存在する場合）
2. デフォルトのプロンプト

## テスト

実装をテストするには：

```bash
cd 2025workspace/task_vectors/20_icl_task_vectors
python test_dataset_prompts.py
```

## 例：ja_enデータセットの場合

### デフォルトフォーマット
```
example:こんにちは->Hello
example:ありがとう->Thank you
example:さようなら->
```

### ja_en固有フォーマット
```
Follow the example to translate from Japanese to English.
example:こんにちは->Hello
example:ありがとう->Thank you
example:さようなら->
```

## 注意事項

- データセット名を指定しない場合は、デフォルトのフォーマットが使用されます
- 新しいデータセット固有プロンプトを追加する際は、テストスクリプトで動作確認を行ってください
- 既存のコードとの互換性を保つため、従来の使用方法も引き続きサポートされています
