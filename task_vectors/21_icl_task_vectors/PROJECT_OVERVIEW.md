# 21_icl_task_vectors プロジェクト概要

## プロジェクト説明

このプロジェクトは、論文「[In-Context Learning Creates Task Vectors](https://arxiv.org/abs/2310.15916)」(Roee Hendel, Mor Geva, Amir Globerson, 2023) の公式実装リポジトリです。

In-Context Learning (ICL) がタスクベクトルを生成する仕組みを研究し、言語モデルの内部表現を分析するための実験環境を提供します。

## 主な特徴

- **タスクベクトル抽出**: ICLを通じてモデルの隠れ状態からタスクベクトルを抽出
- **翻訳タスク中心**: 日英、英日、英西、英仏などの翻訳タスクに焦点
- **多様なモデル対応**: LLaMA, Pythia, Swallow, Youkoなど複数のLLMに対応
- **評価メトリクス**: BLEU、COMETスコアによる包括的な翻訳品質評価

## プロジェクト構造

```
21_icl_task_vectors/
├── core/                           # コアライブラリ
│   ├── task_vectors.py            # タスクベクトル抽出・実行のメイン実装
│   ├── experiments_config.py      # 実験設定（評価対象モデル・タスク）
│   ├── config.py                  # 基本設定
│   ├── analysis/                  # 分析ツール
│   │   ├── evaluation.py         # 精度評価
│   │   └── utils.py              # 分析ユーティリティ
│   ├── data/                      # データ処理
│   │   ├── datasets/             # データセット定義
│   │   │   ├── few_shot_dataset.py
│   │   │   └── few_shot_format.py
│   │   ├── tasks/                # タスク定義
│   │   │   ├── translation_task.py  # 翻訳タスク実装（COMET/BLEU評価含む）
│   │   │   ├── task.py
│   │   │   └── ...
│   │   ├── preparation/          # データ準備
│   │   │   ├── translation_data.py
│   │   │   ├── knowledge_data.py
│   │   │   └── linguistic_data.py
│   │   └── task_helpers.py       # タスクヘルパー関数
│   ├── models/                    # モデル関連
│   │   ├── llm_loading.py        # モデルロード
│   │   ├── context_managers/     # フォワード処理の変更
│   │   │   ├── forward_modifiers/
│   │   │   │   ├── hidden_injector.py  # 隠れ状態の注入
│   │   │   │   ├── attention_masker.py
│   │   │   │   └── layer_disabler.py
│   │   │   └── tracing/
│   │   │       ├── forward_tracer.py
│   │   │       └── forward_trace.py
│   │   └── utils/
│   │       ├── inference.py      # 推論ユーティリティ
│   │       └── llm_layers.py     # レイヤー操作
│   └── utils/                     # 汎用ユーティリティ
│       ├── misc.py
│       └── nested.py
│
├── scripts/                       # 実験スクリプト
│   ├── experiments/
│   │   ├── main.py               # メイン実験（タスクベクトル評価）
│   │   ├── overriding.py         # 競合タスク実験
│   │   └── task_vectors_robustness.py  # ロバストネス実験
│   ├── figures/                   # 図生成スクリプト
│   │   ├── main.py
│   │   └── overriding.py
│   ├── data/
│   │   └── prepare_data.py       # データ準備スクリプト
│   └── utils.py
│
├── data/                          # データファイル
│   ├── translation/               # 翻訳データ
│   │   ├── en_ja.json
│   │   ├── ja_en.json
│   │   ├── en_es.json
│   │   ├── en_fr.json
│   │   ├── es_en.json
│   │   ├── fr_en.json
│   │   └── ...
│   ├── knowledge/                 # 知識タスクデータ
│   └── linguistic/                # 言語学タスクデータ
│
├── raw/                           # 生データ（JESC日英翻訳コーパス）
│   ├── raw                        # 大規模翻訳コーパス（~224MB）
│   ├── short_200.tsv             # サンプルデータ
│   └── data.py                    # データ処理スクリプト
│
├── outputs/                       # 出力ディレクトリ
│   ├── results/                   # 実験結果（pickleファイル）
│   └── figures/                   # 生成された図
│
├── logs/                          # ログファイル
│
├── test_translation_evaluation.py # 翻訳評価テストスクリプト
├── exploration.ipynb              # 探索的分析ノートブック
├── run_script.sh                  # 実験実行スクリプト
├── stop_script.sh                 # 実験停止スクリプト
├── environment.yml                # Conda環境定義
├── requirements.in                # Python依存関係
├── git_workflow_guide.md          # Gitワークフローガイド（日本語）
├── a_memo.md                      # メモ
└── README.md                      # プロジェクトREADME
```

## 実験設定（experiments_config.py）

### 評価対象タスク
```python
TASKS_TO_EVALUATE = [
    "translation_ja_en",   # 日本語→英語
    "translation_en_ja",   # 英語→日本語
    "translation_es_en",   # スペイン語→英語
    "translation_en_fr",   # 英語→フランス語
    "translation_en_es",   # 英語→スペイン語
]
```

### 評価対象モデル
```python
MODELS_TO_EVALUATE = [
    ("swallow", "7B"),     # Swallow 7B
    ("pythia", "2.8B"),    # Pythia 2.8B
    ("llama", "7B"),       # LLaMA 7B
    ("youko", "8B"),       # Youko 8B
    ("pythia", "6.9B"),    # Pythia 6.9B
    ("llama", "13B"),      # LLaMA 13B
    ("pythia", "12B"),     # Pythia 12B
]
```

## 主要機能

### 1. タスクベクトル生成と評価

**core/task_vectors.py** の主要関数:

- `run_icl()`: In-Context Learningの実行
- `run_task_vector()`: タスクベクトルの抽出と適用
- タスク固有の隠れ状態を抽出し、それをテスト時に注入することでICLを再現

### 2. 翻訳品質評価

**core/data/tasks/translation_task.py** に実装:

- **BLEU**: 標準的な機械翻訳評価メトリクス
- **COMET**: 最新のニューラル翻訳評価モデル
  - `Unbabel/wmt22-comet-da` モデルを使用
  - ソース文、予測文、参照文を総合的に評価

### 3. レイヤーごとの分析

各トランスフォーマーレイヤーでのタスクベクトルの効果を分析:

- レイヤーごとの精度変化の追跡
- トップ予測トークンの分析
- 隠れ状態の可視化

## 実験の実行方法

### 1. メイン実験（タスクベクトル評価）

```bash
./run_script.sh experiments.main
```

- 定義されたすべてのモデルとタスクに対して評価
- ベースライン、ICL、タスクベクトル手法を比較
- 結果は `outputs/results/main/<experiment_id>/` に保存

### 2. 競合タスク実験

```bash
./run_script.sh experiments.overriding
```

- 複数のタスクが競合する状況での挙動を分析
- 結果は `outputs/results/overriding/` に保存

### 3. ロバストネス実験

```bash
./run_script.sh experiments.task_vector_robustness
```

- タスクベクトルの頑健性を評価
- 結果と図は `outputs/figures/` に保存

### 4. 図の生成

```bash
# メイン実験の図
python scripts/figures/main.py

# 競合タスク実験の図
python scripts/figures/overriding.py
```

## データ準備

### 翻訳データの準備

データは既にリポジトリに含まれていますが、再生成する場合:

```bash
python scripts/data/prepare_data.py
```

### 日英翻訳コーパス（JESC）

- `raw/raw`: 大規模な日英対訳コーパス（約224MB）
- `raw/short_200.tsv`: サンプルデータ（200行）
- 日本語と英語の映画字幕データから構築

## テストスクリプト

### 翻訳評価のテスト

```bash
python test_translation_evaluation.py
```

このスクリプトは以下をテスト:
- 英仏、英西、英日翻訳の評価
- BLEUスコアの計算
- COMETスコアの計算
- 詳細な評価結果の表示

## 評価メトリクス

### 精度評価
- **Baseline Accuracy**: Few-shot例なしでの精度
- **ICL Accuracy**: 通常のIn-Context Learningでの精度
- **Task Vector (TV) Accuracy**: タスクベクトル手法での精度

### 翻訳品質評価
- **BLEU**: n-gramベースの表層的な一致度
- **COMET**: ニューラルモデルによる意味的な品質評価（0-1スコア）

### レイヤー分析
- **Layer-wise Accuracy**: 各レイヤーでのタスクベクトル効果
- **Top Tokens Analysis**: 各レイヤーで影響を受けたトップトークン

## 環境設定

### Conda環境

```bash
conda env create -f environment.yml -n icl_task_vectors
conda activate icl_task_vectors
```

### LLaMAモデルのセットアップ

1. LLaMAの公式ウェイトをダウンロード
2. 環境変数 `LLAMA_DIR` を設定
3. HuggingFace形式に変換して `<LLAMA_DIR>/huggingface/` に配置

### Docker環境

```bash
docker exec -it umezawa-icl_task_vectors bash
```

## 主要な技術実装

### タスクベクトルの仕組み

1. **Few-shot例の処理**: ICL時の隠れ状態をトレース
2. **タスクベクトル抽出**: 隠れ状態の平均を計算
3. **テスト時注入**: `HiddenInjector` を使用して隠れ状態を注入
4. **レイヤーごと評価**: 各レイヤーでの効果を個別に測定

### モデルの推論フロー

```
入力テキスト
  ↓
トークン化
  ↓
モデルのフォワードパス（トレース/変更可能）
  ↓
隠れ状態の抽出/注入
  ↓
ロジット → トークン予測
  ↓
デコード → 出力テキスト
  ↓
評価（精度/BLEU/COMET）
```

## ログとデバッグ

- **logs/**: 実験ログが保存される
- **outputs/results/**: 実験結果（pickleファイル）
- 各モデル・タスクごとに個別のファイルで結果を保存

## 連絡先

論文著者への連絡: roeehendel@mail.tau.ac.il

## 参考情報

- **論文**: [In-Context Learning Creates Task Vectors](https://arxiv.org/abs/2310.15916)
- **Git ワークフロー**: `git_workflow_guide.md` を参照

## 注意事項

1. 実験には大規模なLLMが必要なため、十分なGPUメモリが必要
2. `core/experiments_config.py` で評価対象を調整可能
3. データセット数は開発時に削減されている（50データセット）が、本番では400/100に変更推奨
4. COMET評価には追加のモデルダウンロードが必要

## 今後の拡張可能性

- より多くの言語ペアの追加
- 言語学的タスク（linguistic）の有効化
- 知識タスク（knowledge）の有効化
- アルゴリズムタスク（algorithmic）の追加
- より大規模なモデル（LLaMA 30B+）の評価
