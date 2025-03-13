from typing import Dict, List, Optional, Tuple, Union, Iterable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from core.analysis.evaluation import calculate_accuracy_on_datasets
from core.data.datasets.few_shot_dataset import FewShotDataset
from core.data.tasks.task import Task
from core.models.context_managers.forward_modifiers.hidden_injector import HiddenInjector
from core.models.utils.inference import (
    batch_forward,
    batch_generate,
    decode_predictions,
    get_input_type,
    modified_forward,
    tokenize_datasets,
    traced_forward,
)
from core.models.utils.llm_layers import get_layers
from core.utils.nested import nested_apply

BATCH_SIZE = 4


def _move_inputs_to_device(inputs: dict, device: torch.device) -> dict:
    """
    入力辞書に含まれるテンソルをすべて指定デバイスへ移動するユーティリティ関数。
    """
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)
    return inputs


def run_icl(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    test_datasets: List[FewShotDataset],
    include_train: bool = True,
) -> List[str]:
    """
    (In-Context Learning) のベースライン実行例。
    """
    format_dataset_kwargs = {"include_train": include_train}
    inputs = tokenize_datasets(tokenizer, test_datasets, format_dataset_kwargs=format_dataset_kwargs)

    # 入力全体をモデルデバイスへ移動
    _move_inputs_to_device(inputs, model.device)

    # 例: サンプリングを使いたい場合のパラメータ（必要に応じて変更）
    generate_kwargs = {
        "do_sample": True,     # サンプリングするかどうか
        "temperature": 0.6,    # サンプリングの温度
        "top_p": 0.9,          # nucleus sampling パラメータ
        "max_new_tokens": 1,
        # num_return_sequences など追加してもよい
    }

    # 二重定義にならないよう、引数で do_sample は渡さない
    new_ids = batch_generate(
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        generate_kwargs=generate_kwargs,
        batch_size=BATCH_SIZE,
    )

    predictions = decode_predictions(new_ids, tokenizer)
    return predictions

def run_task_vector(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    test_datasets: List[FewShotDataset],
    dev_datasets: List[FewShotDataset],
    layers_to_test: Optional[Iterable[int]] = None,
    multi_context: bool = False,
):
    """
    タスクベクターを使い、dev セットで最も良い中間層を選び、
    その層に対して修正を加えて予測を行う。
    """
    # すべての層をテスト（layers_to_testはNoneのまま）
    dev_accuracy_by_layer = task_vector_accuracy_by_layer(
        model,
        tokenizer,
        task,
        dev_datasets,
        layers_to_test=layers_to_test,
        multi_context=multi_context,
    )
    
    # 実験結果の要約を表示
    print("\n=== 各層の精度サマリー ===")
    for layer, acc in sorted(dev_accuracy_by_layer.items()):
        print(f"層 {layer}: {acc:.4f}")
    
    best_intermediate_layer = int(max(dev_accuracy_by_layer, key=dev_accuracy_by_layer.get))
    best_accuracy = dev_accuracy_by_layer[best_intermediate_layer]
    print(f"\nDebug: 最適な中間層: {best_intermediate_layer} (精度: {best_accuracy:.4f})")

    task_hiddens = get_task_hiddens(
        model,
        tokenizer,
        task,
        test_datasets,
        multi_context=multi_context,
    )

    predictions = modulated_generate(
        model,
        tokenizer,
        task,
        test_datasets,
        task_hiddens=task_hiddens,
        intermediate_layer=best_intermediate_layer,
    )

    return predictions, dev_accuracy_by_layer, task_hiddens
def task_vector_accuracy_by_layer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    datasets: List[FewShotDataset],
    layers_to_test: Optional[Iterable[int]] = None,
    multi_context: bool = False,
) -> Dict[int, float]:
    """
    指定したデータセットに対して、各中間層を使ったときの精度を計算する。
    """
    try:
        # すべての層をテスト
        num_layers = len(get_layers(model))
        if layers_to_test is None:
            layers_to_test = range(num_layers)
        
        # リストに変換して長さを取得
        layers_list = list(layers_to_test)
        total_layers = len(layers_list)
        
        print(f"Debug: モデルは合計{num_layers}層あります")
        print(f"Debug: テスト対象の層は合計{total_layers}層です")

        # タスクベクター計算用 hidden states
        task_hiddens = get_task_hiddens(
            model,
            tokenizer,
            task,
            datasets,
            multi_context=multi_context,
        )

        # トークナイズ & デバイス移動
        inputs = tokenize_datasets(tokenizer, datasets, format_dataset_kwargs={"include_train": False})
        _move_inputs_to_device(inputs, model.device)

        # 通常の forward (バッチ対応)
        outputs = batch_forward(
            model=model,
            inputs=inputs,
            forward_kwargs={"use_cache": True},
            batch_size=BATCH_SIZE,
        )

        # 直近トークン以外を past_key_values 化
        past_key_values = outputs.past_key_values
        past_key_values = nested_apply(past_key_values, lambda x: x[:, :, :-1])

        # 最後のトークンだけ残して次の生成用 input_ids に
        inputs["input_ids"] = inputs["input_ids"][..., -1].unsqueeze(1)

        accuracies = []
        for i, layer_num in enumerate(layers_list):
            # 進行状況を表示
            progress = (i / total_layers) * 100
            progress_bar = "[" + "=" * int(progress / 5) + " " * (20 - int(progress / 5)) + "]"
            print(f"\rDebug: 実験進行状況: {progress_bar} {progress:.1f}% (層 {i+1}/{total_layers}, 現在: {layer_num})", end="")
            
            try:
                print(f"\nDebug: 層{layer_num}のテスト中...")
                answers = modulated_generate(
                    model=model,
                    tokenizer=tokenizer,
                    task=task,
                    datasets=datasets,
                    intermediate_layer=layer_num,
                    task_hiddens=task_hiddens,
                    past_key_values=past_key_values,
                )
                accuracy = calculate_accuracy_on_datasets(task, answers, datasets)
            except Exception as e:
                print(f"Error: 層{layer_num}のテスト中にエラーが発生しました: {e}")
                print("Error: このエラーをキャッチして次の層に進みます")
                answers = ["ERROR"] * len(datasets)
                accuracy = 0.0
                
                # GPUのキャッシュをクリア
                torch.cuda.empty_cache()
                
            accuracies.append(accuracy)
            print(f"Debug: 層{layer_num}の精度: {accuracy:.4f}")

        print("\nDebug: すべての層のテストが完了しました")
        accuracy_by_layer = {layer: acc for layer, acc in zip(layers_list, accuracies)}
        return accuracy_by_layer
        
    except Exception as e:
        print(f"Critical Error: 実験中に重大なエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        
        # できるだけ結果を返す
        if 'layers_list' in locals() and 'accuracies' in locals() and len(layers_list) == len(accuracies):
            accuracy_by_layer = {layer: acc for layer, acc in zip(layers_list, accuracies)}
            return accuracy_by_layer
        else:
            # 最低限の結果を返す
            return {0: 0.0}
        
def modulated_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    datasets: List[FewShotDataset],
    task_hiddens=None,
    intermediate_layer=None,
    past_key_values=None,
):
    """
    指定した中間層に隠れ状態を注入しながら生成を行う
    """
    print(f"Debug: modulated_generate開始 (層={intermediate_layer})")
    
    # 中間層が指定されていない場合は何もせずに通常生成
    if intermediate_layer is None:
        print("Warning: 中間層が指定されていません。通常の生成を行います。")
        # 通常のICL生成を行う
        return run_icl(model, tokenizer, task, datasets, include_train=False)
    
    # データセットの準備
    inputs = tokenize_datasets(tokenizer, datasets, format_dataset_kwargs={"include_train": False})
    _move_inputs_to_device(inputs, model.device)
    
    # past_key_valuesがない場合は通常のforward
    if past_key_values is None:
        outputs = batch_forward(
            model=model,
            inputs=inputs,
            forward_kwargs={"use_cache": True},
            batch_size=BATCH_SIZE,
        )
        past_key_values = outputs.past_key_values
        past_key_values = nested_apply(past_key_values, lambda x: x[:, :, :-1])
        inputs["input_ids"] = inputs["input_ids"][..., -1].unsqueeze(1)
    
    # モデルのデータ型を取得
    model_dtype = next(model.parameters()).dtype
    print(f"Debug: モデルのデータ型: {model_dtype}")
    
    # バッチに分けて処理
    all_predictions = []
    num_examples = inputs["input_ids"].shape[0]
    
    for batch_idx in range(0, num_examples, BATCH_SIZE):
        batch_end = min(batch_idx + BATCH_SIZE, num_examples)
        print(f"Debug: バッチ {batch_idx//BATCH_SIZE + 1} 処理中 ({batch_idx}〜{batch_end-1})")
        
        # バッチデータの抽出
        batch_inputs = {k: v[batch_idx:batch_end] for k, v in inputs.items()}
        batch_past = None
        if past_key_values is not None:
            batch_past = nested_apply(past_key_values, lambda x: x[batch_idx:batch_end])
        
        # 隠れ状態注入を設定
        batch_task_hiddens = None
        if task_hiddens is not None:
            batch_task_hiddens = task_hiddens[batch_idx:batch_end]
            # モデルのデータ型に合わせる
            batch_task_hiddens = batch_task_hiddens.to(dtype=model_dtype)
            print(f"Debug: バッチの隠れ状態の形状: {batch_task_hiddens.shape}")
        
        # HiddenInjectorコンテキストマネージャを使用して生成
        # 整数値の単一要素リストとして層を指定
        try:
            # intermediate_layerが整数であることを確認
            layer_idx = int(intermediate_layer)
            
            injector = HiddenInjector(
                layers=[layer_idx],  # 整数のリストとして渡す
                positions=[4],       # 最初のトークン位置を注入
                hiddens=batch_task_hiddens
            )
            
            # コンテキストマネージャを使用して生成
            with injector.apply_to(model):
                generate_kwargs = {
                    "do_sample": True,
                    "temperature": 0.6,
                    "top_p": 0.9,
                    "max_new_tokens": 1,
                    "past_key_values": batch_past
                }
                
                # デバッグのためにキーと値をログ出力
                print(f"Debug: generate_kwargs keys: {generate_kwargs.keys()}")
                print(f"Debug: batch_inputs keys: {batch_inputs.keys()}")
                
                outputs = model.generate(
                    **batch_inputs,
                    **generate_kwargs
                )
                
                # 予測結果を保存
                batch_predictions = tokenizer.batch_decode(outputs[:, -1:], skip_special_tokens=True)
                all_predictions.extend(batch_predictions)
        except Exception as e:
            print(f"警告: バッチ{batch_idx}〜{batch_end-1}の生成中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()  # スタックトレースを出力
            # エラー回避のため、ダミー予測を返す
            dummy_predictions = ["ERROR"] * (batch_end - batch_idx)
            all_predictions.extend(dummy_predictions)
    
    print(f"Debug: 予測完了、{len(all_predictions)}個の回答を生成")
    return all_predictions


def get_task_hiddens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    datasets: List[FewShotDataset],
    multi_context: bool = False,
):
    """
    タスク固有の隠れ状態を取得する
    """
    print(f"Debug: タスクhiddensを取得中...")
    
    if multi_context:
        print(f"Debug: 複数コンテキストからタスクhiddensを取得中...")
        # マルチコンテキスト用の実装
        all_context_hiddens = []
        
        for ctx_idx, dataset in enumerate(datasets):
            # このコンテキストのみを含むデータセット
            ctx_dataset = [dataset]
            
            # トークナイズ
            ctx_inputs = tokenize_datasets(tokenizer, ctx_dataset, format_dataset_kwargs={"include_train": True})
            _move_inputs_to_device(ctx_inputs, model.device)
            
            # トレース付きforward実行
            traced_outputs = traced_forward(
                model=model,
                inputs=ctx_inputs,
                forward_kwargs={"output_hidden_states": True}
            )
            
            # 隠れ状態を取得 - トレース出力フォーマットに対応
            hidden_states = extract_hidden_states(traced_outputs)
            all_context_hiddens.append(hidden_states)
        
        # すべてのコンテキストの隠れ状態を平均
        combined_hiddens = torch.stack(all_context_hiddens).mean(dim=0)
        print(f"Debug: マルチコンテキストhiddens取得完了: shape={combined_hiddens.shape}")
    else:
        print(f"Debug: 単一コンテキストからタスクhiddensを取得中...")
        # トークナイズ
        format_dataset_kwargs = {"include_train": True}
        inputs = tokenize_datasets(tokenizer, datasets, format_dataset_kwargs=format_dataset_kwargs)
        _move_inputs_to_device(inputs, model.device)
        
        # 隠れ状態を保存するための配列
        all_hiddens = []
        
        # バッチごとに処理
        num_examples = inputs["input_ids"].shape[0]
        for i in range(num_examples):
            print(f"Debug: データセット {i+1}/{num_examples} 処理中")
            batch_inputs = {k: v[i:i+1] for k, v in inputs.items()}
            
            # トレース付きのforward（隠れ状態を取得するため）
            traced_outputs = traced_forward(
                model=model, 
                inputs=batch_inputs,
                forward_kwargs={"output_hidden_states": True}
            )
            
            # 隠れ状態を取得 - トレース出力フォーマットに対応
            hidden_states = extract_hidden_states(traced_outputs)
            all_hiddens.append(hidden_states)
        
        # すべての隠れ状態をスタック
        combined_hiddens = torch.stack(all_hiddens)
        print(f"Debug: 単一コンテキストhiddens取得完了: shape={combined_hiddens.shape}")
    
    print(f"Debug: タスクhiddens取得完了: shape={combined_hiddens.shape}")
    return combined_hiddens


def extract_hidden_states(traced_outputs):
    """
    様々な形式のトレース出力から隠れ状態を抽出するヘルパー関数
    """
    # トレース出力がタプルの場合の対応
    if isinstance(traced_outputs, tuple):
        print(f"Debug: トレース出力はタプル型です (長さ: {len(traced_outputs)})")
        
        # 出力タプルの中を探索して隠れ状態を見つける
        for i, item in enumerate(traced_outputs):
            # オブジェクトにhidden_states属性がある場合
            if hasattr(item, 'hidden_states'):
                print(f"Debug: hidden_states属性を持つオブジェクトを出力タプルの第{i+1}要素として検出しました")
                # hidden_statesはタプルであることが多い、各層の隠れ状態のタプル
                # ここで直接タプルではなく、最後の層のテンソルを返す
                if isinstance(item.hidden_states, tuple):
                    print(f"Debug: hidden_statesはタプル型です (長さ: {len(item.hidden_states)})")
                    # 通常、最後の層の隠れ状態を使用
                    return item.hidden_states[-1]  # 最後の層のテンソルを返す
                else:
                    # タプルでない場合はそのまま返す
                    return item.hidden_states
        
        # どの要素も適切な隠れ状態が見つからない場合
        print(f"Debug: hidden_states属性を持つオブジェクトが見つかりませんでした。別の方法を試みます。")
        
        # 2番目の要素がタプルで、すべてがテンソルの場合（一般的なHuggingFaceモデルの形式）
        if len(traced_outputs) >= 2 and isinstance(traced_outputs[1], tuple) and all(isinstance(h, torch.Tensor) for h in traced_outputs[1]):
            print(f"Debug: hidden_statesを出力タプルの第2要素として検出しました")
            # 最後の層のテンソルを返す
            return traced_outputs[1][-1]
        
        # フォールバック：最初の要素が適切なテンソルの場合
        if isinstance(traced_outputs[0], torch.Tensor):
            print(f"Debug: 他に適切なhidden_statesが見つからないため、出力タプルの第1要素を使用します")
            return traced_outputs[0]
        
        # 最終フォールバック：すべての試行が失敗した場合、エラーを報告
        print(f"Warning: 適切な隠れ状態が見つかりません。トレース出力の構造をより詳細に調査してください。")
        print(f"Debug: トレース出力の詳細構造:")
        for i, item in enumerate(traced_outputs):
            print(f"  要素[{i}]: 型={type(item)}")
            if hasattr(item, '__dict__'):
                print(f"    属性: {dir(item)}")
        
        # エラー回避のためにダミーテンソルを返す
        print(f"Warning: ダミーの隠れ状態を返します。これは正しい結果ではありません。")
        return torch.zeros(1, 1, 1)
    
    # オブジェクトにhidden_states属性がある場合
    elif hasattr(traced_outputs, 'hidden_states'):
        print(f"Debug: トレース出力オブジェクトにhidden_states属性を検出しました")
        # hidden_statesがタプルの場合、最後の層を返す
        if isinstance(traced_outputs.hidden_states, tuple):
            return traced_outputs.hidden_states[-1]
        return traced_outputs.hidden_states
    
    # その他の場合はそのまま返す
    else:
        print(f"Debug: トレース出力は予期しない形式です: {type(traced_outputs)}")
        if isinstance(traced_outputs, torch.Tensor):
            return traced_outputs
        else:
            print(f"Warning: テンソル形式の隠れ状態が見つかりません。ダミーテンソルを返します。")
            return torch.zeros(1, 1, 1)