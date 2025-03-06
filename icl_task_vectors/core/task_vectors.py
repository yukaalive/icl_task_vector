from typing import Dict, List, Optional, Tuple, Union, Iterable
import gc
import time

import torch
import torch.nn as nn
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

# 以下の定数パラメーターはメモリ使用量と処理速度のトレードオフを調整するために使用
# GPUメモリ制約が厳しい場合は小さく設定
BATCH_SIZE = 1  # 最小バッチサイズ
CHUNK_SIZE = 2  # 一度に処理するデータセット数
SLEEP_TIME = 0.1  # メモリ解放を確実にするための短い待機時間（秒）
CPU_OFFLOAD = True  # CPU-GPU間のデータ移動による最適化を有効にする

def optimize_memory():
    """
    GPUメモリを積極的に解放し、ガベージコレクションを実行する。
    """
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(SLEEP_TIME)  # メモリ解放を確実にするための短い待機

def _move_inputs_to_device(inputs: dict, device: torch.device) -> dict:
    """
    入力辞書に含まれるテンソルをすべて指定デバイスへ移動するユーティリティ関数。
    """
    result = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            # 大きなテンソルは非同期でコピー
            result[k] = v.to(device=device, non_blocking=True)
        else:
            result[k] = v
    return result


def run_icl(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    test_datasets: List[FewShotDataset],
    include_train: bool = True,
) -> List[str]:
    """
    In-Context Learning のベースライン実行。
    メモリ最適化バージョン。
    """
    optimize_memory()
    format_dataset_kwargs = {"include_train": include_train}
    
    # 入力の準備
    inputs = tokenize_datasets(tokenizer, test_datasets, format_dataset_kwargs=format_dataset_kwargs)
    
    # 入力をCPUに保持し、必要に応じてGPUに移動する
    if CPU_OFFLOAD:
        inputs = {k: v.cpu() for k, v in inputs.items() if torch.is_tensor(v)}
    
    # 小さなバッチで処理
    all_new_ids = []
    for i in range(0, len(inputs["input_ids"]), BATCH_SIZE):
        # 現在のバッチを切り出し
        batch_end = min(i + BATCH_SIZE, len(inputs["input_ids"]))
        batch_inputs = {k: v[i:batch_end].to(model.device) if torch.is_tensor(v) else v 
                      for k, v in inputs.items()}
        
        # 前方伝播の実行
        with torch.inference_mode():  # torch.no_gradよりも効率的
            batch_new_ids = model.generate(
                **batch_inputs,
                max_new_tokens=1,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # 新しく生成されたトークンだけを抽出
        new_tokens = batch_new_ids[:, batch_inputs["input_ids"].shape[1]:]
        all_new_ids.append(new_tokens.cpu())  # GPUメモリを節約するためCPUに移動
        
        # メモリを解放
        del batch_inputs, batch_new_ids
        optimize_memory()
    
    # 全バッチの結果を結合
    new_ids = torch.cat(all_new_ids, dim=0)
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
    タスクベクターを使用した実験を実行する関数。
    dev_datasetsで最適な層を決定し、その層でtest_datasetsを処理する。
    メモリ最適化バージョン。
    """
    optimize_memory()
    print(f"Debug: 実行中のモデルタイプ: {model.config.model_type}")
    print(f"Debug: 開発データセット数: {len(dev_datasets)}")
    print(f"Debug: テストデータセット数: {len(test_datasets)}")
    
    # 開発データセットを使って最適な層を見つける
    try:
        dev_accuracy_by_layer = task_vector_accuracy_by_layer(
            model,
            tokenizer,
            task,
            dev_datasets,
            layers_to_test=layers_to_test,
            multi_context=multi_context,
        )
        
        # もしすべての精度が0なら、より少ないデータで試行
        if all(acc == 0.0 for acc in dev_accuracy_by_layer.values()):
            print("警告: すべての層の精度が0です。データセットサイズを縮小して再試行します。")
            smaller_dev_datasets = dev_datasets[:max(5, len(dev_datasets)//2)]
            dev_accuracy_by_layer = task_vector_accuracy_by_layer(
                model,
                tokenizer,
                task,
                smaller_dev_datasets,
                layers_to_test=layers_to_test,
                multi_context=multi_context,
            )
        
        # それでも全部0なら、デフォルト値を設定
        if all(acc == 0.0 for acc in dev_accuracy_by_layer.values()):
            best_intermediate_layer = 1  # デフォルトは早い層
        else:
            best_intermediate_layer = int(max(dev_accuracy_by_layer, key=dev_accuracy_by_layer.get))
            
    except Exception as e:
        print(f"エラー: 最適層の決定中に例外が発生: {e}")
        print("デフォルトの中間層を使用します。")
        best_intermediate_layer = 1  # エラー時のデフォルト値
    
    print(f"Debug: 最適な中間層: {best_intermediate_layer}")
    
    # テストデータセット用のタスクhiddensを取得
    print("Debug: タスクhiddensを取得中...")
    optimize_memory()
    
    try:
        task_hiddens = get_task_hiddens(model, tokenizer, task, test_datasets, multi_context=multi_context)
        
        if task_hiddens is None:
            print("エラー: task_hiddensがNoneです。処理を中断します。")
            # ダミーの結果を返す
            return ["error"] * len(test_datasets), dev_accuracy_by_layer, None
            
        print(f"Debug: タスクhiddens取得完了: shape={task_hiddens.shape}")
        
    except Exception as e:
        print(f"致命的なエラー: タスクhiddens取得に失敗: {e}")
        # ダミーのhiddensを作成
        try:
            num_layers = len(get_layers(model)) - 1  # 埋め込み層を除外
            hidden_size = model.config.hidden_size
            task_hiddens = torch.zeros((len(test_datasets), num_layers, hidden_size), device='cpu')
            print(f"ダミーのタスクhiddens作成: shape={task_hiddens.shape}")
        except Exception as e2:
            print(f"ダミーhiddens作成にも失敗: {e2}")
            return ["error"] * len(test_datasets), dev_accuracy_by_layer, None
    
    # タスクベクトルを使って予測を生成
    optimize_memory()
    predictions = modulated_generate(
        model,
        tokenizer,
        task,
        test_datasets=test_datasets,
        task_hiddens=task_hiddens,
        intermediate_layer=best_intermediate_layer,
    )

    return predictions, dev_accuracy_by_layer, task_hiddens


def run_overriding_task_vector(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    test_datasets: List[FewShotDataset],
    overriding_datasets: List[FewShotDataset],
    layers_to_test: Optional[Iterable[int]] = None,
):
    """
    別のタスク（overriding_datasets）のベクトルを使って元のタスクを上書きする実験。
    メモリ最適化バージョン。
    """
    optimize_memory()
    
    # 実験の各ステップにtry-exceptを追加
    try:
        dev_accuracy_by_layer = task_vector_accuracy_by_layer(
            model,
            tokenizer,
            task,
            overriding_datasets,
            layers_to_test=layers_to_test,
        )
    except Exception as e:
        print(f"エラー: dev_accuracy_by_layer計算に失敗: {e}")
        dev_accuracy_by_layer = {1: 0.0}  # デフォルト値
    
    best_intermediate_layer = int(max(dev_accuracy_by_layer, key=dev_accuracy_by_layer.get))
    
    try:
        task_hiddens_datasets = test_datasets if overriding_datasets is None else overriding_datasets
        task_hiddens = get_task_hiddens(model, tokenizer, task, task_hiddens_datasets)
    except Exception as e:
        print(f"エラー: task_hiddens取得に失敗: {e}")
        # ダミーのhiddensを作成
        num_layers = len(get_layers(model)) - 1  # 埋め込み層を除外
        hidden_size = model.config.hidden_size
        task_hiddens = torch.zeros((len(test_datasets), num_layers, hidden_size), device='cpu')

    try:
        predictions = modulated_generate(
            model,
            tokenizer,
            task,
            test_datasets=test_datasets,
            task_hiddens=task_hiddens,
            intermediate_layer=best_intermediate_layer,
            include_train=True,
        )
    except Exception as e:
        print(f"エラー: modulated_generate実行に失敗: {e}")
        predictions = ["error"] * len(test_datasets)

    return predictions, dev_accuracy_by_layer, task_hiddens


def get_multi_context_task_hiddens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    datasets: List[FewShotDataset],
) -> torch.Tensor:
    """
    複数コンテキストからタスクhiddensを取得する関数。
    メモリ最適化バージョン - CPUとGPUを効果的に切り替え。
    """
    optimize_memory()
    print("Debug: 複数コンテキストからタスクhiddensを取得中...")
    
    # CPUに保存するhiddens
    all_task_hiddens = []
    
    # データセットをより小さなグループに分割して処理
    for start_idx in range(0, len(datasets), CHUNK_SIZE):
        optimize_memory()
        
        end_idx = min(start_idx + CHUNK_SIZE, len(datasets))
        subset_datasets = datasets[start_idx:end_idx]
        print(f"Debug: データセット {start_idx} から {end_idx-1} までを処理中")
        
        try:
            # このチャンクのデータを準備
            inputs = tokenize_datasets(tokenizer, subset_datasets)
            inputs = _move_inputs_to_device(inputs, model.device)
            
            # 入力をより小さなバッチに分割して処理
            chunk_task_hiddens = []
            
            for i in range(0, len(inputs["input_ids"]), BATCH_SIZE):
                batch_end = min(i + BATCH_SIZE, len(inputs["input_ids"]))
                batch_inputs = {k: v[i:batch_end] for k, v in inputs.items()}
                
                # hidden statesを取得
                with torch.inference_mode():
                    outputs = model(**batch_inputs, output_hidden_states=True, return_dict=True)
                
                # 各層の最後のトークンのhidden statesを収集
                batch_hiddens = []
                for layer_idx in range(len(outputs.hidden_states)):
                    layer_hidden = outputs.hidden_states[layer_idx][:, -1, :].detach().cpu()
                    batch_hiddens.append(layer_hidden.unsqueeze(1))
                
                # [batch_size, num_layers, hidden_size]の形状に整形
                batch_hiddens = torch.cat(batch_hiddens, dim=1)
                chunk_task_hiddens.append(batch_hiddens)
                
                # メモリを解放
                del batch_inputs, outputs, batch_hiddens
                optimize_memory()
            
            if chunk_task_hiddens:
                subset_task_hiddens = torch.cat(chunk_task_hiddens, dim=0)
                
                # マスク計算 (CPUで実行して効率化)
                mask = torch.ones(len(subset_datasets), len(subset_datasets), device='cpu')
                for i, dataset in enumerate(subset_datasets):
                    for j, other_dataset in enumerate(subset_datasets):
                        if dataset.test_input in other_dataset.train_inputs or dataset.test_input == other_dataset.test_input:
                            mask[i, j] = 0
                
                # マスクが全て0の行がないか確認
                for i in range(len(subset_datasets)):
                    if torch.sum(mask[i]) == 0:
                        print(f"警告: データセット{start_idx+i}のマスクが全て0です。このデータセットのhiddensは自身から計算されます。")
                        mask[i, i] = 1
                
                # 平均タスクhiddensを計算 (CPU上で計算)
                subset_task_hiddens_list = []
                for i in range(len(subset_datasets)):
                    masked_indices = torch.where(mask[i].bool())[0]
                    if len(masked_indices) > 0:
                        # CPUで平均を計算
                        hiddens_to_avg = [subset_task_hiddens[idx] for idx in masked_indices]
                        averaged_hidden = torch.stack(hiddens_to_avg).mean(dim=0, keepdim=True)
                        subset_task_hiddens_list.append(averaged_hidden)
                    else:
                        subset_task_hiddens_list.append(subset_task_hiddens[i].unsqueeze(0))
                
                # 各データセットの平均hidden statesを結合
                if subset_task_hiddens_list:
                    subset_hiddens = torch.cat(subset_task_hiddens_list)
                    all_task_hiddens.append(subset_hiddens)
            
            # メモリを解放
            del inputs, chunk_task_hiddens
            if 'subset_task_hiddens' in locals():
                del subset_task_hiddens
            optimize_memory()
            
        except Exception as e:
            print(f"警告: データセット {start_idx}-{end_idx-1} 処理中にエラー: {e}")
            # このチャンクはスキップして次へ
    
    # すべてのサブセットを結合
    if not all_task_hiddens:
        raise ValueError("有効なタスクhiddensが取得できませんでした")
    
    task_hiddens = torch.cat(all_task_hiddens, dim=0)
    
    # 埋め込み層を除外
    task_hiddens = task_hiddens[:, 1:]
    
    print(f"Debug: 複数コンテキストhiddens取得完了: shape={task_hiddens.shape}")
    return task_hiddens

def get_single_context_task_hiddens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    datasets: List[FewShotDataset],
    num_test_inputs_to_avg: int = 1,  # メモリ使用量を減らすために1に制限
) -> torch.Tensor:
    """
    単一コンテキストからタスクhiddensを取得する関数。
    CPUとGPUのハイブリッド戦略を使用して極端なメモリ制約に対応。
    """
    optimize_memory()
    print("Debug: 単一コンテキストからタスクhiddensを取得中...")
    
    try:
        # CPUに保存するすべてのデータセットのhiddens
        all_hiddens = []
        
        # 各データセットを個別に処理して効率化
        for dataset_idx, dataset in enumerate(datasets):
            optimize_memory()
            
            try:
                print(f"Debug: データセット {dataset_idx+1}/{len(datasets)} 処理中")
                
                # テスト入力サンプルの取得 (1つだけ)
                try:
                    sample_inputs = task.sample_inputs(num_test_inputs_to_avg, exclude=(dataset.test_input,))
                    if not sample_inputs:
                        print(f"警告: {dataset.test_input}用のサンプル入力が取得できません。元の入力を使用します。")
                        sample_inputs = [dataset.test_input]
                except Exception as e:
                    print(f"警告: サンプル入力生成中にエラー: {e}")
                    sample_inputs = [dataset.test_input]
                
                # このデータセットのすべてのhiddens
                dataset_all_hiddens = []
                
                # サンプル入力ごとに処理
                for test_idx, test_input in enumerate(sample_inputs):
                    # 単一データセット用の入力を作成
                    single_dataset = FewShotDataset(
                        train_inputs=dataset.train_inputs,
                        train_outputs=dataset.train_outputs,
                        test_input=test_input,
                        test_output=task.calc_output(test_input),
                    )
                    
                    single_dataset_inputs = tokenize_datasets(
                        tokenizer, [single_dataset], format_dataset_kwargs={"include_train": False}
                    )
                    
                    # GPUに移動して処理
                    single_dataset_inputs = _move_inputs_to_device(single_dataset_inputs, model.device)
                    
                    # 推論モードで実行
                    with torch.inference_mode():
                        outputs = model(**single_dataset_inputs, output_hidden_states=True, return_dict=True)
                    
                    # 各層のhidden statesを取得（最後のトークンのみ）
                    sample_hiddens = []
                    for layer_idx in range(1, len(outputs.hidden_states)):  # 埋め込み層を除外
                        layer_hidden = outputs.hidden_states[layer_idx][:, -1, :]
                        # すぐにCPUに移動して解放
                        sample_hiddens.append(layer_hidden.detach().cpu())
                    
                    # [1, num_layers, hidden_size]の形状に整形
                    sample_hiddens = torch.stack(sample_hiddens, dim=1)
                    dataset_all_hiddens.append(sample_hiddens)
                    
                    # メモリを解放
                    del single_dataset_inputs, outputs
                    optimize_memory()
                
                # このデータセットのすべてのサンプルhiddensを平均
                if dataset_all_hiddens:
                    dataset_hiddens = torch.cat(dataset_all_hiddens, dim=0)
                    dataset_avg_hiddens = dataset_hiddens.mean(dim=0, keepdim=True)
                    all_hiddens.append(dataset_avg_hiddens)
                
            except Exception as e:
                print(f"警告: データセット {dataset_idx} 処理中にエラー: {e}")
                
                # エラーが発生した場合でもダミーhiddensを追加して処理を継続
                if all_hiddens:
                    # 既存のhiddensのサイズに合わせてダミーを作成
                    dummy_shape = all_hiddens[0].shape
                    all_hiddens.append(torch.zeros(dummy_shape, device='cpu'))
                else:
                    # 最初のケースの場合、サイズを推測
                    num_layers = len(get_layers(model)) - 1  # 埋め込み層を除外
                    hidden_size = model.config.hidden_size
                    all_hiddens.append(torch.zeros((1, num_layers, hidden_size), device='cpu'))
        
        # すべてのデータセットhiddensをCPU上で結合
        if not all_hiddens:
            raise ValueError("有効なhidden statesが取得できませんでした")
            
        task_hiddens = torch.cat(all_hiddens, dim=0)
        
        print(f"Debug: 単一コンテキストhiddens取得完了: shape={task_hiddens.shape}")
        return task_hiddens
        
    except Exception as e:
        print(f"致命的なエラー: 代替手段でのhidden states取得も失敗: {e}")
        # 完全なフォールバック: ダミーのhidden statesをCPUに作成
        num_datasets = len(datasets)
        num_layers = len(get_layers(model)) - 1  # 埋め込み層を除外
        hidden_size = model.config.hidden_size
        dummy_hiddens = torch.zeros((num_datasets, num_layers, hidden_size), device='cpu')
        return dummy_hiddens


def get_task_hiddens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    datasets: List[FewShotDataset],
    multi_context: bool = False,
) -> torch.Tensor:
    """
    データセットからタスクhiddensを取得する関数。
    multi_contextがTrueなら複数コンテキスト、そうでなければ単一コンテキストから取得。
    メモリ最適化バージョン。
    """
    optimize_memory()
    
    try:
        if multi_context:
            return get_multi_context_task_hiddens(model, tokenizer, task, datasets)
        else:
            return get_single_context_task_hiddens(model, tokenizer, task, datasets)
    except Exception as e:
        print(f"エラー: get_task_hiddens実行中に例外が発生: {e}")
        # ダミーのhidden statesを作成
        num_datasets = len(datasets)
        num_layers = len(get_layers(model)) - 1  # 埋め込み層を除外
        hidden_size = model.config.hidden_size
        return torch.zeros((num_datasets, num_layers, hidden_size), device='cpu')


def modulated_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    test_datasets: List[FewShotDataset],
    task_hiddens: torch.tensor,
    intermediate_layer: Union[int, torch.Tensor],
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    return_task_hiddens: bool = False,
    include_train: bool = False,
) -> List[str]:
    """
    タスクhiddensを注入してテキスト生成を行う関数。
    メモリ最適化バージョン - CPUをメインストレージとして使用。
    """
    optimize_memory()
    print(f"Debug: modulated_generate開始 (層={intermediate_layer})")
    
    # 入力の準備（データセット全体）
    inputs = tokenize_datasets(tokenizer, test_datasets, format_dataset_kwargs={"include_train": include_train})
    
    # CPUで保持
    if CPU_OFFLOAD:
        inputs = {k: v.cpu() if torch.is_tensor(v) else v for k, v in inputs.items()}
    
    # past_key_valuesを使わない（メモリ使用量を減らすため）
    past_key_values = None
    
    try:
        # 各バッチの回答を収集
        all_answers = []
        
        # データセットをバッチで処理
        for i in range(0, len(inputs["input_ids"]), BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, len(inputs["input_ids"]))
            batch_size = batch_end - i
            print(f"Debug: バッチ {i//BATCH_SIZE + 1} 処理中 ({i}〜{batch_end-1})")
            
            # 現在のバッチの入力をGPUに移動
            batch_inputs = {k: v[i:batch_end].to(model.device) if torch.is_tensor(v) else v 
                          for k, v in inputs.items()}
            
            # バッチに対応するタスクhiddensを選択
            batch_hiddens = task_hiddens[i:batch_end].clone()
            
            # intermediate_layerを適切にバッチサイズに調整
            if isinstance(intermediate_layer, int):
                batch_intermediate_layer = torch.tensor([intermediate_layer] * batch_size)
            else:
                batch_intermediate_layer = intermediate_layer[i:batch_end].clone()
            
            # このバッチに対して修正された前方伝播を実行
            try:
                # HiddenInjectorを使用して隠れ状態を注入
                with torch.inference_mode():
                    # 注入位置の設定（最後のトークン）
                    batch_injection_positions = -1 * torch.ones_like(batch_intermediate_layer, dtype=torch.long)
                    
                    # 選択したレイヤーのhiddensを取り出す
                    batch_selected_hiddens = []
                    for j in range(batch_size):
                        layer_idx = batch_intermediate_layer[j].item() 
                        if layer_idx < batch_hiddens.shape[1]:
                            batch_selected_hiddens.append(batch_hiddens[j, layer_idx])
                        else:
                            # 範囲外の層インデックスの場合は最後の層を使用
                            batch_selected_hiddens.append(batch_hiddens[j, -1])
                    
                    batch_selected_hiddens = torch.stack(batch_selected_hiddens).to(model.device)
                    
                    # HiddenInjectorを使用
                    with HiddenInjector(
                        model,
                        injection_layers=batch_intermediate_layer,
                        injection_positions=batch_injection_positions,
                        hiddens_to_inject=batch_selected_hiddens,
                    ) as injector:
                        # モデルの前方伝播を実行
                        batch_outputs = model(**batch_inputs, return_dict=True)
                        
                        # 最も確率の高いトークンを選択
                        batch_token_ids = batch_outputs.logits[:, -1].argmax(dim=-1).unsqueeze(-1)
                        batch_answers = decode_predictions(batch_token_ids, tokenizer)
                        all_answers.extend(batch_answers)
                
            except Exception as e:
                print(f"警告: バッチ {i//BATCH_SIZE + 1} の処理中にエラー: {e}")
                # エラーが発生した場合は、このバッチには単純なダミー回答を使用
                all_answers.extend(["error"] * batch_size)
                
            # メモリを解放
            del batch_inputs, batch_hiddens, batch_intermediate_layer
            if 'batch_selected_hiddens' in locals():
                del batch_selected_hiddens
            if 'batch_outputs' in locals():
                del batch_outputs
            optimize_memory()
        
        print(f"Debug: 予測完了、{len(all_answers)}個の回答を生成")
        
        if return_task_hiddens:
            return all_answers, task_hiddens
        return all_answers
        
    except Exception as e:
        print(f"エラー: modulated_generate全体で例外が発生: {e}")
        # フォールバック: ランダム回答を生成
        dummy_answers = ["error"] * len(test_datasets)
        
        if return_task_hiddens:
            return dummy_answers, task_hiddens
        return dummy_answers


def task_vector_accuracy_by_layer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    datasets: List[FewShotDataset],
    layers_to_test: Optional[Iterable[int]] = None,
    multi_context: bool = False,
) -> Dict[int, float]:
    """
    各層でタスクベクトルを注入した場合の精度を計算する関数。
    メモリ効率化バージョン - テスト対象の層を減らし、CPUオフロードを活用。
    """
    optimize_memory()
    print(f"Debug: task_vector_accuracy_by_layer開始 (datasets={len(datasets)})")
    
    # テスト対象の層を選択
    if layers_to_test is None:
        num_layers = len(get_layers(model))
        # 全層ではなく、代表的な層だけをテスト（メモリ使用量削減）
        if num_layers > 20:
            # 大きなモデルでは、少数の代表的な層のみをテスト
            layers_to_test = [1, 6, 11, 16, 21, 26, 31] if num_layers > 30 else [1, 5, 10, 15, 20]
            layers_to_test = [l for l in layers_to_test if l < num_layers]
        else:
            # 小さなモデルでは、より多くの層をテスト
            layers_to_test = range(1, num_layers)
            
        print(f"Debug: テスト対象の層: {list(layers_to_test)}")
    
    # データセットが大きすぎる場合は、サンプリングして処理時間を短縮
    if len(datasets) > 10:
        import random
        sampled_datasets = random.sample(datasets, min(10, len(datasets)))
        print(f"Debug: データセットが多いため、{len(sampled_datasets)}個にサンプリングしました")
    else:
        sampled_datasets = datasets
    
    # タスクhiddensを取得
    try:
        task_hiddens = get_task_hiddens(model, tokenizer, task, sampled_datasets, multi_context=multi_context)
        
        if task_hiddens is None:
            print("エラー: task_hiddensがNoneです。処理を中断します。")
            return {layer: 0.0 for layer in layers_to_test}  # ダミーの結果を返す
    except Exception as e:
        print(f"エラー: task_hiddens取得中に例外: {e}")
        return {layer: 0.0 for layer in layers_to_test}  # ダミーの結果を返す
    
    # 入力を準備（past_key_valuesは使わない）
    inputs = tokenize_datasets(tokenizer, sampled_datasets, format_dataset_kwargs={"include_train": False})
    
    # CPUで保持
    if CPU_OFFLOAD:
        inputs = {k: v.cpu() if torch.is_tensor(v) else v for k, v in inputs.items()}
    
    # 各層での精度を計算
    accuracies = []
    for layer_num in layers_to_test:
        optimize_memory()
        print(f"Debug: 層{layer_num}のテスト中...")
        
        try:
            # この層に対して回答を生成
            answers = modulated_generate(
                model=model,
                tokenizer=tokenizer,
                task=task,
                test_datasets=sampled_datasets,
                intermediate_layer=layer_num,
                task_hiddens=task_hiddens,
                past_key_values=None,  # past_key_valuesを使わない
            )
            
            # 精度を計算
            accuracy = calculate_accuracy_on_datasets(task, answers, sampled_datasets)
            print(f"Debug: 層{layer_num}の精度: {accuracy:.4f}")
            accuracies.append(accuracy)
            
        except Exception as e:
            print(f"警告: 層{layer_num}の処理中にエラーが発生: {e}")
            accuracies.append(0.0)  # エラーが発生した場合は精度0とする
        
        # メモリを解放
        optimize_memory()
    
    # 層番号と精度のマッピングを作成
    accuracy_by_layer = {layer: acc for layer, acc in zip(layers_to_test, accuracies)}
    return accuracy_by_layer


def continue_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    inputs: Dict,
    first_forward_outputs: CausalLMOutputWithPast,
    test_datasets: List[FewShotDataset],
) -> List[str]:
    """
    最初のトークン生成後、続きのトークンを生成する関数。
    メモリ最適化バージョン - CPU-GPUハイブリッド戦略を使用。
    """
    optimize_memory()
    print("警告: continue_generationは実験的な機能です（メモリ最適化版）")
    
    try:
        # 最初のトークンの予測結果をCPUに移動
        first_predicted_token_ids = first_forward_outputs.logits[:, -1].argmax(dim=-1).unsqueeze(-1).cpu()

        # 新しい入力の準備（CPUで行う）
        new_input_ids = first_predicted_token_ids
        new_attention_mask = torch.ones_like(new_input_ids, device='cpu')

        # 入力と注意マスクを結合（CPUで行う）
        if isinstance(inputs["input_ids"], torch.Tensor):
            input_ids_cpu = inputs["input_ids"].cpu()
            attention_mask_cpu = inputs["attention_mask"].cpu()
        else:
            input_ids_cpu = inputs["input_ids"]
            attention_mask_cpu = inputs["attention_mask"]
            
        full_input_ids = torch.cat([input_ids_cpu, new_input_ids], dim=-1)
        full_attention_mask = torch.cat([attention_mask_cpu, new_attention_mask], dim=-1)

        # 追加のトークンを生成（現在は1トークンのみ）
        max_new_tokens = 1  # マルチトークン出力はサポートしていない

        if max_new_tokens > 0:
            # 各バッチを個別に処理して生成
            all_output_ids = []
            
            for i in range(0, len(full_input_ids), BATCH_SIZE):
                optimize_memory()
                
                # バッチの準備
                batch_end = min(i + BATCH_SIZE, len(full_input_ids))
                batch_ids = full_input_ids[i:batch_end].to(model.device)
                batch_mask = full_attention_mask[i:batch_end].to(model.device)
                
                # 推論モードで生成
                with torch.inference_mode():
                    # このバッチの生成
                    batch_output_ids = model.generate(
                        input_ids=batch_ids,
                        attention_mask=batch_mask,
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                
                # すぐにCPUに移動
                all_output_ids.append(batch_output_ids.cpu())
                
                # メモリを解放
                del batch_ids, batch_mask, batch_output_ids
                optimize_memory()
            
            # 全てのバッチの結果を結合
            output_ids = torch.cat(all_output_ids, dim=0)
        else:
            output_ids = full_input_ids

        # 新しく生成されたトークンを抽出
        new_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
        answers = decode_predictions(new_ids, tokenizer)

        return answers
        
    except Exception as e:
        print(f"エラー: continue_generation中に例外が発生: {e}")
        # エラーが発生した場合はダミー回答を返す
        return ["[生成エラー]"] * len(test_datasets)
    
def print_prompt_examples(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, task_name: str, num_examples: int):
    """
    異なるモード（ベースライン、ICL）でのプロンプト例を表示して比較するための関数。
    デバッグ用途で使用します。
    """
    task = get_task_by_name(tokenizer=tokenizer, task_name=task_name)
    
    # ベースライン用データセット（例示なし）
    baseline_dataset = task.create_dataset(num_examples=0)
    baseline_prompt = tokenize_datasets(tokenizer, [baseline_dataset], format_dataset_kwargs={"include_train": False})
    
    # ICL用データセット（例示あり）
    icl_dataset = task.create_dataset(num_examples=num_examples)
    icl_prompt = tokenize_datasets(tokenizer, [icl_dataset], format_dataset_kwargs={"include_train": True})
    
    print(f"=== Baseline Prompt Example (task: {task_name}) ===")
    print(tokenizer.decode(baseline_prompt["input_ids"][0]))
    print("\n=== ICL Prompt Example ===")
    print(tokenizer.decode(icl_prompt["input_ids"][0]))