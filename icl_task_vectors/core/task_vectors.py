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

# バッチサイズを指定
BATCH_SIZE = 2


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
    dev_accuracy_by_layer = task_vector_accuracy_by_layer(
        model,
        tokenizer,
        task,
        dev_datasets,
        layers_to_test=layers_to_test,
        multi_context=multi_context,
    )
    best_intermediate_layer = int(max(dev_accuracy_by_layer, key=dev_accuracy_by_layer.get))

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
    if layers_to_test is None:
        num_layers = len(get_layers(model))
        layers_to_test = range(num_layers)

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
    for layer_num in layers_to_test:
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
        accuracies.append(accuracy)

    accuracy_by_layer = {layer: acc for layer, acc in zip(layers_to_test, accuracies)}
    return accuracy_by_layer


# 以下はダミー実装例。
def modulated_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    datasets: List[FewShotDataset],
    task_hiddens=None,
    intermediate_layer=None,
    past_key_values=None,
):
    # 実際には HiddenInjector などのコンテキストマネージャを使い、
    # model.generate() をカスタマイズして回答トークンを得る想定です。
    return ["dummy_answer"] * sum(len(ds) for ds in datasets)


def get_task_hiddens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    datasets: List[FewShotDataset],
    multi_context: bool = False,
):
    # 実際にはデータセットをバッチで forwardし、中間層の hidden states を抜き出す処理などを行う想定
    return None
