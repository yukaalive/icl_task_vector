# This must be first
from dotenv import load_dotenv

load_dotenv(".env")

import sys
import os
import pickle
import time
from typing import Optional

from transformers import PreTrainedModel, PreTrainedTokenizer

from scripts.utils import MAIN_RESULTS_DIR, main_experiment_results_dir

from core.data.task_helpers import get_all_tasks, get_task_by_name
from core.models.llm_loading import load_model_and_tokenizer
from core.models.utils.inference import hidden_to_logits
from core.analysis.utils import logits_top_tokens
from core.analysis.evaluation import calculate_accuracy_on_datasets
from core.task_vectors import run_icl, run_task_vector
from core.utils.misc import limit_gpus, seed_everything
from core.experiments_config import MODELS_TO_EVALUATE, TASKS_TO_EVALUATE
from core.data.datasets.few_shot_format import FewShotFormat
from core.data.datasets.format_factory import create_few_shot_format_for_task, extract_dataset_name_from_task


def run_icl_with_format(model, tokenizer, task, datasets, include_train=True, few_shot_format=None, dataset_name=None):
    """指定されたフォーマットでICL実験を実行"""
    from core.models.utils.inference import tokenize_datasets, batch_generate, decode_predictions
    
    if few_shot_format is None:
        few_shot_format = FewShotFormat()
    
    format_dataset_kwargs = {"include_train": include_train}
    inputs = tokenize_datasets(
        tokenizer, 
        datasets, 
        few_shot_format=few_shot_format,
        format_dataset_kwargs=format_dataset_kwargs,
        dataset_name=dataset_name
    )
    new_ids = batch_generate(model, tokenizer, inputs=inputs, generate_kwargs={"max_new_tokens": 1})
    predictions = decode_predictions(new_ids, tokenizer, few_shot_format)
    return predictions


def run_task_vector_with_format(model, tokenizer, task, test_datasets, dev_datasets, few_shot_format=None, dataset_name=None):
    """指定されたフォーマットでTask Vector実験を実行"""
    from core.models.utils.inference import tokenize_datasets, batch_forward, traced_forward
    from core.models.utils.llm_layers import get_layers
    from core.utils.nested import nested_apply
    from core.task_vectors import modulated_generate
    from core.data.datasets.few_shot_dataset import FewShotDataset
    
    if few_shot_format is None:
        few_shot_format = FewShotFormat()
    
    # 開発セットでの精度を各層で計算
    layers_to_test = range(len(get_layers(model)))
    
    # タスク隠れ層を取得
    num_test_inputs_to_avg = 2
    new_datasets = [
        FewShotDataset(
            train_inputs=dataset.train_inputs,
            train_outputs=dataset.train_outputs,
            test_input=test_input,
            test_output=task.calc_output(test_input),
        )
        for dataset in test_datasets
        for test_input in task.sample_inputs(num_test_inputs_to_avg, exclude=(dataset.test_input,))
    ]

    inputs = tokenize_datasets(
        tokenizer, 
        new_datasets, 
        few_shot_format=few_shot_format,
        dataset_name=dataset_name
    )
    outputs, forward_trace = traced_forward(model, inputs=inputs)
    task_hiddens = forward_trace.residual_stream.hidden[:, :, -1, :]
    _, num_layers, hidden_size = task_hiddens.shape
    task_hiddens = task_hiddens.view(len(test_datasets), num_test_inputs_to_avg, num_layers, hidden_size).mean(dim=1)
    task_hiddens = task_hiddens[:, 1:]  # 埋め込み層を除く
    
    device = model.device
    task_hiddens = task_hiddens.to(device)
    
    # 入力のpast_key_valuesを取得
    inputs = tokenize_datasets(
        tokenizer, 
        dev_datasets, 
        few_shot_format=few_shot_format,
        format_dataset_kwargs={"include_train": False},
        dataset_name=dataset_name
    )
    outputs = batch_forward(model, inputs=inputs, forward_kwargs={"use_cache": True})
    past_key_values = outputs.past_key_values
    past_key_values = nested_apply(past_key_values, lambda x: x[:, :, :-1])
    inputs["input_ids"] = inputs["input_ids"][:, -1].unsqueeze(1)

    # 各層での精度を計算
    accuracies = []
    for layer_num in layers_to_test:
        answers = modulated_generate(
            model,
            tokenizer,
            task,
            dev_datasets,
            intermediate_layer=layer_num,
            task_hiddens=task_hiddens,
            past_key_values=past_key_values,
        )
        accuracy = calculate_accuracy_on_datasets(task, answers, dev_datasets)
        accuracies.append(accuracy)
    
    tv_dev_accuracy_by_layer = {layer: accuracy for layer, accuracy in zip(layers_to_test, accuracies)}
    best_intermediate_layer = int(max(tv_dev_accuracy_by_layer, key=tv_dev_accuracy_by_layer.get))

    # テストセットでの予測を生成
    predictions = modulated_generate(
        model,
        tokenizer,
        task,
        test_datasets,
        task_hiddens=task_hiddens,
        intermediate_layer=best_intermediate_layer,
    )
    
    # テストセットでの詳細結果を表示
    print(f"\n=== Task Vector テストセット結果 (best layer: {best_intermediate_layer}) ===")
    test_accuracy = calculate_accuracy_on_datasets(task, predictions, test_datasets)
    print(f"テストセット精度: {test_accuracy:.3f}")
    
    # 最初の10個のテストセット結果を表示
    print("テストセット予測結果（最初の10個）:")
    for i, (dataset, prediction) in enumerate(zip(test_datasets[:10], predictions[:10])):
        correct_answer = dataset.test_output
        is_correct = "〇" if prediction.strip().lower() == correct_answer.strip().lower() else "✗"
        print(f"  {i+1}. 入力: {dataset.test_input}")
        print(f"     予測: '{prediction}' | 正解: '{correct_answer}' {is_correct}")
    
    if len(test_datasets) > 10:
        print(f"  ... (残り{len(test_datasets)-10}個のテストセット)")
    print()

    return predictions, tv_dev_accuracy_by_layer, task_hiddens


def get_results_file_path(model_type: str, model_variant: str, experiment_id: str = "", use_dataset_specific: bool = False) -> str:
    """結果ファイルのパスを取得"""
    suffix = "_dataset_specific" if use_dataset_specific else ""
    return os.path.join(main_experiment_results_dir(experiment_id), f"{model_type}_{model_variant}{suffix}.pkl")


def evaluate_task(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, task_name: str, num_examples: int, use_dataset_specific: bool = False) -> None:
    """
    タスクを評価する
    Args:
        use_dataset_specific: データセット固有プロンプトを使用するかどうか
    """
    seed_everything(41)
    accuracies = {}

    task = get_task_by_name(tokenizer=tokenizer, task_name=task_name)
    
    # データセット固有プロンプトの設定
    few_shot_format = None
    dataset_name = None
    if use_dataset_specific:
        dataset_name = extract_dataset_name_from_task(task_name)
        if dataset_name:
            few_shot_format = create_few_shot_format_for_task(task_name)
            print(f"拡張プロンプト: {dataset_name}")
        else:
            print("データセット固有プロンプトが見つかりません。デフォルトプロンプトを使用します。")
    else:
        print("ノーマルプロンプト")

    # Evaluate baseline
    baseline_datasets = task.create_datasets(num_datasets=100, num_examples=0)
    predictions = run_icl_with_format(model, tokenizer, task, baseline_datasets, include_train=False, few_shot_format=few_shot_format, dataset_name=dataset_name)
    accuracies["baseline"] = calculate_accuracy_on_datasets(task, predictions, baseline_datasets)

    # Evaluate ICL and Task Vector
    # TODO: Change back to 400, 100
    # num_test_datasets, num_dev_datasets = 400, 100
    num_test_datasets, num_dev_datasets = 50, 50
    test_datasets = task.create_datasets(num_datasets=num_test_datasets, num_examples=num_examples)
    dev_datasets = task.create_datasets(num_datasets=num_dev_datasets, num_examples=num_examples)
    
    # 実験の初回だけプロンプトの例を表示
    if len(test_datasets) > 0:
        if few_shot_format is None:
            display_format = FewShotFormat()
        else:
            display_format = few_shot_format
        
        sample_prompt = display_format.format_dataset(test_datasets[0], dataset_name=dataset_name)
        print(f"\n--- 使用されているプロンプトの例 ---")
        print(f"プロンプトタイプ: {'拡張プロンプト' if use_dataset_specific else 'ノーマルプロンプト'}")
        print(f"データセット名: {dataset_name if dataset_name else 'なし'}")
        print(f"プロンプト内容:")
        print(sample_prompt)
        print(f"--- プロンプト例終了 ---\n")
    icl_predictions = run_icl_with_format(model, tokenizer, task, test_datasets, few_shot_format=few_shot_format, dataset_name=dataset_name)
    tv_predictions, tv_dev_accuracy_by_layer, task_hiddens = run_task_vector_with_format(
        model,
        tokenizer,
        task,
        test_datasets,
        dev_datasets,
        few_shot_format=few_shot_format,
        dataset_name=dataset_name,
    )
    accuracies["tv_dev_by_layer"] = tv_dev_accuracy_by_layer
    accuracies["icl"] = calculate_accuracy_on_datasets(task, icl_predictions, test_datasets)
    accuracies["tv"] = calculate_accuracy_on_datasets(task, tv_predictions, test_datasets)

    tv_ordered_tokens_by_layer = {}
    try:
        for layer_num in tv_dev_accuracy_by_layer.keys():
            task_hidden = task_hiddens.mean(axis=0)[layer_num]
            logits = hidden_to_logits(model, task_hidden)
            tv_ordered_tokens_by_layer[layer_num] = logits_top_tokens(logits, tokenizer, k=100)
    except Exception as e:
        print("Error:", e)

    return accuracies, tv_ordered_tokens_by_layer


def run_main_experiment(
    model_type: str,
    model_variant: str,
    experiment_id: str = "",
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    use_dataset_specific: bool = False,
) -> None:
    prompt_type = "拡張プロンプト" if use_dataset_specific else "ノーマルプロンプト"
    print(f"Evaluating model with {prompt_type}: {model_type} {model_variant}")

    results_file = get_results_file_path(model_type, model_variant, experiment_id=experiment_id, use_dataset_specific=use_dataset_specific)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    if os.path.exists(results_file):
        with open(results_file, "rb") as f:
            results = pickle.load(f)
    else:
        results = {}

    limit_gpus(range(0, 8))

    print("Loading model and tokenizer...")
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(model_type, model_variant)
    print("Loaded model and tokenizer.")

    tasks = get_all_tasks(tokenizer=tokenizer)

    num_examples = 5

    for i, task_name in enumerate(TASKS_TO_EVALUATE):
        task = tasks[task_name]
        if task_name in results:
            print(f"Skipping task {i+1}/{len(TASKS_TO_EVALUATE)}: {task_name}")
            continue
        results[task_name] = {}

        print("\n" + "=" * 50)
        print(f"Running task {i+1}/{len(TASKS_TO_EVALUATE)}: {task_name}")

        tic = time.time()
        accuracies, tv_ordered_tokens_by_layer = evaluate_task(model, tokenizer, task_name, num_examples, use_dataset_specific)

        print(f"Baseline Accuracy: {accuracies['baseline']:.2f}")
        print(f"ICL Accuracy: {accuracies['icl']:.2f}")
        print(f"Task Vector Accuracy: {accuracies['tv']:.2f}")
        print(f"Dev Accuracy by layer: ", end="")
        for layer, accuracy in accuracies["tv_dev_by_layer"].items():
            print(f"{layer}: {accuracy:.2f}, ", end="")
        print()
        print("Time:", time.time() - tic)

        results[task_name] = {
            "baseline_accuracy": accuracies["baseline"],
            "num_examples": num_examples,
            "icl_accuracy": accuracies["icl"],
            "tv_accuracy": accuracies["tv"],
            "tv_dev_accruacy_by_layer": accuracies["tv_dev_by_layer"],
            "tv_ordered_tokens_by_layer": tv_ordered_tokens_by_layer,
            "use_dataset_specific": use_dataset_specific,
        }

        with open(results_file, "wb") as f:
            pickle.dump(results, f)


def get_new_experiment_id() -> str:
    return str(
        max([int(results_dir) for results_dir in os.listdir(MAIN_RESULTS_DIR) if results_dir.isdigit()] + [0]) + 1
    )


def main():
    if len(sys.argv) == 1:
        # Run all models
        # Calculate the experiment_id as the max experiment_id + 1
        experiment_id = get_new_experiment_id()
        for model_type, model_variant in MODELS_TO_EVALUATE:
            # 1回目: ノーマルプロンプトで実験
            print(f"\n{'='*60}")
            print(f"【1回目】ノーマルプロンプト: {model_type} {model_variant}")
            print(f"{'='*60}")
            run_main_experiment(model_type, model_variant, experiment_id=experiment_id, use_dataset_specific=False)
            
            # 2回目: 拡張プロンプトで実験
            print(f"\n{'='*60}")
            print(f"【2回目】拡張プロンプト: {model_type} {model_variant}")
            print(f"{'='*60}")
            run_main_experiment(model_type, model_variant, experiment_id=experiment_id, use_dataset_specific=True)
    else:
        if len(sys.argv) == 2:
            model_num = int(sys.argv[1])
            model_type, model_variant = MODELS_TO_EVALUATE[model_num]
        elif len(sys.argv) == 3:
            model_type, model_variant = sys.argv[1:]

        # 1回目: ノーマルプロンプトで実験
        print(f"\n{'='*60}")
        print(f"【1回目】ノーマルプロンプト: {model_type} {model_variant}")
        print(f"{'='*60}")
        run_main_experiment(model_type, model_variant, use_dataset_specific=False)
        
        # 2回目: 拡張プロンプトで実験
        print(f"\n{'='*60}")
        print(f"【2回目】拡張プロンプト: {model_type} {model_variant}")
        print(f"{'='*60}")
        run_main_experiment(model_type, model_variant, use_dataset_specific=True)


if __name__ == "__main__":
    main()
