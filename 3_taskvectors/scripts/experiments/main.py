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
from core.models.utils.inference import hidden_to_logits, tokenize_datasets
from core.analysis.utils import logits_top_tokens
from core.analysis.evaluation import calculate_accuracy_on_datasets
from core.task_vectors import run_icl, run_task_vector
from core.utils.misc import limit_gpus, seed_everything
from core.experiments_config import MODELS_TO_EVALUATE, TASKS_TO_EVALUATE


def get_results_file_path(model_type: str, model_variant: str, experiment_id: str = "") -> str:
    return os.path.join(main_experiment_results_dir(experiment_id), f"{model_type}_{model_variant}.pkl")


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


def evaluate_task(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, task_name: str, num_examples: int) -> None:
    seed_everything(41)
    accuracies = {}

    task = get_task_by_name(tokenizer=tokenizer, task_name=task_name)

    # Evaluate baseline - 論文の実装と同じように設定
    print(f"Evaluating baseline for task: {task_name}")
    baseline_datasets = task.create_datasets(num_datasets=100, num_examples=0)
    
    # ベースラインはタスク説明とテスト入力の形式を含むが、例示は含まない
    baseline_predictions = run_icl(model, tokenizer, task, baseline_datasets, include_train=False)
    accuracies["baseline"] = calculate_accuracy_on_datasets(task, baseline_predictions, baseline_datasets)
    print(f"Baseline accuracy: {accuracies['baseline']:.4f}")

    # Evaluate ICL and Task Vector
    print(f"Evaluating ICL and Task Vector for task: {task_name}")
    # num_test_datasets, num_dev_datasets = 400, 100
    num_test_datasets, num_dev_datasets = 25, 25
    test_datasets = task.create_datasets(num_datasets=num_test_datasets, num_examples=num_examples)
    dev_datasets = task.create_datasets(num_datasets=num_dev_datasets, num_examples=num_examples)
    
    # ICL評価 - タスク説明、例示、テスト入力の形式を含む
    icl_predictions = run_icl(model, tokenizer, task, test_datasets, include_train=True)
    accuracies["icl"] = calculate_accuracy_on_datasets(task, icl_predictions, test_datasets)
    print(f"ICL accuracy: {accuracies['icl']:.4f}")
    
    # タスクベクトル評価
    tv_predictions, tv_dev_accuracy_by_layer, task_hiddens = run_task_vector(
        model,
        tokenizer,
        task,
        test_datasets,
        dev_datasets,
    )
    accuracies["tv_dev_by_layer"] = tv_dev_accuracy_by_layer
    accuracies["tv"] = calculate_accuracy_on_datasets(task, tv_predictions, test_datasets)
    print(f"Task Vector accuracy: {accuracies['tv']:.4f}")

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
    debug_prompts: bool = False,
) -> None:
    print("Evaluating model:", model_type, model_variant)

    results_file = get_results_file_path(model_type, model_variant, experiment_id=experiment_id)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    if os.path.exists(results_file):
        with open(results_file, "rb") as f:
            results = pickle.load(f)
    else:
        results = {}

    #limit_gpus(range(0, 8))
    limit_gpus([0])

    print("Loading model and tokenizer...")
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(model_type, model_variant)
    print("Loaded model and tokenizer.")

    tasks = get_all_tasks(tokenizer=tokenizer)

    num_examples = 5

    # デバッグモードの場合、プロンプト例を表示
    if debug_prompts:
        for task_name in TASKS_TO_EVALUATE:
            print_prompt_examples(model, tokenizer, task_name, num_examples)
            print("\n" + "=" * 80 + "\n")

    for i, task_name in enumerate(TASKS_TO_EVALUATE):
        task = tasks[task_name]
        if task_name in results:
            print(f"Skipping task {i+1}/{len(tasks)}: {task_name}")
            continue
        results[task_name] = {}

        print("\n" + "=" * 50)
        print(f"Running task {i+1}/{len(tasks)}: {task_name}")

        tic = time.time()
        accuracies, tv_ordered_tokens_by_layer = evaluate_task(model, tokenizer, task_name, num_examples)

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
        }

        with open(results_file, "wb") as f:
            pickle.dump(results, f)


def get_new_experiment_id() -> str:
    return str(
        max([int(results_dir) for results_dir in os.listdir(MAIN_RESULTS_DIR) if results_dir.isdigit()] + [0]) + 1
    )


def main():
    debug_mode = "--debug" in sys.argv
    if debug_mode:
        sys.argv.remove("--debug")
        
    if len(sys.argv) == 1:
        # Run all models
        # Calculate the experiment_id as the max experiment_id + 1
        experiment_id = get_new_experiment_id()
        
        # デバッグモードが有効な場合、最初のモデルで全タスクのプロンプト例を表示
        if debug_mode:
            model_type, model_variant = MODELS_TO_EVALUATE[0]
            model, tokenizer = load_model_and_tokenizer(model_type, model_variant)
            run_main_experiment(model_type, model_variant, experiment_id=experiment_id, 
                              model=model, tokenizer=tokenizer, debug_prompts=True)
        
        for model_type, model_variant in MODELS_TO_EVALUATE:
            run_main_experiment(model_type, model_variant, experiment_id=experiment_id, debug_prompts=False)
    else:
        if len(sys.argv) == 2:
            model_num = int(sys.argv[1])
            model_type, model_variant = MODELS_TO_EVALUATE[model_num]
        elif len(sys.argv) >= 3:
            model_type, model_variant = sys.argv[1:3]

        run_main_experiment(model_type, model_variant, debug_prompts=debug_mode)


if __name__ == "__main__":
    main()