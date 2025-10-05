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


def get_results_file_path(model_type: str, model_variant: str, experiment_id: str = "") -> str:
    return os.path.join(main_experiment_results_dir(experiment_id), f"{model_type}_{model_variant}.pkl")


def evaluate_task(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, task_name: str, num_examples: int) -> None:
    seed_everything(41)
    accuracies = {}
    comet_results = {}

    task = get_task_by_name(tokenizer=tokenizer, task_name=task_name)

    # Evaluate baseline
    baseline_datasets = task.create_datasets(num_datasets=100, num_examples=0)
    predictions = run_icl(model, tokenizer, task, baseline_datasets, include_train=False)
    accuracies["baseline"] = calculate_accuracy_on_datasets(task, predictions, baseline_datasets)

    # Evaluate ICL and Task Vector
    # TODO: Change back to 400, 100
    # num_test_datasets, num_dev_datasets = 400, 100
    num_test_datasets, num_dev_datasets = 50, 50
    test_datasets = task.create_datasets(num_datasets=num_test_datasets, num_examples=num_examples)
    dev_datasets = task.create_datasets(num_datasets=num_dev_datasets, num_examples=num_examples)
    
    icl_predictions = run_icl(model, tokenizer, task, test_datasets)
    tv_predictions, tv_dev_accuracy_by_layer, task_hiddens = run_task_vector(
        model,
        tokenizer,
        task,
        test_datasets,
        dev_datasets,
        max_new_tokens=30,  # Task Vectorで複数トークン生成を有効化
    )
    
    accuracies["tv_dev_by_layer"] = tv_dev_accuracy_by_layer
    accuracies["icl"] = calculate_accuracy_on_datasets(task, icl_predictions, test_datasets)
    accuracies["tv"] = calculate_accuracy_on_datasets(task, tv_predictions, test_datasets)
    
    # Add COMET evaluation for translation tasks
    if task_name.startswith("translation_") and hasattr(task, 'evaluate_with_comet'):
        print("\n--- COMET Evaluation ---")
        
        # DEBUG: Check actual Task Vector predictions
        print("\n=== TASK VECTOR PREDICTIONS DEBUG ===")
        print(f"TV predictions type: {type(tv_predictions)}, length: {len(tv_predictions)}")
        for i in range(min(5, len(tv_predictions))):
            print(f"TV[{i}]: '{tv_predictions[i]}' (type: {type(tv_predictions[i])}, len: {len(tv_predictions[i])})")
        
        try:
            # Prepare data for COMET evaluation
            sources = [dataset.test_input for dataset in test_datasets]
            references = [dataset.test_output for dataset in test_datasets]
            
            print(f"\nCOMET Input Debug:")
            print(f"Sources length: {len(sources)}")
            print(f"ICL predictions length: {len(icl_predictions)}")
            print(f"TV predictions length: {len(tv_predictions)}")
            print(f"References length: {len(references)}")
            
            # COMET evaluation for ICL predictions
            icl_comet_results = task.evaluate_with_comet(sources, icl_predictions, references)
            comet_results["icl_comet"] = icl_comet_results["comet"]
            print(f"ICL COMET Score: {icl_comet_results['comet']:.4f}")
            
            # COMET evaluation for Task Vector predictions - check for issues
            print("\nEvaluating Task Vector with COMET...")
            tv_comet_results = task.evaluate_with_comet(sources, tv_predictions, references)
            comet_results["tv_comet"] = tv_comet_results["comet"]
            print(f"Task Vector COMET Score: {tv_comet_results['comet']:.4f}")
            
            # Compare individual COMET scores
            if "comet_scores" in icl_comet_results and "comet_scores" in tv_comet_results:
                print(f"\nFirst 3 individual COMET scores:")
                for i in range(min(3, len(icl_comet_results["comet_scores"]))):
                    icl_score = icl_comet_results["comet_scores"][i]
                    tv_score = tv_comet_results["comet_scores"][i]
                    print(f"  Example {i+1}: ICL={icl_score:.4f}, TV={tv_score:.4f}")
            
        except Exception as e:
            print(f"COMET evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Display example predictions for ICL and Task Vector (corrected from baseline predictions)
    for i in range(min(10, len(test_datasets))):
        dataset = test_datasets[i]
        icl_pred = icl_predictions[i]
        tv_pred = tv_predictions[i]
        correct_answer = dataset.test_output
        
        icl_correct = "〇" if icl_pred.strip().lower() == correct_answer.strip().lower() else "✗"
        tv_correct = "〇" if tv_pred.strip().lower() == correct_answer.strip().lower() else "✗"
        
        print(f"  {i+1}. 入力: {dataset.test_input}")
        print(f"     ICL予測: '{icl_pred}' | 正解: '{correct_answer}' {icl_correct}")
        print(f"     TV予測:  '{tv_pred}' | 正解: '{correct_answer}' {tv_correct}")

    tv_ordered_tokens_by_layer = {}
    try:
        for layer_num in tv_dev_accuracy_by_layer.keys():
            task_hidden = task_hiddens.mean(axis=0)[layer_num]
            logits = hidden_to_logits(model, task_hidden)
            tv_ordered_tokens_by_layer[layer_num] = logits_top_tokens(logits, tokenizer, k=100)
    except Exception as e:
        print("Error:", e)

    return accuracies, comet_results, tv_ordered_tokens_by_layer


def run_main_experiment(
    model_type: str,
    model_variant: str,
    experiment_id: str = "",
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
) -> None:
    print("Evaluating model:", model_type, model_variant)

    results_file = get_results_file_path(model_type, model_variant, experiment_id=experiment_id)
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
            print(f"Skipping task {i+1}/{len(tasks)}: {task_name}")
            continue
        results[task_name] = {}

        print("\n" + "=" * 50)
        print(f"Running task {i+1}/{len(tasks)}: {task_name}")

        tic = time.time()
        accuracies, comet_results, tv_ordered_tokens_by_layer = evaluate_task(model, tokenizer, task_name, num_examples)

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
            **comet_results,  # Include COMET results
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
            run_main_experiment(model_type, model_variant, experiment_id=experiment_id)
    else:
        if len(sys.argv) == 2:
            model_num = int(sys.argv[1])
            model_type, model_variant = MODELS_TO_EVALUATE[model_num]
        elif len(sys.argv) == 3:
            model_type, model_variant = sys.argv[1:]

        run_main_experiment(model_type, model_variant)


if __name__ == "__main__":
    main()
