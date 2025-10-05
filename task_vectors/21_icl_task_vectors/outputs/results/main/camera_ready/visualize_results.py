import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

def load_results(results_dir):
    """Load all pickle files from the results directory"""
    results = {}

    for pkl_file in Path(results_dir).glob("*.pkl"):
        model_name = pkl_file.stem  # e.g., "swallow_7B"
        with open(pkl_file, 'rb') as f:
            results[model_name] = pickle.load(f)

    return results

def extract_metrics_to_dataframe(results):
    """Extract metrics into a pandas DataFrame"""
    rows = []

    for model_name, tasks_data in results.items():
        for task_name, metrics in tasks_data.items():
            row = {
                'model': model_name,
                'task': task_name,
                'baseline_accuracy': metrics.get('baseline_accuracy', np.nan),
                'icl_accuracy': metrics.get('icl_accuracy', np.nan),
                'tv_accuracy': metrics.get('tv_accuracy', np.nan),
                'icl_comet': metrics.get('icl_comet', np.nan),
                'tv_comet': metrics.get('tv_comet', np.nan),
                'num_examples': metrics.get('num_examples', np.nan),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    return df

def plot_accuracy_comparison(df, output_dir):
    """Plot accuracy comparison: ICL vs Task Vector"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Prepare data
    models = df['model'].unique()
    tasks = df['task'].unique()

    # Plot 1: Grouped bar chart by task
    ax1 = axes[0]
    x = np.arange(len(tasks))
    width = 0.15

    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        icl_acc = [model_data[model_data['task'] == task]['icl_accuracy'].values[0]
                   if len(model_data[model_data['task'] == task]) > 0 else 0
                   for task in tasks]
        tv_acc = [model_data[model_data['task'] == task]['tv_accuracy'].values[0]
                  if len(model_data[model_data['task'] == task]) > 0 else 0
                  for task in tasks]

        ax1.bar(x + i * width * 2 - width/2, icl_acc, width, label=f'{model} (ICL)', alpha=0.8)
        ax1.bar(x + i * width * 2 + width/2, tv_acc, width, label=f'{model} (TV)', alpha=0.8, hatch='//')

    ax1.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Exact Match Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Exact Match Accuracy: ICL vs Task Vector by Task', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width * (len(models) - 1))
    ax1.set_xticklabels(tasks, rotation=45, ha='right')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Grouped bar chart by model
    ax2 = axes[1]
    x = np.arange(len(models))
    width = 0.08

    for i, task in enumerate(tasks):
        task_data = df[df['task'] == task]
        icl_acc = [task_data[task_data['model'] == model]['icl_accuracy'].values[0]
                   if len(task_data[task_data['model'] == model]) > 0 else 0
                   for model in models]
        tv_acc = [task_data[task_data['model'] == model]['tv_accuracy'].values[0]
                  if len(task_data[task_data['model'] == model]) > 0 else 0
                  for model in models]

        ax2.bar(x + i * width * 2 - width/2, icl_acc, width, label=f'{task} (ICL)', alpha=0.8)
        ax2.bar(x + i * width * 2 + width/2, tv_acc, width, label=f'{task} (TV)', alpha=0.8, hatch='//')

    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Exact Match Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Exact Match Accuracy: ICL vs Task Vector by Model', fontsize=14, fontweight='bold')
    ax2.set_xticks(x + width * (len(tasks) - 1))
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'accuracy_comparison.png')}")
    plt.close()

def plot_comet_comparison(df, output_dir):
    """Plot COMET score comparison: ICL vs Task Vector"""
    # Filter only translation tasks with COMET scores
    df_comet = df[df['icl_comet'].notna() & df['tv_comet'].notna()].copy()

    if len(df_comet) == 0:
        print("No COMET scores available to plot.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    models = df_comet['model'].unique()
    tasks = df_comet['task'].unique()

    # Plot 1: Grouped bar chart by task
    ax1 = axes[0]
    x = np.arange(len(tasks))
    width = 0.15

    for i, model in enumerate(models):
        model_data = df_comet[df_comet['model'] == model]
        icl_comet = [model_data[model_data['task'] == task]['icl_comet'].values[0]
                     if len(model_data[model_data['task'] == task]) > 0 else 0
                     for task in tasks]
        tv_comet = [model_data[model_data['task'] == task]['tv_comet'].values[0]
                    if len(model_data[model_data['task'] == task]) > 0 else 0
                    for task in tasks]

        ax1.bar(x + i * width * 2 - width/2, icl_comet, width, label=f'{model} (ICL)', alpha=0.8)
        ax1.bar(x + i * width * 2 + width/2, tv_comet, width, label=f'{model} (TV)', alpha=0.8, hatch='//')

    ax1.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax1.set_ylabel('COMET Score', fontsize=12, fontweight='bold')
    ax1.set_title('COMET Score: ICL vs Task Vector by Task', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width * (len(models) - 1))
    ax1.set_xticklabels(tasks, rotation=45, ha='right')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1])

    # Plot 2: Grouped bar chart by model
    ax2 = axes[1]
    x = np.arange(len(models))
    width = 0.08

    for i, task in enumerate(tasks):
        task_data = df_comet[df_comet['task'] == task]
        icl_comet = [task_data[task_data['model'] == model]['icl_comet'].values[0]
                     if len(task_data[task_data['model'] == model]) > 0 else 0
                     for model in models]
        tv_comet = [task_data[task_data['model'] == model]['tv_comet'].values[0]
                    if len(task_data[task_data['model'] == model]) > 0 else 0
                    for model in models]

        ax2.bar(x + i * width * 2 - width/2, icl_comet, width, label=f'{task} (ICL)', alpha=0.8)
        ax2.bar(x + i * width * 2 + width/2, tv_comet, width, label=f'{task} (TV)', alpha=0.8, hatch='//')

    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('COMET Score', fontsize=12, fontweight='bold')
    ax2.set_title('COMET Score: ICL vs Task Vector by Model', fontsize=14, fontweight='bold')
    ax2.set_xticks(x + width * (len(tasks) - 1))
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comet_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'comet_comparison.png')}")
    plt.close()

def plot_heatmaps(df, output_dir):
    """Plot heatmaps for accuracy and COMET scores"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Prepare pivot tables
    icl_acc_pivot = df.pivot(index='task', columns='model', values='icl_accuracy')
    tv_acc_pivot = df.pivot(index='task', columns='model', values='tv_accuracy')
    icl_comet_pivot = df.pivot(index='task', columns='model', values='icl_comet')
    tv_comet_pivot = df.pivot(index='task', columns='model', values='tv_comet')

    # Plot heatmaps
    sns.heatmap(icl_acc_pivot, annot=True, fmt='.3f', cmap='YlGnBu', ax=axes[0, 0],
                cbar_kws={'label': 'Accuracy'}, vmin=0, vmax=1)
    axes[0, 0].set_title('ICL: Exact Match Accuracy', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Model', fontsize=10, fontweight='bold')
    axes[0, 0].set_ylabel('Task', fontsize=10, fontweight='bold')

    sns.heatmap(tv_acc_pivot, annot=True, fmt='.3f', cmap='YlGnBu', ax=axes[0, 1],
                cbar_kws={'label': 'Accuracy'}, vmin=0, vmax=1)
    axes[0, 1].set_title('Task Vector: Exact Match Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Model', fontsize=10, fontweight='bold')
    axes[0, 1].set_ylabel('Task', fontsize=10, fontweight='bold')

    sns.heatmap(icl_comet_pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1, 0],
                cbar_kws={'label': 'COMET Score'}, vmin=0, vmax=1)
    axes[1, 0].set_title('ICL: COMET Score', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Model', fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel('Task', fontsize=10, fontweight='bold')

    sns.heatmap(tv_comet_pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1, 1],
                cbar_kws={'label': 'COMET Score'}, vmin=0, vmax=1)
    axes[1, 1].set_title('Task Vector: COMET Score', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Model', fontsize=10, fontweight='bold')
    axes[1, 1].set_ylabel('Task', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmaps.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'heatmaps.png')}")
    plt.close()

def save_summary_table(df, output_dir):
    """Save summary statistics as CSV and LaTeX"""
    # Summary by task
    summary_by_task = df.groupby('task').agg({
        'icl_accuracy': ['mean', 'std'],
        'tv_accuracy': ['mean', 'std'],
        'icl_comet': ['mean', 'std'],
        'tv_comet': ['mean', 'std']
    }).round(4)

    # Summary by model
    summary_by_model = df.groupby('model').agg({
        'icl_accuracy': ['mean', 'std'],
        'tv_accuracy': ['mean', 'std'],
        'icl_comet': ['mean', 'std'],
        'tv_comet': ['mean', 'std']
    }).round(4)

    # Save to CSV
    summary_by_task.to_csv(os.path.join(output_dir, 'summary_by_task.csv'))
    summary_by_model.to_csv(os.path.join(output_dir, 'summary_by_model.csv'))
    df.to_csv(os.path.join(output_dir, 'all_results.csv'), index=False)

    print(f"Saved: {os.path.join(output_dir, 'summary_by_task.csv')}")
    print(f"Saved: {os.path.join(output_dir, 'summary_by_model.csv')}")
    print(f"Saved: {os.path.join(output_dir, 'all_results.csv')}")

    # Save to LaTeX
    with open(os.path.join(output_dir, 'summary_by_task.tex'), 'w') as f:
        f.write(summary_by_task.to_latex())
    with open(os.path.join(output_dir, 'summary_by_model.tex'), 'w') as f:
        f.write(summary_by_model.to_latex())

    print(f"Saved: {os.path.join(output_dir, 'summary_by_task.tex')}")
    print(f"Saved: {os.path.join(output_dir, 'summary_by_model.tex')}")

def main():
    # Configuration
    results_dir = "/home/yukaalive/2025workspace/task_vectors/21_icl_task_vectors/outputs/results/main/camera_ready"
    output_dir = "/home/yukaalive/2025workspace/task_vectors/21_icl_task_vectors/outputs/results/main/camera_ready"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print("Loading results...")
    results = load_results(results_dir)
    print(f"Loaded {len(results)} models")

    print("\nExtracting metrics to DataFrame...")
    df = extract_metrics_to_dataframe(results)
    print(f"DataFrame shape: {df.shape}")
    print("\nDataFrame preview:")
    print(df.head(10))

    print("\n" + "="*50)
    print("Generating visualizations...")
    print("="*50)

    # Generate plots
    plot_accuracy_comparison(df, output_dir)
    plot_comet_comparison(df, output_dir)
    plot_heatmaps(df, output_dir)

    # Save summary tables
    print("\nSaving summary tables...")
    save_summary_table(df, output_dir)

    print("\n" + "="*50)
    print("Visualization complete!")
    print(f"All outputs saved to: {output_dir}")
    print("="*50)

if __name__ == "__main__":
    main()
