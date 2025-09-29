#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
camera_readyデータから統合グラフを作成するスクリプト

統合されたグラフ:
1. 全モデル・全タスクのレイヤー別精度比較（1つの大きなグラフ）
2. 全モデル・全タスクの手法別精度比較（1つの大きなグラフ）
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def load_data(file_path):
    """pickleファイルからデータを読み込む"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def create_unified_layer_accuracy_graph(data_dir):
    """
    全モデル・全タスクを統合したレイヤー別精度グラフを作成
    1つの大きなグラフに全ての情報を集約
    """
    # データファイルを取得
    model_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl') and 'dataset_specific' not in f]
    
    plt.figure(figsize=(20, 12))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    plot_index = 0
    
    for model_file in model_files:
        model_name = model_file.replace('.pkl', '')
        dataset_specific_file = f"{model_name}_dataset_specific.pkl"
        
        # ノーマルプロンプトデータを読み込み
        normal_data = load_data(os.path.join(data_dir, model_file))
        
        # 拡張プロンプトデータを読み込み（存在する場合）
        extended_data = None
        if os.path.exists(os.path.join(data_dir, dataset_specific_file)):
            extended_data = load_data(os.path.join(data_dir, dataset_specific_file))
        
        # 各タスクの平均レイヤー別精度を計算
        all_layers = set()
        task_layer_accs = {}
        
        for task_name in normal_data.keys():
            normal_layer_acc = normal_data[task_name]['tv_dev_accruacy_by_layer']
            all_layers.update(normal_layer_acc.keys())
            task_layer_accs[task_name] = normal_layer_acc
        
        # レイヤー別の平均精度を計算
        layers = sorted(all_layers)
        avg_normal_accuracies = []
        avg_extended_accuracies = []
        
        for layer in layers:
            normal_accs = [task_layer_accs[task][layer] for task in task_layer_accs.keys() if layer in task_layer_accs[task]]
            avg_normal_accuracies.append(np.mean(normal_accs))
            
            if extended_data:
                extended_accs = []
                for task_name in extended_data.keys():
                    if layer in extended_data[task_name]['tv_dev_accruacy_by_layer']:
                        extended_accs.append(extended_data[task_name]['tv_dev_accruacy_by_layer'][layer])
                avg_extended_accuracies.append(np.mean(extended_accs) if extended_accs else 0)
        
        # ノーマルプロンプトをプロット
        color = colors[plot_index % len(colors)]
        marker = markers[plot_index % len(markers)]
        
        plt.plot(layers, avg_normal_accuracies, 
                color=color, marker=marker, linestyle='-', 
                label=f'{model_name} (Normal)', linewidth=2, markersize=6)
        
        # 拡張プロンプトをプロット（存在する場合）
        if extended_data and any(avg_extended_accuracies):
            plt.plot(layers, avg_extended_accuracies, 
                    color=color, marker=marker, linestyle='--', 
                    label=f'{model_name} (Extended)', linewidth=2, markersize=6, alpha=0.7)
        
        plot_index += 1
    
    plt.xlabel('Layer', fontsize=14)
    plt.ylabel('Average Accuracy', fontsize=14)
    plt.title('All Models - Task Vector Development Accuracy by Layer\n(Averaged across all tasks)', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # グラフを保存
    output_filename = "unified_layer_accuracy_comparison.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"保存完了: {output_filename}")

def create_unified_method_comparison_graph(data_dir):
    """
    モデルごとかつタスクごとの詳細な手法別精度比較グラフを作成
    各モデルの各タスクでの精度を個別に表示
    ノーマルプロンプトと拡張プロンプトのbaselineを別々に表示
    """
    # データファイルを取得
    model_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl') and 'dataset_specific' not in f]
    
    # 全データを収集
    all_data = []
    
    for model_file in model_files:
        model_name = model_file.replace('.pkl', '')
        dataset_specific_file = f"{model_name}_dataset_specific.pkl"
        
        # ノーマルプロンプトデータを読み込み
        normal_data = load_data(os.path.join(data_dir, model_file))
        
        # 拡張プロンプトデータを読み込み（存在する場合）
        extended_data = None
        if os.path.exists(os.path.join(data_dir, dataset_specific_file)):
            extended_data = load_data(os.path.join(data_dir, dataset_specific_file))
        
        # 各タスクのデータを収集
        for task_name in normal_data.keys():
            task_data = normal_data[task_name]
            
            row_data = {
                'model': model_name,
                'task': task_name,
                'icl': task_data['icl_accuracy'],
                'normal_baseline': task_data['baseline_accuracy'],
                'normal_tv': task_data['tv_accuracy'],
                'extended_baseline': extended_data[task_name]['baseline_accuracy'] if extended_data and task_name in extended_data else None,
                'extended_tv': extended_data[task_name]['tv_accuracy'] if extended_data and task_name in extended_data else None
            }
            all_data.append(row_data)
    
    # モデルとタスクの組み合わせを作成
    models = sorted(list(set([d['model'] for d in all_data])))
    
    # タスクを指定された順序で並べる
    task_order = ['translation_en_fr', 'translation_fr_en', 'translation_en_es', 'translation_es_en', 'translation_en_ja', 'translation_ja_en']
    available_tasks = list(set([d['task'] for d in all_data]))
    tasks = [task for task in task_order if task in available_tasks]
    # 指定されていない追加のタスクがあれば最後に追加
    for task in available_tasks:
        if task not in tasks:
            tasks.append(task)
    
    # グラフを作成（大きなサイズで詳細表示）
    fig, ax = plt.subplots(1, 1, figsize=(28, 12))
    
    # モデル×タスクの組み合わせでX軸ラベルを作成
    x_labels = []
    x_positions = []
    x_pos = 0
    
    # 手法を6つに拡張（baselineを2つに分ける）
    methods = ['ICL', 'Normal Baseline', 'Extended Baseline', 'Normal TV', 'Extended TV']
    colors = ['gray', 'lightpink', 'fuchsia', 'lightskyblue', 'blue']
    width = 0.15
    
    # 各手法のデータを準備
    method_data = {method: [] for method in methods}
    
    for model in models:
        for task in tasks:
            # 該当するデータを検索
            data_point = next((d for d in all_data if d['model'] == model and d['task'] == task), None)
            
            if data_point:
                method_data['ICL'].append(data_point['icl'])
                method_data['Normal Baseline'].append(data_point['normal_baseline'])
                method_data['Extended Baseline'].append(data_point['extended_baseline'] if data_point['extended_baseline'] is not None else 0)
                method_data['Normal TV'].append(data_point['normal_tv'])
                method_data['Extended TV'].append(data_point['extended_tv'] if data_point['extended_tv'] is not None else 0)
                
                x_labels.append(f"{model}\n{task}")
                x_positions.append(x_pos)
                x_pos += 1
    
    # 各手法の棒グラフを描画
    x_array = np.arange(len(x_positions))
    
    for i, method in enumerate(methods):
        values = method_data[method]
        bars = ax.bar(x_array + i * width, values, width, label=method, color=colors[i], alpha=0.8)
        
        # 値をバーの上に表示（小さなフォントで）
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                       f'{val:.2f}', ha='center', va='bottom', fontsize=6, rotation=90)
    
    # X軸の設定
    ax.set_xlabel('Model - Task', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Detailed Accuracy Comparison by Model and Task\n(Including Separate Baselines for Normal and Extended Prompts)', fontsize=16)
    ax.set_xticks(x_array + width * 2)
    ax.set_xticklabels(x_labels, fontsize=8, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # モデル間の区切り線を追加
    model_boundaries = []
    current_pos = 0
    for model in models:
        model_task_count = len([d for d in all_data if d['model'] == model])
        current_pos += model_task_count
        if current_pos < len(x_positions):
            model_boundaries.append(current_pos - 0.5)
    
    for boundary in model_boundaries:
        ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    
    # グラフを保存
    output_filename = "unified_method_comparison.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"保存完了: {output_filename}")

def create_comprehensive_summary_table(data_dir):
    """
    全データの包括的なサマリーテーブルを作成
    ノーマルプロンプトと拡張プロンプトのbaselineを別々に表示
    """
    # データファイルを取得
    model_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl') and 'dataset_specific' not in f]
    
    # 全データを収集
    summary_data = []
    
    # タスクを指定された順序で並べる
    task_order = ['translation_en_fr', 'translation_fr_en', 'translation_en_es', 'translation_es_en', 'translation_en_ja', 'translation_ja_en']
    
    for model_file in sorted(model_files):
        model_name = model_file.replace('.pkl', '')
        dataset_specific_file = f"{model_name}_dataset_specific.pkl"
        
        # ノーマルプロンプトデータを読み込み
        normal_data = load_data(os.path.join(data_dir, model_file))
        
        # 拡張プロンプトデータを読み込み（存在する場合）
        extended_data = None
        if os.path.exists(os.path.join(data_dir, dataset_specific_file)):
            extended_data = load_data(os.path.join(data_dir, dataset_specific_file))
        
        # タスクを指定された順序で処理
        available_tasks = list(normal_data.keys())
        ordered_tasks = [task for task in task_order if task in available_tasks]
        # 指定されていない追加のタスクがあれば最後に追加
        for task in available_tasks:
            if task not in ordered_tasks:
                ordered_tasks.append(task)
        
        # 各タスクのデータを収集（指定された順序で）
        for task_name in ordered_tasks:
            task_data = normal_data[task_name]
            
            summary_data.append({
                'Model': model_name,
                'Task': task_name,
                'ICL': f"{task_data['icl_accuracy']:.3f}",
                'Normal Baseline': f"{task_data['baseline_accuracy']:.3f}",
                'Extended Baseline': f"{extended_data[task_name]['baseline_accuracy']:.3f}" if extended_data and task_name in extended_data else "N/A",
                'Normal TV': f"{task_data['tv_accuracy']:.3f}",
                'Extended TV': f"{extended_data[task_name]['tv_accuracy']:.3f}" if extended_data and task_name in extended_data else "N/A"
            })
    
    # テーブルとして表示・保存
    print("\n=== 全データサマリーテーブル ===")
    print(f"{'Model':<12} {'Task':<20} {'ICL':<8} {'N.Base':<8} {'E.Base':<8} {'N.TV':<8} {'E.TV':<8}")
    print("-" * 90)
    
    for row in summary_data:
        print(f"{row['Model']:<12} {row['Task']:<20} {row['ICL']:<8} {row['Normal Baseline']:<8} {row['Extended Baseline']:<8} {row['Normal TV']:<8} {row['Extended TV']:<8}")
    
    # CSVファイルとして保存
    import csv
    with open('summary_table.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Model', 'Task', 'ICL', 'Normal Baseline', 'Extended Baseline', 'Normal TV', 'Extended TV']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_data:
            writer.writerow(row)
    
    print(f"\n保存完了: summary_table.csv")

def main():
    """メイン関数"""
    # データディレクトリ
    data_dir = "."  # 現在のディレクトリ
    
    print("=== camera_readyデータから統合グラフを作成中 ===")
    print()
    
    # 1. 統合レイヤー別精度グラフを作成
    print("1. 統合レイヤー別精度グラフを作成中...")
    create_unified_layer_accuracy_graph(data_dir)
    print()
    
    # 2. 統合手法比較グラフを作成
    print("2. 統合手法比較グラフを作成中...")
    create_unified_method_comparison_graph(data_dir)
    print()
    
    print("=== 全ての統合グラフ作成が完了しました ===")
    print("作成されたファイル:")
    output_files = [f for f in os.listdir(".") if f.endswith('.png') or f.endswith('.csv')]
    for output_file in sorted(output_files):
        if output_file.startswith(('unified_', 'summary_')):
            print(f"  - {output_file}")

if __name__ == "__main__":
    main()
