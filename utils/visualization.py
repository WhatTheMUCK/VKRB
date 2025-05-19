import inspect
import numpy as np
import matplotlib.pyplot as plt

from utils.model_evaluation import evaluate_model_on_slices, evaluate_model_on_ranges


def plot_model_results(X, y, slice_sizes, models, random_seed=42):
    """
    Рисует графики точности для списка моделей.
    
    Параметры:
        X: DataFrame с признаками.
        y: Series с целевой переменной.
        slice_sizes: Список размеров срезов (например, [5, 10, 15]).
        models: Список экземпляров моделей (например, [RandomForestClassifier(), LogisticRegression()]).
        random_seed: Значение для random_state (по умолчанию 42).
    """
    frame = inspect.currentframe()
    try:
        for name, value in frame.f_back.f_locals.items():
            if value is X:
                dataset_name = name
                break
        else:
            dataset_name = "X"
    finally:
        del frame

    print(f'Сид генерации: {random_seed}')
    for model in models:
        model_name = type(model).__name__
        results = evaluate_model_on_slices(X, y, slice_sizes, model, random_seed)
        
        labels = [str(size) for size in slice_sizes]
        accuracies_start = [results[f"First {size}"] for size in slice_sizes]
        accuracies_end = [results[f"Last {size}"] for size in slice_sizes]
        
        max_accuracy = max(accuracies_start + accuracies_end)
        min_accuracy = min(accuracies_start + accuracies_end)
        
        x = np.arange(len(slice_sizes))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        bars_start = ax.bar(x - width/2, accuracies_start, width, label='Первые столбцы', color='blue')
        bars_end = ax.bar(x + width/2, accuracies_end, width, label='Последние столбцы', color='orange')

        for bar in bars_start + bars_end:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', 
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')

        ax.axhline(max_accuracy, color='green', linestyle='--', linewidth=1, label=f'Максимум: {max_accuracy:.6f}')
        ax.axhline(min_accuracy, color='red', linestyle='--', linewidth=1, label=f'Минимум: {min_accuracy:.6f}')

        ax.text(-0.1, 1.1, dataset_name, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))

        ax.set_title(f"Accuracy {model_name} на различных срезах")
        ax.set_xlabel("Количество столбцов")
        ax.set_ylabel("Accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(loc='lower right')

        plt.tight_layout()
        plt.show()


def plot_model_results_for_ranges(X, y, ranges, models, random_seed=42, mode='base'):
    """
    Рисует графики точности для списка моделей на основе заданных диапазонов.
    
    Параметры:
        X: DataFrame с признаками.
        y: Series с целевой переменной.
        ranges: Список кортежей диапазонов (например, [(0, 5), (5, 10), (10, 15)]).
        models: Список экземпляров моделей (например, [RandomForestClassifier(), LogisticRegression()]).
        random_seed: Значение для random_state (по умолчанию 42).
    """
    frame = inspect.currentframe()
    try:
        for name, value in frame.f_back.f_locals.items():
            if value is X:
                dataset_name = name
                break
        else:
            dataset_name = "X"
    finally:
        del frame

    print(f'Сид генерации: {random_seed}')
    for model in models:
        model_name = type(model).__name__
        results = evaluate_model_on_ranges(X, y, ranges, model, random_seed, mode)
        
        labels = [f"{start}-{end}" for start, end in ranges]
        accuracies_range = [results[f"Range {start}-{end}"] for start, end in ranges]

        # Если режим 'extended', добавляем данные для остальных диапазонов
        if mode == 'extended':
            accuracies_other = [results[f"Other {start}-{end}"] for start, end in ranges]
            max_accuracy = max(accuracies_range + accuracies_other)
            min_accuracy = min(accuracies_range + accuracies_other)
        else:
            accuracies_other = None
            max_accuracy = max(accuracies_range)
            min_accuracy = min(accuracies_range)
        
        x = np.arange(len(ranges))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        bars_range = ax.bar(x - width/2, accuracies_range, width, label='Заданный диапазон', color='orange')
        bars = bars_range

        # Если режим 'expand', добавляем столбцы для остальных данных
        if mode == 'extended':
            bars_other = ax.bar(x + width/2, accuracies_other, width, label='Остальные данные', color='blue')
            bars += bars_other

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', 
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')

        ax.axhline(max_accuracy, color='green', linestyle='--', linewidth=1, label=f'Максимум: {max_accuracy:.6f}')
        ax.axhline(min_accuracy, color='red', linestyle='--', linewidth=1, label=f'Минимум: {min_accuracy:.6f}')

        ax.text(-0.1, 1.1, dataset_name, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))

        ax.set_title(f"Accuracy {model_name} на различных диапазонах")
        ax.set_xlabel("Диапазоны")
        ax.set_ylabel("Accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(loc='lower right')

        plt.tight_layout()
        plt.show()