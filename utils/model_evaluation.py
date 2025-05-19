import numpy as np
from sklearn.model_selection import cross_val_score
import pandas as pd

def evaluate_model_on_slices(X, y, slice_sizes, model, random_seed=42):
    """
    Оценивает работу модели на различных срезах данных.
    
    Параметры:
        X: DataFrame с признаками.
        y: Series с целевой переменной.
        slice_sizes: Список размеров срезов (например, [5, 10, 15]).
        model: Экземпляр модели (например, RandomForestClassifier, LogisticRegression и т.д.).
        random_seed: Значение для random_state (по умолчанию 42).
    
    Возвращает:
        scores: Словарь с результатами кросс-валидации для каждого среза.
    """
    if hasattr(model, 'random_state'):
        model.random_state = random_seed
    
    scores = {}
    
    for size in slice_sizes:
        X_slice_start = X.iloc[:, :size]
        X_slice_end = X.iloc[:, -size:]

        cv_scores_start = cross_val_score(model, X_slice_start, y, cv=5, scoring='accuracy')
        cv_scores_end = cross_val_score(model, X_slice_end, y, cv=5, scoring='accuracy')
        
        scores[f"First {size}"] = np.mean(cv_scores_start)
        scores[f"Last {size}"] = np.mean(cv_scores_end)
    
    return scores


def evaluate_model_on_ranges(X, y, ranges, model, random_seed=42, mode='base'):
    """
    Оценивает работу модели на различных диапазонах данных.
    
    Параметры:
        X: DataFrame с признаками.
        y: Series с целевой переменной.
        ranges: Список кортежей диапазонов (например, [(0, 5), (5, 10), (10, 15)]).
        model: Экземпляр модели (например, RandomForestClassifier, LogisticRegression и т.д.).
        random_seed: Значение для random_state (по умолчанию 42).
    
    Возвращает:
        scores: Словарь с результатами кросс-валидации для каждого диапазона.
    """
    if hasattr(model, 'random_state'):
        model.random_state = random_seed
    
    scores = {}

    if mode not in ['base', 'extended']:
        raise ValueError(f"Неподдерживаемый режим: {mode}. Используйте 'base' или 'extended'.")
    
    for start, end in ranges:
        # Проверяем, что диапазон корректен
        if start > end:
            raise ValueError(f"Некорректный диапазон: start={start}, end={end}. Убедитесь, что start <= end.")
        
        # Заданный диапазон
        X_range = X.iloc[:, start-1:end]

        # Обучение модели на заданном диапазоне
        cv_scores_range = cross_val_score(model, X_range, y, cv=5, scoring='accuracy')
        scores[f"Range {start}-{end}"] = np.mean(cv_scores_range)

        if mode == 'extended':
            X_other = pd.concat([X.iloc[:, :start-1], X.iloc[:, end:]], axis=1)
            cv_scores_other = cross_val_score(model, X_other, y, cv=5, scoring='accuracy')
            scores[f"Other {start}-{end}"] = np.mean(cv_scores_other)
    
    return scores