import numpy as np
import pandas as pd
import pywt


def apply_fourier_transform(X):
    """
    Применяет преобразование Фурье к каждой строке X.
    
    Параметры:
        X: DataFrame или массив NumPy.
    
    Возвращает:
        DataFrame с результатами преобразования Фурье.
    """
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    
    fourier_transformed = np.fft.fft(X, axis=1)
    fourier_result = np.abs(fourier_transformed)
    
    return pd.DataFrame(fourier_result, columns=[i for i in range(1, fourier_result.shape[1] + 1)])


def apply_wavelet_transform(X, wavelet='db1', level=1, debug=False):
    """
    Применяет вейвлет-преобразование к каждому пациенту (строке) в X.

    Параметры:
        X: Двумерный массив, где строки - пациенты, столбцы - коэффициенты разложения АЧХ.
        wavelet: Имя вейвлета для применения (по умолчанию 'db1').
        level: Уровень разложения (по умолчанию 1).

    Возвращает:
        DataFrame с преобразованными данными.
    """
    max_level = pywt.dwt_max_level(data_len=X.shape[1], filter_len=pywt.Wavelet(wavelet).dec_len)
    if level > max_level:
        if debug:
            print(f"Уровень разложения {level} слишком высок. Используется максимальный уровень {max_level}.")
        level = max_level

    X_wavelet = []

    for i in range(X.shape[0]):
        coeffs = pywt.wavedec(X.iloc[i, :], wavelet, level=level)
        coeffs_combined = np.concatenate(coeffs)  # Объединяем аппроксимирующие и детализирующие коэффициенты
        X_wavelet.append(coeffs_combined)

    X_wavelet = np.array(X_wavelet)
    column_names = [i for i in range(1, X_wavelet.shape[1] + 1)]
    return pd.DataFrame(X_wavelet, columns=column_names)