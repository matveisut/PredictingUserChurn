import pandas as pd
import matplotlib.pyplot as plt

def change_dtype_col_to_numeric(df, col_name):
    # Шаг 1: Преобразуем столбец в числа, неконвертируемые значения станут NaN
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

    # Заполняем средними
    mean_value = df[col_name].mean()
    df[col_name] = df[col_name].fillna(mean_value)
    
    
def plot_stacked_bars(df, categorical_cols, target_col):
    n_cols = 2
    n_rows = (len(categorical_cols) + 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, n_rows*3))
    axes = axes.flatten()
    
    for i, col in enumerate(categorical_cols):
        if i < len(axes):
            # Создаем кросс-таблицу
            cross_tab = pd.crosstab(df[col], df[target_col])
            
            cross_tab.plot(kind='bar', stacked=True, ax=axes[i])
            axes[i].set_title(f'{col} vs {target_col}')
            axes[i].set_ylabel('count')
            axes[i].tick_params(axis='x', rotation=45)
 
    
    plt.tight_layout()
    plt.show()