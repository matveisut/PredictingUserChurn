# This script needs these libraries to be installed:
#   numpy, sklearn

import wandb
from wandb.sklearn import plot_precision_recall, plot_feature_importances
from wandb.sklearn import plot_class_proportions, plot_learning_curve, plot_roc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import yaml
import os
import numpy as np


def log_wandb_sklearn(model, X_train, y_train, X_test, y_test, data_info = None, name = None):
    """
    Логирует метрики модели в wandb
    """
    
    if name != None:
        name_model = name
    else:
        name_model = model.__class__.__name__
    
    model_params = model.get_params()
    
    
    model_params['algorithm'] = name_model
    
    y_probas = model.predict_proba(X_test)
    labels = [0,1]
    
    # start a new wandb run and add your model hyperparameters
    wandb.init(project='churn_predict', config=model_params, name = name_model)

    # Add additional configs to wandb
    test_size = round(len(X_test) / (len(X_train) + len(X_test)), 2)
    wandb.config.update({"test_size" : test_size, 'data_info': data_info})

    # log additional visualisations to wandb
    plot_class_proportions(y_train, y_test, labels)
    plot_learning_curve(model, X_train, y_train)
    plot_roc(y_test, y_probas, labels)
    plot_precision_recall(y_test, y_probas, labels)
    plot_feature_importances(model, feature_names = list(X_train.columns))

    # [optional] finish the wandb run, necessary in notebooks
    
def train_and_log(name, model, X_train, y_train, X_test, y_test, save_dir = '.\configs\experiments', reports = None, roc_aucs = None):
    """
    
    принимает на вход модель, имя конфигурации, данные и два словаря, в которые будут добавлены 
    по ключу name метрики по classification report и roc auc в соответсвующие словари.
    А также выводит эти метрики
    
    Также добавляет параметры модели в yaml файл 
    
    Сохраняет эксперимент в save_dir
    """
    print("=" * 50, "model: ", name ,"=" * 50)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if reports != None:
        report = classification_report(y_pred = y_pred, y_true = y_test, output_dict=True)
        reports[name] = report
    
    
    probas = model.predict_proba(X_test)[:, 1]
    roc_auc_scor = roc_auc_score(y_test, probas)
    
    if roc_aucs != None:
        roc_aucs[name] = roc_auc_scor
    
    print(classification_report(y_pred = y_pred, y_true = y_test))
    print('roc_auc =', roc_auc_scor)

    log_wandb_sklearn(model, X_train, y_train, X_test, y_test)
    
    quick_save_model(model, save_dir= save_dir)





def quick_save_model(model, feature_names=None, metrics=None,  save_dir= '.\configs\experiments'):
    """Сохранение с очисткой данных"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Находим следующий номер
    existing = [f for f in os.listdir(save_dir) if f.startswith('exp') and f.endswith('.yaml')]
    next_num = len(existing) + 1
    exp_name = f"exp{next_num}"
    filename = os.path.join(save_dir, f"{exp_name}.yaml")
    
    # ОЧИСТКА ДАННЫХ: преобразуем все в простые типы
    def clean_data(data):
        """Рекурсивно очищает данные от сложных объектов"""
        if isinstance(data, (np.integer, np.int64, np.int32)):
            return int(data)
        elif isinstance(data, (np.floating, np.float64, np.float32)):
            return float(data)
        elif isinstance(data, (np.ndarray, list)):
            return [clean_data(item) for item in data]
        elif isinstance(data, dict):
            return {str(key): clean_data(value) for key, value in data.items()}
        elif hasattr(data, 'tolist'):  # для numpy arrays
            return data.tolist()
        elif hasattr(data, '__dict__'):  # для объектов
            return clean_data(data.__dict__)
        else:
            return data
    
    # Подготовка данных с очисткой
    data = {
        'model_type': model.__class__.__name__,
        'parameters': clean_data({k: v for k, v in model.get_params().items() if v is not None})
    }
    
    if feature_names is not None:
        # Преобразуем Index/pandas объекты в простой список
        if hasattr(feature_names, 'tolist'):
            data['feature_names'] = clean_data(feature_names.tolist())
        else:
            data['feature_names'] = clean_data(list(feature_names))

    
    if metrics is not None:
        data['metrics'] = clean_data(metrics)
    
    # Сохраняем с безопасным дампом
    with open(filename, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ Эксперимент сохранен: {filename}")
    return exp_name
