import yaml
import pandas as pd

# Загрузка конфигурации из YAML файла
def get_raw_data_path_and_target_name(path = './configs/data_config.yaml'):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)

    # Извлечение параметров
    data_path = config['data']['raw_path']
    target_column = config['data']['target']

    return (data_path, target_column)

# Загрузка конфигурации из YAML файла
def get_processed_data_path(version = 0, path = './configs/data_config.yaml'):
    """ 
    2 версии
    0: это просто версия после EDA
    processed_path_processed_path_without_unimportant: версия после удаления признаков
    """
    
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
        
        
    if version == 0:
        processed_path = config['data']['processed_path']
        
    elif version == 'processed_path_processed_path_without_unimportant':
        processed_path = config['data']['processed_path_processed_path_without_unimportant']
        
    else:
        return None
        
    return processed_path


def get_seed(path = './configs/data_config.yaml'):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)

    seed = config['seed']
    return seed