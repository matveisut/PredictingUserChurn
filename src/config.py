import yaml
import pandas as pd

# Загрузка конфигурации из YAML файла
def get_raw_data_path_and_target_name():
    with open('../configs/data_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Извлечение параметров
    data_path = config['data']['raw_path']
    target_column = config['data']['target']

    return (data_path, target_column)

# Загрузка конфигурации из YAML файла
def get_processed_data_path():
    with open('../configs/data_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    processed_path = config['data']['processed_path']
    return processed_path