import yaml
import pandas as pd

# Загрузка конфигурации из YAML файла
def get_raw_data_path_and_target_name():
    with open('../configs/data_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Извлечение параметров
    data_path = config['data']['path']
    target_column = config['data']['target']

    return (data_path, target_column)
