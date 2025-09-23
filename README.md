# PredictingUserChurn

## Описание проекта

Этот проект посвящен **прогнозированию оттока пользователей** (User Churn) с использованием различных моделей машинного обучения.

## Структура репозитория

  * `configs/`: Файлы конфигурации, включая настройки модели.
  * `notebooks/`: Jupyter-ноутбуки для исследования данных, обучения моделей и экспериментов.
  * `scripts/`: Скрипты для запуска обучения и предсказаний.
  * `.gitignore`: Файл для игнорирования служебных файлов Git.
  * `LICENSE`: Лицензия на использование проекта.
  * `README.md`: Этот файл.
  * `requirements.txt`: Список зависимостей проекта.

## Начало работы

### Загрузка данных

Исходные данные можно скачать по следующей ссылке:
[https://www.kaggle.com/competitions/advanced-dls-spring-2021](https://www.google.com/search?q=https://www.kaggle.com/competitions/advanced-dls-spring-2021)

### Установка зависимостей

Чтобы запустить проект, сначала установите все необходимые библиотеки из файла `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Запуск проекта
Перед тем как это запускать вам понадобятся предобработанные данные, которые появятся в data/processed после выполнения ноутбука main и experiments(для них нужны raw данные с kaggle)

В  data_config.yaml указаываются пути к данным, raw откуда будет загружены данные с kaggle. и  processed - куда будут сохранены обработанные в ноутбуках данные

в model_config.yaml указываются настройки модели, которую будете тренировать. Возможные модели: LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, KNeighborsClassifier, LGBMClassifier, XGBClassifier, CatBoostClassifier

Вы можете запустить обучение модели с использованием файла конфигурации, выполнив следующую команду:

```bash
python scripts/main.py --model_config configs/model_config.yaml
```


## Лицензия

Этот проект распространяется под лицензией **MIT**. Подробности можно найти в файле `LICENSE`.
