   
import click
import yaml
import os
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import config
import pandas as pd
import config

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import os
from dotenv import load_dotenv
import wandb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from train import train_and_log

def get_yaml(path):
    with open(path, 'r') as file:
        params = yaml.safe_load(file)
    return params


@click.command()
@click.option("--model_config", default="configs/model_config.yaml", help="Path to config")
def main(model_config):
    params = get_yaml(model_config)
    model_type = params['model_type']
    model_parametrs = params['parameters']

    path = config.get_processed_data_path(version = 'processed_path_processed_path_without_unimportant')
    SEED = int(config.get_seed())
    df = pd.read_csv(path)
    df.drop(['Unnamed: 0'], axis = 1, inplace= True)
    
    logreg = LogisticRegression(max_iter = 150)
    tree = DecisionTreeClassifier(max_depth = 5)
    rf = RandomForestClassifier()
    grad_boosting = GradientBoostingClassifier()
    knn = KNeighborsClassifier()
    lgbm = LGBMClassifier()
    xgb = XGBClassifier()
    catboost = CatBoostClassifier()


    models = [logreg, tree, rf, grad_boosting, knn, lgbm, xgb, catboost ]
    names = [model.__class__.__name__ for model in models]

    models_dict = dict(zip(names, models))
    
    model = models_dict[model_type]

    model.set_params(**model_parametrs)
    load_dotenv()  

    api_key = os.getenv('WANDB_API_KEY')
    if api_key:
        wandb.login(key=api_key)
        print("Successfully logged in to W&B")
    else:
        print("Please set WANDB_API_KEY in .env file")
        

    X = df.drop(['Churn'], axis = 1)
    y = df.Churn


    test_size = 0.2


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=SEED)
    scaler = StandardScaler()
    scale_columns = ['ClientPeriod', 'MonthlySpending',  'Spending_Change_Ratio']

    X_train[scale_columns] = scaler.fit_transform(X_train[scale_columns])
    X_test[scale_columns] = scaler.transform(X_test[scale_columns])
        
    train_and_log(model_type, model, X_train, y_train, X_test, y_test)

if __name__ == "__main__":

    main()
    