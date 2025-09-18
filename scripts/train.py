# This script needs these libraries to be installed:
#   numpy, sklearn

import wandb
from wandb.sklearn import plot_precision_recall, plot_feature_importances
from wandb.sklearn import plot_class_proportions, plot_learning_curve, plot_roc

def log_wandb_sklearn(model, test_size, X_train, y_train, X_test, y_test):
    model_params = model.get_params()
    y_probas = model.predict_proba(X_test)
    labels = [0,1]
    
    # start a new wandb run and add your model hyperparameters
    wandb.init(project='churn_predict', config=model_params)

    # Add additional configs to wandb
    wandb.config.update({"test_size" : test_size})

    # log additional visualisations to wandb
    plot_class_proportions(y_train, y_test, labels)
    plot_learning_curve(model, X_train, y_train)
    plot_roc(y_test, y_probas, labels)
    plot_precision_recall(y_test, y_probas, labels)
    plot_feature_importances(model)

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()

