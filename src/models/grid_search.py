import pandas as pd
import pickle
import yaml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

param_grid = {
    'n_estimators': params['gridsearch']['n_estimators'],
    'max_depth': params['gridsearch']['max_depth'],
    'learning_rate': params['gridsearch']['learning_rate']
}

grid = GridSearchCV(GradientBoostingRegressor(), param_grid, cv=3, scoring='r2')
grid.fit(X_train, y_train)

with open("models/best_params.pkl", "wb") as f:
    pickle.dump(grid.best_params_, f)
