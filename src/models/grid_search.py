import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1]
}

grid = GridSearchCV(GradientBoostingRegressor(), param_grid, cv=3, scoring='r2')
grid.fit(X_train, y_train)

with open("models/best_params.pkl", "wb") as f:
    pickle.dump(grid.best_params_, f)
