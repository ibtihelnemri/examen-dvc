import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor

X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

with open("models/best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

model = GradientBoostingRegressor(**best_params)
model.fit(X_train, y_train)

with open("models/gbr_model.pkl", "wb") as f:
    pickle.dump(model, f)
