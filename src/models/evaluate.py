import pandas as pd
import pickle
import json
from sklearn.metrics import mean_squared_error, r2_score

X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

with open("models/gbr_model.pkl", "rb") as f:
    model = pickle.load(f)

y_pred = model.predict(X_test)
pred_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
pred_df.to_csv("data/prediction.csv", index=False)

scores = {
    "mse": mean_squared_error(y_test, y_pred),
    "r2": r2_score(y_test, y_pred)
}

with open("metrics/scores.json", "w") as f:
    json.dump(scores, f, indent=4)
