import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

# On ne garde que les colonnes numériques
X_train = X_train.select_dtypes(include=["float64", "int64"])
X_test = X_test.select_dtypes(include=["float64", "int64"])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv("data/processed/X_train_scaled.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv("data/processed/X_test_scaled.csv", index=False)
