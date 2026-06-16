import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def prepare_data(df, save=True):
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values
    y = df["species"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if save:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)

        pd.DataFrame(X_train_scaled, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]).to_csv(
            os.path.join(PROCESSED_DIR, "X_train.csv"), index=False
        )
        pd.DataFrame(X_test_scaled, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]).to_csv(
            os.path.join(PROCESSED_DIR, "X_test.csv"), index=False
        )
        pd.DataFrame(y_train, columns=["species"]).to_csv(
            os.path.join(PROCESSED_DIR, "y_train.csv"), index=False
        )
        pd.DataFrame(y_test, columns=["species"]).to_csv(
            os.path.join(PROCESSED_DIR, "y_test.csv"), index=False
        )
        joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
