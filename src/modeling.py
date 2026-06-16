import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
RANDOM_STATE = 42


def get_param_grids():
    return {
        "Logistic Regression": {
            "estimator": LogisticRegression(solver="lbfgs", random_state=RANDOM_STATE),
            "param_grid": {
                "C": [0.01, 0.1, 1, 10, 100],
                "max_iter": [500, 1000, 2000, 5000],
            },
        },
        "K-Nearest Neighbors": {
            "estimator": KNeighborsClassifier(),
            "param_grid": {
                "n_neighbors": [3, 5, 7, 9, 11],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"],
            },
        },
        "Support Vector Classifier": {
            "estimator": SVC(kernel="rbf", random_state=RANDOM_STATE),
            "param_grid": {
                "C": [0.1, 1, 10, 100],
                "gamma": ["scale", "auto", 0.01, 0.1],
            },
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10],
            },
        },
        "Decision Tree": {
            "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "max_depth": [3, 5, 7, 10, None],
                "min_samples_split": [2, 5, 10],
                "criterion": ["gini", "entropy"],
            },
        },
    }


def train_all_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(
            solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Classifier": CalibratedClassifierCV(
            SVC(kernel="rbf", random_state=RANDOM_STATE), ensemble=False
        ),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model

    return trained


def train_with_gridsearch(X_train, y_train):
    param_grids = get_param_grids()
    best_models = {}
    best_params_all = {}

    for name, config in param_grids.items():
        gs = GridSearchCV(
            config["estimator"], config["param_grid"],
            cv=5, scoring="accuracy", n_jobs=-1, refit=True, return_train_score=False
        )
        gs.fit(X_train, y_train)

        if name == "Support Vector Classifier":
            best_models[name] = CalibratedClassifierCV(
                SVC(kernel="rbf", random_state=RANDOM_STATE, **{
                    k: v for k, v in gs.best_params_.items() if k != "ensemble"
                }), ensemble=False
            )
            best_models[name].fit(X_train, y_train)
        else:
            best_models[name] = gs.best_estimator_

        cv_df = pd.DataFrame(gs.cv_results_)
        best_params_all[name] = {
            "best_params": gs.best_params_,
            "best_cv_score": gs.best_score_,
            "cv_results_df": cv_df,
            "n_combos": len(cv_df),
        }

    return best_models, best_params_all


def cross_validate_models(models, X_train, y_train, cv=5):
    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        cv_results[name] = {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "scores": scores,
        }
    return cv_results


def save_best_model(models, X_test, y_test):
    best_name = None
    best_score = -1
    best_model = None

    for name, model in models.items():
        score = model.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_name = name
            best_model = model

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))

    return best_name, best_score
